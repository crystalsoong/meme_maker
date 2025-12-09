# predict_image.py

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import argparse
import os
import json
import torch.nn.functional as F

# ---------------------------------------------------------------------
# --- MODEL ARCHITECTURE (CRITICAL: PASTE ENTIRE CLASSES BELOW) ---
# ---------------------------------------------------------------------

# You MUST paste the complete classes here:
# - TransformerBlock

# -------------- Transformer building blocks --------------
class TransformerBlock(nn.Module):
    def __init__(self, d_embed, num_heads, dropout_rate=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_embed, num_heads)
        self.attn_norm = nn.LayerNorm(d_embed)
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.ff_linear = nn.Linear(d_embed, 4 * d_embed)
        self.ff_linear2 = nn.Linear(4 * d_embed, d_embed)
        self.ff_norm = nn.LayerNorm(d_embed)
        self.relu = nn.ReLU()
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask):
        # unbatched seq-first: (seq_len, d_embed) OR batched (batch, seq_len, d_embed)
        if x.dim() == 2:
            x_b = x.unsqueeze(1)
            attn_out_b, _ = self.multihead_attn(x_b, x_b, x_b, attn_mask=attn_mask)
            attn_out = attn_out_b.squeeze(1)
            x = x + attn_out
            x = self.attn_norm(x)
            ff = self.ff_linear(x)
            ff = self.relu(ff)
            ff = self.ff_linear2(ff)
            x = x + ff
            x = self.ff_norm(x)
            return x
        elif x.dim() == 3:
            x_t = x.transpose(0, 1)
            attn_out_t, _ = self.multihead_attn(x_t, x_t, x_t, attn_mask=attn_mask)
            h = x_t + self.attn_dropout(attn_out_t)
            h = self.attn_norm(h)
            ff = self.ff_linear(h)
            ff = self.relu(ff)
            ff = self.ff_linear2(ff)
            h = h + self.ff_dropout(ff)
            h = self.ff_norm(h)
            return h.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported input dim {x.dim()} in TransformerBlock")


# - VisionEncoder

# -------------- Vision encoder --------------
class VisionEncoder(nn.Module):
    def __init__(self, d_embed, num_heads, n_blocks, img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Patch Embedding: Convolution with stride=patch_size acts as patch extraction
        self.patch_embedding = nn.Conv2d(in_channels, d_embed, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Embedding(self.num_patches, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embedding(x)                          # (B, d, H', W')
        x = x.flatten(2).transpose(1, 2)                    # (B, num_patches, d)
        
        # Add Positional Embedding
        pos = torch.arange(self.num_patches, device=x.device)
        x = x + self.positional_embedding(pos).unsqueeze(0)
        
        # Pass through Transformer Blocks and normalize
        for b in self.blocks:
            x = b(x, None)
        
        x = self.norm(x)
        # Final aggregation (mean pooling over patches)
        return x.mean(dim=1)  # (B, d)



# - TransformerDecoder

# -------------- Transformer decoder --------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, max_length=512, n_blocks=4):
        super().__init__()
        self.d_embed = d_embed
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, d_embed)
        # Positional embedding size accounts for max_length tokens + 1 image feature slot
        self.pos_embed = nn.Embedding(max_length + 1, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d_embed, vocab_size)

    def forward(self, x, image_features):
        # x: (B, seq_len_text), image_features: (B, d)
        if x.dim() != 2:
            raise ValueError("TransformerDecoder.forward expects x shape (batch, seq_len)")
        B, S = x.shape
        tok_emb = self.embed(x)                              # (B, S, d)
        
        # Prepend Image Feature to the sequence
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0).expand(B, -1)
        img_feat = image_features.unsqueeze(1)               # (B,1,d)
        combined = torch.cat([img_feat, tok_emb], dim=1)     # (B, S+1, d)
        L = combined.shape[1]
        
        # Add Positional Embedding
        if L > self.pos_embed.num_embeddings:
            raise ValueError(f"Caption too long for positional embeddings: {L} > {self.pos_embed.num_embeddings}")
        pos_idx = torch.arange(L, device=combined.device)
        h = combined + self.pos_embed(pos_idx).unsqueeze(0)
        
        # Causal Masking: Blocks attention to future tokens.
        attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=combined.device), diagonal=1)
        # Image feature (index 0) must see all subsequent text tokens
        attn_mask[0, :] = False
        
        for b in self.blocks:
            h = b(h, attn_mask)
        
        # Final projection, skipping the image feature output (index 0)
        logits = self.unembed(h[:, 1:, :])                   # (B, S, V)
        return logits

# - ImageCaptioningModel

# -------------- Full model --------------
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, n_blocks=4, max_length=512,
                 img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        # Initialize the Vision Encoder
        self.vision_encoder = VisionEncoder(d_embed, num_heads, n_blocks, img_size, patch_size, in_channels)
        # Initialize the Transformer Decoder
        self.transformer_decoder = TransformerDecoder(vocab_size, d_embed, num_heads, max_length, n_blocks)

    def forward(self, images, caption_input):
        # 1. Encode the image
        img_feat = self.vision_encoder(images)
        # 2. Decode the caption sequence, conditioned on the image features
        logits = self.transformer_decoder(caption_input, img_feat)
        return logits




# --- Start of required model class definitions ---
# Define D_EMBED, NUM_HEADS, N_BLOCKS, IMG_SIZE, PATCH_SIZE to match training
D_EMBED = 256
NUM_HEADS = 4
N_BLOCKS = 4
IMG_SIZE = 224
PATCH_SIZE = 16

class TransformerBlock(nn.Module):
    def __init__(self, d_embed, num_heads, dropout_rate=0.1):
        super().__init__()
        # ... (PASTE FULL TransformerBlock __init__ and forward methods here)
        self.multihead_attn = nn.MultiheadAttention(d_embed, num_heads, batch_first=False)
        self.attn_norm = nn.LayerNorm(d_embed)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ff_linear = nn.Linear(d_embed, 4 * d_embed)
        self.ff_linear2 = nn.Linear(4 * d_embed, d_embed)
        self.ff_norm = nn.LayerNorm(d_embed)
        self.relu = nn.ReLU()
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.transpose(0, 1)
        attn_out, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        h = x + self.attn_dropout(attn_out)
        h = self.attn_norm(h)
        ff = self.ff_linear(h)
        ff = self.relu(ff)
        ff = self.ff_linear2(ff)
        h = h + self.ff_dropout(ff)
        h = self.ff_norm(h)
        return h.transpose(0, 1)


class VisionEncoder(nn.Module):
    # ... (PASTE FULL VisionEncoder class here)
    def __init__(self, d_embed, num_heads, n_blocks, img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, d_embed, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Embedding(self.num_patches, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        pos = torch.arange(self.num_patches, device=x.device)
        x = x + self.positional_embedding(pos).unsqueeze(0)
        for b in self.blocks:
            x = b(x, None)
        x = self.norm(x)
        return x.mean(dim=1)


class TransformerDecoder(nn.Module):
    # ... (PASTE FULL TransformerDecoder class here)
    def __init__(self, vocab_size, d_embed=D_EMBED, num_heads=NUM_HEADS, max_length=512, n_blocks=N_BLOCKS):
        super().__init__()
        self.d_embed = d_embed
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.pos_embed = nn.Embedding(max_length + 1, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d_embed, vocab_size)

    def forward(self, x, image_features):
        B, S = x.shape
        tok_emb = self.embed(x)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0).expand(B, -1)
        img_feat = image_features.unsqueeze(1)
        combined = torch.cat([img_feat, tok_emb], dim=1)
        L = combined.shape[1]
        pos_idx = torch.arange(L, device=combined.device)
        h = combined + self.pos_embed(pos_idx).unsqueeze(0)
        attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=combined.device), diagonal=1)
        attn_mask[0, :] = False
        for b in self.blocks:
            h = b(h, attn_mask)
        logits = self.unembed(h[:, 1:, :])
        return logits


class ImageCaptioningModel(nn.Module):
    # ... (PASTE FULL ImageCaptioningModel class here)
    def __init__(self, vocab_size, d_embed=D_EMBED, num_heads=NUM_HEADS, n_blocks=N_BLOCKS, max_length=512,
                 img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_channels=3):
        super().__init__()
        self.vision_encoder = VisionEncoder(d_embed, num_heads, n_blocks, img_size, patch_size, in_channels)
        self.transformer_decoder = TransformerDecoder(vocab_size, d_embed, num_heads, max_length, n_blocks)

    def forward(self, images, caption_input):
        img_feat = self.vision_encoder(images)
        logits = self.transformer_decoder(caption_input, img_feat)
        return logits
# --- End of required model class definitions ---

# ---------------------------------------------------------------------
# --- TOKENIZER (Must match training script's construction) ---
# ---------------------------------------------------------------------
class Tokenizer:
    def __init__(self, vocab_file):
        """Initializes the tokenizer by loading the exact vocab mapping."""
        # Check for the existence of the saved vocabulary file
        if not vocab_file or not os.path.exists(vocab_file):
             raise FileNotFoundError(
                 f"ERROR: Saved Tokenizer vocab file not found at: {vocab_file}. "
                 "You must run the training script once to generate this file."
             )

        # Load from the saved vocab file (This is the critical fix)
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
            self.word_to_index = vocab_data['word_to_index']
            # Keys are strings in JSON, convert them back to integers
            self.index_to_word = {int(i): w for i, w in vocab_data['index_to_word'].items()}
            self.vocab = list(self.word_to_index.keys())

    def encode(self, text):
        text = text.lower().strip(".")
        return [self.word_to_index.get(w, self.word_to_index['<unk>']) for w in text.split()]

    def decode(self, ids):
        return ' '.join(self.index_to_word.get(i, '<unk>') for i in ids)

# ---------------------------------------------------------------------
# --- GENERATION FUNCTIONS (Must match training script) ---
# ---------------------------------------------------------------------
@torch.no_grad()
def generate_greedy(model, image_tensor, tokenizer, device, max_len=50, min_len=2, temperature=0.9, top_p=0.9):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    sos = tokenizer.word_to_index['<sos>']
    eos = tokenizer.word_to_index.get('<eos>', None)
    cur = torch.tensor([[sos]], device=device, dtype=torch.long)

    for t in range(max_len):
        logits = model(image_tensor, cur)
        last_logits = logits[:, -1, :] / temperature

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
            filtered_logits = torch.full_like(last_logits, float('-inf'))
            filtered_logits.scatter_(1, sorted_indices, sorted_logits)
            last_logits = filtered_logits

        probs = F.softmax(last_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        nxt = nxt.to(device).long()
        cur = torch.cat([cur, nxt], dim=1)

        if eos is not None and nxt.numel() == 1 and nxt.item() == eos and cur.shape[1]-1 >= min_len:
            break

    seq = cur.squeeze(0).tolist()[1:]
    if eos is not None and eos in seq:
        seq = seq[:seq.index(eos)]
    return tokenizer.decode(seq)

@torch.no_grad()
def generate_beam(model, image_tensor, tokenizer, device, max_len=50, beam_width=5):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    sos = tokenizer.word_to_index['<sos>']
    eos = tokenizer.word_to_index.get('<eos>', None)
    beams = [([sos], 0.0)]
    for step in range(max_len):
        candidates = []
        for seq, score in beams:
            cur = torch.tensor([seq], device=device)
            logits = model(image_tensor, cur)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk_logp, topk_idx = torch.topk(logp, beam_width)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                candidates.append((seq + [idx], score + lp))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if any((b[0][-1] == eos) for b in beams if eos is not None):
            break
    best = beams[0][0][1:]
    if eos is not None and eos in best:
        best = best[:best.index(eos)]
    return tokenizer.decode(best)

# ---------------------------------------------------------------------
# --- CORE INFERENCE LOGIC ---
# ---------------------------------------------------------------------
# In predict_image.py, near the top where paths are defined:
TOKENIZER_VOCAB_PATH = 'data/processed/tokenizer_vocab.json'
MANIFEST_PATH = 'data/processed/imgflip575k_manifest.json' 
# ...

def load_model_and_tokenizer(checkpoint_path, manifest_path, device):
    
    # 1. Initialize the Tokenizer ONLY from the saved vocabulary file
    tokenizer = Tokenizer(vocab_file=TOKENIZER_VOCAB_PATH) 
    
    # 2. Extract necessary values for model initialization
    vocab_size = len(tokenizer.vocab) 
    
    # We still need the manifest to determine the maximum length for the decoder's positional embedding
    with open(manifest_path, 'r') as f:
        image_data = json.load(f)
    captions = [it['caption'] for it in image_data]
    
    # Compute max length using the loaded tokenizer
    max_enc_len = max(len(tokenizer.encode(c)) for c in captions)
    max_length_for_model = max_enc_len + 1 # +1 for the image feature prefix

    

    # 3. Initialize Model 
    d_embed = 256; num_heads = 4; n_blocks = 4
    model = ImageCaptioningModel(vocab_size=vocab_size,
                                 d_embed=d_embed, num_heads=num_heads, n_blocks=n_blocks,
                                 max_length=max_length_for_model).to(device)

    # 4. Load weights 
    ckpt = torch.load(checkpoint_path, map_location=device)
    # --- FIX APPLIED HERE ---
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict)

    return model, tokenizer

def predict_from_path(image_path, model, tokenizer, device):
    # Image transformations MUST match the validation/test transforms used in training!
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device) # (1, C, H, W)

    # Generate captions
    print("\n--- Generating Captions ---")
    greedy_cap = generate_greedy(model, img_tensor, tokenizer, device, max_len=50, temperature=0.9, top_p=0.9)
    print(f"Greedy/Top-P (temp=0.7, p=0.9): {greedy_cap}")

    beam_cap = generate_beam(model, img_tensor, tokenizer, device, max_len=50, beam_width=5)
    print(f"Beam Search (width=10):        {beam_cap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for an input image using the trained model.")
    # Define the required command-line argument for the image path
    parser.add_argument("image_path", type=str, help="Path to the input image file (e.g., ./my_meme.jpg)")
    args = parser.parse_args()

    # --- Configuration (Must match training script) ---
    CHECKPOINT_PATH = 'models/meme_caption_vit.pt'
    MANIFEST_PATH = 'data/processed/imgflip575k_manifest.json'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 1. Load the model and tokenizer
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH, MANIFEST_PATH, device)

        # 2. Run prediction using the provided image path
        predict_from_path(args.image_path, loaded_model, loaded_tokenizer, device)

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure the checkpoint file exists and the manifest file path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Check that the model architecture classes were copied completely and correctly.")