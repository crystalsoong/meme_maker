import os
import json
import random
import math
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import numpy as np

# -------------- Hyper / Paths --------------
TRAIN = True   # <-- Set True to train, False to load and run inference
MANIFEST_PATH = 'data/processed/imgflip575k_manifest.json'   # update to your dataset manifest
CHECKPOINT_PATH = 'models/meme_caption_vit.pt'               # where to save / load model
PLOT_DIR = 'plots'
SEED = 42

# -------------- Utility / plotting --------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_history(history, title="Training History", save_path=None, live=False):
    if live:
        clear_output(wait=True)
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.legend(); plt.grid(True)
    plt.suptitle(title); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

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

# -------------- Vision encoder --------------
class VisionEncoder(nn.Module):
    def __init__(self, d_embed, num_heads, n_blocks, img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, d_embed, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Embedding(self.num_patches, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embedding(x)                          # (B, d, H', W')
        x = x.flatten(2).transpose(1, 2)                    # (B, num_patches, d)
        pos = torch.arange(self.num_patches, device=x.device)
        x = x + self.positional_embedding(pos).unsqueeze(0)
        for b in self.blocks:
            x = b(x, None)
        x = self.norm(x)
        return x.mean(dim=1)  # (B, d)

# -------------- Transformer decoder --------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, max_length=512, n_blocks=4):
        super().__init__()
        self.d_embed = d_embed
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.pos_embed = nn.Embedding(max_length + 1, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d_embed, vocab_size)

    def forward(self, x, image_features):
        # x: (B, seq_len_text), image_features: (B, d)
        if x.dim() != 2:
            raise ValueError("TransformerDecoder.forward expects x shape (batch, seq_len)")
        B, S = x.shape
        tok_emb = self.embed(x)                              # (B, S, d)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0).expand(B, -1)
        img_feat = image_features.unsqueeze(1)               # (B,1,d)
        combined = torch.cat([img_feat, tok_emb], dim=1)     # (B, S+1, d)
        L = combined.shape[1]
        if L > self.pos_embed.num_embeddings:
            raise ValueError(f"Caption too long for positional embeddings: {L} > {self.pos_embed.num_embeddings}")
        pos_idx = torch.arange(L, device=combined.device)
        h = combined + self.pos_embed(pos_idx).unsqueeze(0)
        attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=combined.device), diagonal=1)
        attn_mask[0, :] = False
        for b in self.blocks:
            h = b(h, attn_mask)
        logits = self.unembed(h[:, 1:, :])                   # (B, S, V)
        return logits

# -------------- Full model --------------
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, n_blocks=4, max_length=512,
                 img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.vision_encoder = VisionEncoder(d_embed, num_heads, n_blocks, img_size, patch_size, in_channels)
        self.transformer_decoder = TransformerDecoder(vocab_size, d_embed, num_heads, max_length, n_blocks)

    def forward(self, images, caption_input):
        img_feat = self.vision_encoder(images)
        logits = self.transformer_decoder(caption_input, img_feat)
        return logits

# -------------- Dataset & collate --------------
class ImageCaptionDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, image_transform=None):
        self.image_filenames = image_filenames
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        path = self.image_filenames[idx]
        cap = self.captions[idx]
        img = Image.open(path).convert('RGB')
        if self.image_transform:
            img = self.image_transform(img)
        enc = self.tokenizer.encode(cap)
        inp = [self.tokenizer.word_to_index['<sos>']] + enc
        tgt = enc + [self.tokenizer.word_to_index.get('<eos>', self.tokenizer.word_to_index.get('<unk>'))]
        return img, torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch, padding_idx):
    imgs = [b[0].unsqueeze(0) for b in batch]
    inps = [b[1] for b in batch]
    tgts = [b[2] for b in batch]
    inps_p = pad_sequence(inps, batch_first=True, padding_value=padding_idx)
    tgts_p = pad_sequence(tgts, batch_first=True, padding_value=padding_idx)
    imgs_b = torch.cat(imgs, dim=0)
    return imgs_b, inps_p, tgts_p

# -------------- Evaluation & training helpers --------------
def evaluate_model(model, dataloader, criterion, device, padding_idx, split_name="Test"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for images, inps, tgts in dataloader:
            images = images.to(device)
            inps = inps.to(device)
            tgts = tgts.to(device)
            logits = model(images, inps)
            loss = criterion(logits.permute(0,2,1), tgts)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            mask = (tgts != padding_idx)
            total_correct += (preds == tgts)[mask].sum().item()
            total_tokens += mask.sum().item()
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples if num_samples else 0.0
    acc = total_correct / total_tokens if total_tokens else 0.0
    print(f"  > {split_name} Loss: {avg_loss:.4f}, {split_name} Accuracy: {acc:.4f}")
    model.train()
    return avg_loss, acc

def run_optimizer_experiment(
    model,                     # <-- train the provided model in-place
    tokenizer,
    train_dataloader,
    val_dataloader,
    criterion,
    num_epochs,
    device,
    config,
    display=True,
    checkpoint_path=CHECKPOINT_PATH
):
    """
    Trains the provided model in-place and returns (history, trained_model).
    """
    print(f"--- Starting Experiment: {config['name']} ---")
    model = model.to(device)
    optimizer = config['optimizer_class'](model.parameters(), **config.get('kwargs', {}))
    scheduler = None
    if config.get('scheduler_class') is not None:
        scheduler = config['scheduler_class'](optimizer, **config.get('scheduler_kwargs', {}))
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    padding_idx = tokenizer.word_to_index.get('<pad>', tokenizer.word_to_index.get('<unk>', 0))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} ({config['name']} Training)")
        for images, inps, tgts in pbar:
            images = images.to(device)
            inps = inps.to(device)
            tgts = tgts.to(device)
            optimizer.zero_grad()
            logits = model(images, inps)
            loss = criterion(logits.permute(0,2,1), tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
        avg_train_loss = running_loss / num_samples if num_samples else 0.0
        history['train_loss'].append(avg_train_loss)

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device, padding_idx, split_name=config['name'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        plot_metrics(history, title=f"Experiment: {config['name']}", save_path=os.path.join(PLOT_DIR, f"{config['name']}_history.png"), smooth=False, live=True)

        # Save checkpoint each epoch
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch+1,
                    'history': history},
                   checkpoint_path)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history, model

# -------------- Generation helpers --------------
@torch.no_grad()
def generate_greedy(model, image_tensor, tokenizer, device, max_len=30, min_len=1, temperature=1.0, top_p=0.9):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    sos = tokenizer.word_to_index['<sos>']
    eos = tokenizer.word_to_index.get('<eos>', None)

    # cur shape: (batch=1, seq_len=1)
    cur = torch.tensor([[sos]], device=device, dtype=torch.long)

    for t in range(max_len):
        logits = model(image_tensor, cur)           # (batch, seq_len, vocab)
        # take last token logits
        last_logits = logits[:, -1, :] / temperature  # (batch, vocab)

        # apply top-p (nucleus) sampling if requested
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True, dim=-1)  # (batch, vocab)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # mask tokens with cumulative prob > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # keep at least one token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # set to -inf in sorted_logits
            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

            # scatter back to original ordering
            # create a tensor same shape as last_logits and fill using sorted_indices
            # Note: scatter requires same dtype; we convert to float and then place back
            filtered_logits = torch.full_like(last_logits, float('-inf'))
            filtered_logits.scatter_(1, sorted_indices, sorted_logits)
            last_logits = filtered_logits

        # Sample
        probs = F.softmax(last_logits, dim=-1)       # (batch, vocab)
        nxt = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        nxt = nxt.to(device).long()

        # Append to current sequence (concatenate along seq dim)
        cur = torch.cat([cur, nxt], dim=1)  # cur: (batch, seq_len+1)

        # stop if EOS and minimum length satisfied (works for batch=1)
        if eos is not None and nxt.numel() == 1 and nxt.item() == eos and cur.shape[1]-1 >= min_len:
            break

    seq = cur.squeeze(0).tolist()[1:]  # drop the initial <sos>
    if eos is not None and eos in seq:
        seq = seq[:seq.index(eos)]
    model.train()
    return tokenizer.decode(seq)


@torch.no_grad()
def generate_beam(model, image_tensor, tokenizer, device, max_len=30, beam_width=3):
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
    model.train()
    return tokenizer.decode(best)

# -------------- Diagnostics --------------
def first_token_stats(model, dataloader, tokenizer, device, max_batches=200):
    model.eval()
    from collections import Counter
    cnt = Counter()
    total = 0
    with torch.no_grad():
        for i, (images, inps, tgts) in enumerate(dataloader):
            images = images.to(device); inps = inps.to(device)
            logits = model(images, inps)
            first_preds = torch.argmax(logits[:, 0, :], dim=-1).cpu().tolist()
            for p in first_preds:
                cnt[p] += 1; total += 1
            if i >= max_batches: break
    return [(tokenizer.index_to_word[idx], c, c/total) for idx, c in cnt.most_common(20)]


def smooth_list(x, window=3):
    if window <= 1 or len(x) < window:
        return x
    arr = np.array(x, dtype=float)
    # simple centered moving average (pad edges)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed.tolist()

def plot_metrics(history, title="Training History", save_path=None, smooth=False, smooth_window=3, live=False, figsize=(10,4)):
    """
    Plot training/validation curves from a `history` dict.

    Args:
      history (dict): expected keys include 'train_loss', 'val_loss', 'val_accuracy' (but any numeric lists are supported).
      title (str): figure title.
      save_path (str|None): path to save the figure (PNG). If None, the figure is not saved.
      smooth (bool): whether to smooth curves using a moving average.
      smooth_window (int): smoothing window (odd recommended).
      live (bool): if True, calls plt.pause(0.001) to enable live updating in notebooks.
      figsize (tuple): figure size.
    Returns:
      str|None: path where the figure was saved, or None.
    """
    if not isinstance(history, dict):
        raise ValueError("history must be a dict of lists")

    # pick known keys if present, otherwise plot whatever numeric lists found
    keys_order = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
    available = [k for k in keys_order if k in history]
    # fallback: any numeric list keys not in keys_order
    other_keys = [k for k in history.keys() if k not in keys_order and isinstance(history[k], (list, tuple))]
    available += other_keys

    if not available:
        raise ValueError("No plottable keys found in history. Expected keys like 'train_loss', 'val_loss', 'val_accuracy'.")

    # Prepare smoothed or raw series
    series = OrderedDict()
    for k in available:
        vals = list(history[k])
        series[k] = smooth_list(vals, window=smooth_window) if smooth else vals

    # Determine number of subplots: up to 2 (loss and accuracy) grouped logically
    loss_keys = [k for k in series.keys() if 'loss' in k]
    acc_keys = [k for k in series.keys() if 'acc' in k or 'accuracy' in k]

    n_plots = 0
    if loss_keys: n_plots += 1
    if acc_keys: n_plots += 1
    if n_plots == 0:
        # fallback: one plot with all series
        n_plots = 1
        loss_keys = list(series.keys())

    plt.figure(figsize=figsize)
    plot_i = 1

    if loss_keys:
        plt.subplot(1, n_plots, plot_i)
        for k in loss_keys:
            plt.plot(range(1, len(series[k]) + 1), series[k], label=k)
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
        plt.grid(True); plt.legend()
        plot_i += 1

    if acc_keys:
        plt.subplot(1, n_plots, plot_i)
        for k in acc_keys:
            plt.plot(range(1, len(series[k]) + 1), series[k], label=k)
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
        plt.grid(True); plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    saved = None
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        saved = save_path

    if live:
        plt.pause(0.001)

    plt.show()
    return saved


# ============================================================
# RUN: Dataset / model creation / train or load / generate
# ============================================================
if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- 1) Load manifest (assumes manifest entries contain 'image' and 'caption') ----------
    with open(MANIFEST_PATH, 'r') as f:
        image_data = json.load(f)
    image_filenames = [it['image'] for it in image_data]
    captions = [it['caption'] for it in image_data]
    print(f"Loaded {len(image_filenames)} items from {MANIFEST_PATH}")

    # ---------- 2) transforms ----------
    image_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    image_transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ---------- 3) Tokenizer (simple example) ----------
    class Tokenizer:
        def __init__(self, sentences):
            if isinstance(sentences, str): sentences = [sentences]
            sentences = [s.lower().strip(".") for s in sentences]
            self.vocab = list(set(word for sent in sentences for word in sent.split()))
            self.special = ['<unk>','<sos>','<eos>','<pad>']
            self.vocab += self.special
            self.word_to_index = {w:i for i,w in enumerate(self.vocab)}
            self.index_to_word = {i:w for w,i in self.word_to_index.items()}
        def encode(self, text):
            text = text.lower().strip("."); return [self.word_to_index.get(w,self.word_to_index['<unk>']) for w in text.split()]
        def decode(self, ids): return ' '.join(self.index_to_word.get(i,'<unk>') for i in ids)

    tokenizer = Tokenizer(captions)
    padding_idx = tokenizer.word_to_index.get('<pad>', tokenizer.word_to_index.get('<unk>', 0))
    print("Vocab size:", len(tokenizer.vocab))

# ======================================================================
    # CRITICAL: SAVE THE EXACT TOKENIZER MAPPING
    # This generates the file needed for the inference script to avoid mismatch.
    # ======================================================================
    TOKENIZER_VOCAB_PATH = 'data/processed/tokenizer_vocab.json'

    # Package the mappings
    vocab_data = {
        'word_to_index': tokenizer.word_to_index,
        # Convert index_to_word keys to strings for JSON serialization
        'index_to_word': {str(i):w for i,w in tokenizer.index_to_word.items()}
    }

    # Save the file
    os.makedirs(os.path.dirname(TOKENIZER_VOCAB_PATH), exist_ok=True)
    with open(TOKENIZER_VOCAB_PATH, 'w') as f:
        json.dump(vocab_data, f, indent=4)
    print(f"Tokenizer vocabulary saved to {TOKENIZER_VOCAB_PATH}")
    # ======================================================================





    # ---------- 4) Train/val/test split & Dataloaders ----------
    n = len(image_filenames)
    train_frac, val_frac, test_frac = 0.8, 0.1, 0.1
    idxs = list(range(n)); random.Random(SEED).shuffle(idxs)
    t_end = int(train_frac*n); v_end = t_end + int(val_frac*n)
    train_idx = idxs[:t_end]; val_idx = idxs[t_end:v_end]; test_idx = idxs[v_end:]
    leftover = n - (len(train_idx)+len(val_idx)+len(test_idx))
    if leftover>0: train_idx += idxs[-leftover:]

    train_files = [image_filenames[i] for i in train_idx]; train_caps = [captions[i] for i in train_idx]
    val_files   = [image_filenames[i] for i in val_idx];   val_caps   = [captions[i] for i in val_idx]
    test_files  = [image_filenames[i] for i in test_idx];  test_caps  = [captions[i] for i in test_idx]

    train_ds = ImageCaptionDataset(train_files, train_caps, tokenizer, image_transform_train)
    val_ds   = ImageCaptionDataset(val_files, val_caps, tokenizer, image_transform_val)
    test_ds  = ImageCaptionDataset(test_files, test_caps, tokenizer, image_transform_val)

    batch_size = 32
    num_workers = 1   # use 0/1 if system warns about many workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=lambda b: collate_fn(b, padding_idx))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=lambda b: collate_fn(b, padding_idx))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             collate_fn=lambda b: collate_fn(b, padding_idx))
    print("Dataset sizes -> train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))

    # ---------- 5) Model, criterion, config ----------
    # compute max caption length and set decoder max_length accordingly
    max_enc_len = max(len(tokenizer.encode(c)) for c in captions)
    required_effective = max_enc_len + 2
    max_length_for_model = required_effective - 1

    d_embed = 256; num_heads = 4; n_blocks = 4; img_size=224; patch_size=16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel(vocab_size=len(tokenizer.vocab),
                                 d_embed=d_embed, num_heads=num_heads, n_blocks=n_blocks,
                                 max_length=max_length_for_model, img_size=img_size,
                                 patch_size=patch_size).to(device)
    print("Decoder pos emb size:", model.transformer_decoder.pos_embed.num_embeddings)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=0.1)

    config = {
        'name': 'default_experiment',
        'optimizer_class': torch.optim.AdamW,
        'kwargs': {'lr':3e-4, 'weight_decay':1e-4},
        'scheduler_class': None,
        'scheduler_kwargs': {}
    }

    # ---------- 6) TRAIN or LOAD ----------
    if TRAIN:
        set_seed(SEED)
        history, trained_model = run_optimizer_experiment(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            criterion=criterion,
            num_epochs=15,
            device=device,
            config=config,
            display=True,
            checkpoint_path=CHECKPOINT_PATH
        )
        # trained_model is the trained instance
    else:
        # Load checkpoint (assumes checkpoint has 'model_state_dict')
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        trained_model = model
        print("Loaded checkpoint from", CHECKPOINT_PATH)

    # ---------- 7) Diagnostics and generation ----------
    print("First-token stats (top 10):", first_token_stats(trained_model, train_loader, tokenizer, device, max_batches=100)[:10])

    # Show a few examples (first N from test set)
    N = 10
    cnt = 0
    for images_b, inps_b, tgts_b in test_loader:
        for i in range(images_b.size(0)):
            img = images_b[i]            # (C,H,W)
            # reference decode (strip pads/eos)
            tgt_seq = tgts_b[i].tolist()
            eos = tokenizer.word_to_index.get('<eos>', None)
            ref = []
            for tok in tgt_seq:
                if tok == padding_idx: break
                if eos is not None and tok == eos: break
                ref.append(tok)
            ref_text = tokenizer.decode(ref)
            pred_greedy = generate_greedy(trained_model, img, tokenizer, device, max_len=50, min_len=2, temperature = 0.9, top_p = 0.9)
            pred_beam   = generate_beam(trained_model, img, tokenizer, device, max_len=50, beam_width=3)
            print(f"=== Example {cnt+1} ===")
            print("Reference:", ref_text)
            print("Greedy:", pred_greedy)
            print("Beam:", pred_beam)
            print()
            cnt += 1
            if cnt >= N: break
        if cnt >= N: break

    # Final evaluation (token-level)
    eval_loss, eval_acc = evaluate_model(trained_model, test_loader, criterion, device, padding_idx, split_name="Final Test")
    print(f"Final test loss: {eval_loss:.4f}, test accuracy: {eval_acc:.4f}")

# End of script
