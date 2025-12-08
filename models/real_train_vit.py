from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import copy
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# Define the optimizer configurations to test
optimizer_configs = [
    {
        'name': 'Adam',
        'optimizer_class': optim.Adam,
        'kwargs': {'lr': 0.001, 'weight_decay': 1e-5},
        'scheduler_class': StepLR,
        'scheduler_kwargs': {'step_size': 3, 'gamma': 0.1}
    },
    {
        'name': 'AdamW',
        'optimizer_class': optim.AdamW,
        'kwargs': {'lr': 0.001, 'weight_decay': 1e-5},
        'scheduler_class': StepLR,
        'scheduler_kwargs': {'step_size': 3, 'gamma': 0.1}
    },
    {
        'name': 'SGD',
        'optimizer_class': optim.SGD,
        'kwargs': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-5}, # Higher LR is typical for SGD
        'scheduler_class': StepLR,
        'scheduler_kwargs': {'step_size': 3, 'gamma': 0.1}
    },
]

class TransformerBlock(nn.Module):
    def __init__(self, d_embed, num_heads, dropout_rate = 0.1):
        super().__init__()
        # MultiheadAttention expects (seq_len, batch, embed) by default
        self.multihead_attn = nn.MultiheadAttention(d_embed, num_heads)
        self.attn_norm = nn.LayerNorm(d_embed)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ff_linear = nn.Linear(d_embed, 4 * d_embed)
        self.ff_linear2 = nn.Linear(4 * d_embed, d_embed)
        self.ff_norm = nn.LayerNorm(d_embed)
        self.relu = nn.ReLU()

        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask):
        """
        Supports:
          - x: (seq_len, d_embed)  -> unbatched
          - x: (batch, seq_len, d_embed) -> batched
        attn_mask should be shape (L, L) or None (L = seq length)
        """
        # Unbatched: (seq_len, d_embed)
        if x.dim() == 2:
            x_b = x.unsqueeze(1)  # (seq_len, 1, d_embed)
            attn_out_b, _ = self.multihead_attn(x_b, x_b, x_b, attn_mask=attn_mask)
            attn_out = attn_out_b.squeeze(1)  # (seq_len, d_embed)
            x = x + attn_out
            x = self.attn_norm(x)
            ff = self.ff_linear(x)
            ff = self.relu(ff)
            ff = self.ff_linear2(ff)
            x = x + ff
            x = self.ff_norm(x)
            return x

        # Batched: (batch, seq_len, d_embed)
        elif x.dim() == 3:
            # Transpose to (seq_len, batch, d_embed) for MultiheadAttention
            x_t = x.transpose(0, 1)  # (seq_len, batch, d_embed)
            attn_out_t, _ = self.multihead_attn(x_t, x_t, x_t, attn_mask=attn_mask)
            h = x_t + self.attn_dropout(attn_out_t)
            h = self.attn_norm(h)  # LayerNorm over last dim (embed)
            ff = self.ff_linear(h)
            ff = self.relu(ff)
            ff = self.ff_linear2(ff)
            
            h = h + self.ff_dropout(ff)
            h = self.ff_norm(h)
            return h.transpose(0, 1)  # (batch, seq_len, d_embed)

        else:
            raise ValueError(f"Unsupported input dim {x.dim()} in TransformerBlock")


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, max_length=512, n_blocks=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.pos_embed = nn.Embedding(max_length + 1, d_embed)
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d_embed, vocab_size)

    def forward(self, x):
        device = x.device
        seq_len = x.shape[0]
        tok_emb = self.embed(x)
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embed(pos_idx)
        h = tok_emb + pos_emb
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        for block in self.blocks:
            h = block(h, attn_mask)
        logits = self.unembed(h)
        return logits
    

# !!! IMPORTANT: Replace 'data.json' with the actual path to your JSON file !!!
json_file_path = 'data/processed/imgflip575k_manifest.json'

# Load the JSON file
with open(json_file_path, 'r') as f:
    image_data = json.load(f)

# Extract image filenames and captions
image_filenames = [item['image'] for item in image_data]
captions = [item['caption'] for item in image_data]
humor_labels = [item['tone'] for item in image_data] # Changed from 'humor_label' to 'tone'

print(f"Loaded {len(image_data)} entries from {json_file_path}")
print("First image filename:", image_filenames[0])
print("First caption:", captions[0])
print("First humor label:", humor_labels[0])

from torchvision import transforms

from torchvision import transforms

# Define image transformations with data augmentation
image_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to a fixed size
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change color properties
    transforms.ToTensor(),              # Convert image to a PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])

print("Image transformations defined with augmentation:")
print(image_transform_train)


# Define image transformations
image_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),          # Convert image to a PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
])

print("Image transformations defined:")
print(image_transform_val)


# Re-initialize the tokenizer with the loaded captions

# Definition of the Tokenizer class
class Tokenizer:
    def __init__(self, sentences):
        # Ensure sentences is a list; if it's a single string, wrap it in a list
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = [sentence.lower().strip(".") for sentence in sentences]
        self.vocab = list(set(word for sentence in sentences for word in sentence.split()))
        self.special_tokens = ['<unk>', '<sos>', '<eos>']
        self.vocab += self.special_tokens
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    def encode(self, text):
        text = text.lower().strip(".")
        return [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in text.split()]

    def decode(self, indices):
        return ' '.join(self.index_to_word.get(idx, '<unk>') for idx in indices)

# Initialize the tokenizer with the captions loaded from data.json
tokenizer = Tokenizer(captions)

print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
print("Example encoding of a caption:")
encoded_caption = tokenizer.encode(captions[0])
print(f"Original caption: '{captions[0]}' ")
print(f"Encoded: {encoded_caption}")
print(f"Decoded: '{tokenizer.decode(encoded_caption)}' ")

import torch
from torch import nn

# Assuming TransformerBlock is already defined as in previous cells
# (Refer to cell Sd_l_IarCz03 for TransformerBlock definition if needed)

class VisionEncoder(nn.Module):
    def __init__(self, d_embed, num_heads, n_blocks, img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding: Convert image patches into d_embed vectors
        # A convolutional layer can serve as a learnable patch embedding
        # Output channels = d_embed, kernel_size = patch_size, stride = patch_size
        self.patch_embedding = nn.Conv2d(in_channels, d_embed, kernel_size=patch_size, stride=patch_size)

        # Positional embedding for patches
        self.positional_embedding = nn.Embedding(self.num_patches, d_embed)

        # Transformer Blocks (reusing the TransformerBlock class previously defined)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)
        ])

        # Layer normalization after the last block
        self.norm = nn.LayerNorm(d_embed)

        # A linear layer to project the sequence of patch embeddings to a single fixed-size feature vector
        # For simplicity, we can average or take the embedding of a special 'CLS' token.
        # Here, we'll average the patch embeddings.
        # self.fc = nn.Linear(d_embed, d_embed) # Optional: if a fixed-size output is needed

    def forward(self, x):
        # x is expected to be (batch_size, in_channels, img_size, img_size)

        # 1. Patch Embedding
        x = self.patch_embedding(x) # Output: (batch_size, d_embed, num_patches_h, num_patches_w)

        # Flatten patches and transpose for Transformer input (seq_len, batch_size, d_embed)
        # We need to adapt this for unbatched input first as per previous Transformer
        # For this VisionEncoder, let's assume a batch dimension for now, and adapt later if needed
        # If input is (C, H, W) for single image, patch_embedding outputs (d_embed, H', W')
        # Reshape to (num_patches, d_embed)

        # Assuming single image input for consistency with earlier Transformer
        # x: (d_embed, num_patches_h, num_patches_w) for unbatched input
        if x.dim() == 3: # (d_embed, H', W') -> assuming single image input
            x = x.flatten(1).transpose(0, 1) # Output: (num_patches, d_embed)
        elif x.dim() == 4:  # (batch_size, d_embed, H', W')
            batch_size = x.shape[0]
            # x -> (batch, d_embed, H'*W')
            x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_embed)

            # positional embeddings: (num_patches, d_embed) -> expand to (batch, num_patches, d_embed)
            positions = torch.arange(0, self.num_patches, device=x.device)
            pos_emb = self.positional_embedding(positions)  # (num_patches, d_embed)
            x = x + pos_emb.unsqueeze(0)  # broadcast to (batch, num_patches, d_embed)

            # pass through Transformer blocks (they now accept batched input)
            attn_mask = None
            for block in self.blocks:
                x = block(x, attn_mask)  # (batch, num_patches, d_embed)

            x = self.norm(x)  # (batch, num_patches, d_embed)

            # Average over patches -> (batch, d_embed)
            feature_vector = x.mean(dim=1)  # (batch, d_embed)

            return feature_vector



        # 2. Add Positional Embedding
        positions = torch.arange(0, self.num_patches, device=x.device).unsqueeze(0)
        # If x is (num_patches, d_embed) for unbatched
        x = x + self.positional_embedding(positions) # (num_patches, d_embed) + (1, num_patches, d_embed) -> error with unbatched
        
        # Correct for unbatched positional embedding
        x = x + self.positional_embedding(positions.squeeze(0)) # (num_patches, d_embed) + (num_patches, d_embed)

        # 3. Pass through Transformer Blocks
        attn_mask = None # No causal mask for vision encoder
        for block in self.blocks:
            x = block(x, attn_mask)

        # 4. Layer Normalization
        x = self.norm(x)
        
        # For the fixed-size feature vector, we can average over the patch embeddings
        # or use a special CLS token (not implemented here).
        # Averaging gives (d_embed) vector for unbatched input
        feature_vector = x.mean(dim=0) # Output: (d_embed,)

        return feature_vector

print("VisionEncoder class defined. Note: This assumes TransformerBlock is already defined.")


from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class ImageCaptionDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, image_transform=None):
        self.image_filenames = image_filenames
        self.captions = captions
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        caption = self.captions[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)

        # Tokenize caption
        encoded_caption = self.tokenizer.encode(caption)
        # Add <sos> and <eos> tokens for the decoder
        input_seq = [self.tokenizer.word_to_index['<sos>']] + encoded_caption
        target_seq = encoded_caption + [self.tokenizer.word_to_index['<eos>']]

        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        return image, input_tensor, target_tensor

# Create the full dataset without any specific transform yet
full_dataset = ImageCaptionDataset(
    image_filenames=image_filenames,
    captions=captions,
    tokenizer=tokenizer,
    image_transform=None # No transform applied here yet
)

# Perform train-test split (e.g., 80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
# Use random_split to get indices, then create separate datasets with different transforms
train_indices, test_indices = random_split(full_dataset, [train_size, test_size])

# Create train and test datasets, applying the appropriate transforms
train_dataset = ImageCaptionDataset(
    image_filenames=[full_dataset.image_filenames[i] for i in train_indices.indices],
    captions=[full_dataset.captions[i] for i in train_indices.indices],
    tokenizer=tokenizer,
    image_transform=image_transform_train # Apply training transforms
)

test_dataset = ImageCaptionDataset(
    image_filenames=[full_dataset.image_filenames[i] for i in test_indices.indices],
    captions=[full_dataset.captions[i] for i in test_indices.indices],
    tokenizer=tokenizer,
    image_transform=image_transform_val # Apply validation transforms
)

# Create DataLoaders
batch_size = 1 # For unbatched Transformer input, we'll iterate one by one
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Total dataset size: {len(full_dataset)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Example of fetching one item from the DataLoader
example_image, example_input_caption, example_target_caption = next(iter(train_dataloader))
print("\nExample from train_dataloader:")
print(f"Image tensor shape: {example_image.shape}")
print(f"Input caption tensor: {example_input_caption}")
print(f"Decoded input caption: {tokenizer.decode(example_input_caption.squeeze(0).tolist())}")
print(f"Target caption tensor: {example_target_caption}")
print(f"Decoded target caption: {tokenizer.decode(example_target_caption.squeeze(0).tolist())}")


from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    input_tensors = [item[1] for item in batch]
    target_tensors = [item[2] for item in batch]
    padding_idx = tokenizer.word_to_index['<unk>']
    input_tensors_padded = pad_sequence(
        input_tensors,
        batch_first = True,
        padding_value = padding_idx
    )
    target_tensors_padded = pad_sequence(
        target_tensors,
        batch_first = True,
        padding_value = padding_idx
    )

    images_batch = torch.cat(images, dim = 0)
    return images_batch, input_tensors_padded, target_tensors_padded
# Create the dataset
full_dataset = ImageCaptionDataset(
    image_filenames=image_filenames,
    captions=captions,
    tokenizer=tokenizer,
    image_transform=image_transform
)

# Perform train-test split (e.g., 80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 32 # For unbatched Transformer input, we'll iterate one by one

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

print(f"Total dataset size: {len(full_dataset)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Example of fetching one item from the DataLoader
example_image, example_input_caption, example_target_caption = next(iter(train_dataloader))
print("\nExample from train_dataloader:")
print(f"Image tensor shape: {example_image.shape}")
print(f"Input caption tensor: {example_input_caption}")
print(f"Decoded input caption: {tokenizer.decode(example_input_caption[0].tolist())}")
print(f"Target caption tensor: {example_target_caption}")
print(f"Decoded target caption: {tokenizer.decode(example_target_caption[0].tolist())}")


import torch
from torch import nn

# Assuming TransformerBlock is already defined (from cell Sd_l_IarCz03)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, max_length=512, n_blocks=4):
        super().__init__()
        self.d_embed = d_embed
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, d_embed)
        # +1 to max_length for the prepended image feature token
        self.pos_embed = nn.Embedding(max_length + 1, d_embed) 
        self.blocks = nn.ModuleList([TransformerBlock(d_embed, num_heads) for _ in range(n_blocks)])
        self.unembed = nn.Linear(d_embed, vocab_size)

    def forward(self, x, image_features):
        """
        Batched decoder:
        x: (batch, seq_len)           -- token indices
        image_features: (batch, d_embed)
        Returns:
        logits: (batch, seq_len, vocab_size)
        """
        if x.dim() != 2:
            raise ValueError("TransformerDecoder.forward expects x shape (batch, seq_len) for batched mode")

        device = x.device
        batch, seq_len_text = x.shape

        # Token embeddings -> (batch, seq_len_text, d_embed)
        tok_emb = self.embed(x)

        # Ensure image_features is (batch, d_embed)
        if image_features.dim() == 1:
            # expand to batch
            image_features = image_features.unsqueeze(0).expand(batch, -1)
        elif image_features.dim() == 2 and image_features.shape[0] != batch:
            raise ValueError("Batch size mismatch between x and image_features")

        # image_features as first token: (batch, 1, d_embed)
        img_feat_unsq = image_features.unsqueeze(1)

        # Combine -> (batch, seq_len_text + 1, d_embed)
        combined = torch.cat((img_feat_unsq, tok_emb), dim=1)
        effective_seq_len = combined.shape[1]

        # Positional embeddings -> (effective_seq_len, d_embed) -> unsqueeze to (1, effective_seq_len, d_embed)
        pos_idx = torch.arange(effective_seq_len, device=device)
        if effective_seq_len > self.pos_embed.num_embeddings:
            print("⚠️ ERROR: seq_len too large:",
                effective_seq_len,
                "max allowed:", self.pos_embed.num_embeddings,
                "caption length:", seq_len_text)
            print("Caption tokens:", x[0].tolist())
            raise ValueError("Caption too long for positional embeddings")

        pos_emb = self.pos_embed(pos_idx).unsqueeze(0)

        h = combined + pos_emb  # broadcasting -> (batch, effective_seq_len, d_embed)

        # causal mask: shape (effective_seq_len, effective_seq_len)
        attn_mask = torch.triu(torch.ones(effective_seq_len, effective_seq_len, dtype=torch.bool, device=device), diagonal=1)
        attn_mask[0, :] = False  # allow image token (index 0) to attend to all

        # Pass through blocks (each block supports batched input)
        for block in self.blocks:
            h = block(h, attn_mask)  # (batch, effective_seq_len, d_embed)

        # Unembed only text tokens (exclude index 0 image token)
        logits = self.unembed(h[:, 1:, :])  # (batch, seq_len_text, vocab_size)
        return logits


print("TransformerDecoder class defined.")



import torch

# Instantiate the VisionEncoder
# Using example values for d_embed, num_heads, n_blocks.
# These should match your desired model configuration.
# For now, we'll use d_embed=64, num_heads=4, n_blocks=4 consistent with previous examples.

import torch
from torch import nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_embed=128, num_heads=4, n_blocks=4, max_length=512,
                 img_size=224, patch_size=16, in_channels=3):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            d_embed=d_embed, num_heads=num_heads, n_blocks=n_blocks,
            img_size=img_size, patch_size=patch_size, in_channels=in_channels
        )
        self.transformer_decoder = TransformerDecoder(
            vocab_size=vocab_size, d_embed=d_embed, num_heads=num_heads,
            max_length=max_length, n_blocks=n_blocks
        )

    def forward(self, image, caption_input):
        # 1. Encode the image to get a fixed-size feature vector
        image_features = self.vision_encoder(image)

        # 2. Decode the caption using the image features as context
        logits = self.transformer_decoder(caption_input, image_features)
        return logits

print("ImageCaptioningModel class defined.")


import torch
from torch import nn
from tqdm import tqdm

# Hyperparameters (adjust as needed)
vocab_size = len(tokenizer.vocab)
d_embed = 128
num_heads = 4
n_blocks = 4
max_length = 512 # Max sequence length for positional encoding
img_size = 224
patch_size = 16
in_channels = 3
learning_rate = 0.001
num_epochs = 5



# Initialize the model
model = ImageCaptioningModel(
    vocab_size=vocab_size, d_embed=d_embed, num_heads=num_heads, n_blocks=n_blocks,
    max_length=max_length, img_size=img_size, patch_size=patch_size, in_channels=in_channels
)

# -----------------------
# LOAD EXISTING MODEL?
# -----------------------

model = ImageCaptioningModel(
    vocab_size=vocab_size, d_embed=d_embed, num_heads=num_heads, n_blocks=n_blocks,
    max_length=max_length, img_size=img_size, patch_size=patch_size, in_channels=in_channels
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model.to(device)



LOAD_EXISTING_MODEL = True   # Change to False if you want to retrain
if LOAD_EXISTING_MODEL:
    try:
        # Load the saved state dict
        state_dict = torch.load("models/meme_caption_vit.pt", map_location=device)

        # Check if the keys have the 'module.' prefix (indicating it was saved via DataParallel)
        is_dataparallel_saved = any(k.startswith('module.') for k in state_dict.keys())
        
        # If the model is currently NOT wrapped in DataParallel but the weights ARE
        if is_dataparallel_saved:
            # Create a new state dict without the 'module.' prefix
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            # Load standard state dict
            model.load_state_dict(state_dict)
            
        print("Loaded existing model weights!")
    except FileNotFoundError:
        print("No saved model found — training from scratch.")
else:
    print("Training from scratch.")

# Evaluating Model
# --- ORIGINAL/FLAWED evaluate_model ---
# (DELETE the version in your submitted code that contains early stopping,
# since it uses undefined variables like epoch, avg_train_loss, etc.)

# --- CORRECTED evaluate_model (Ensure this is defined BEFORE run_experiment) ---

def evaluate_model(model, dataloader, criterion, device, split_name="Test"):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    padding_idx = tokenizer.word_to_index['<unk>'] # Need tokenizer defined earlier

    with torch.no_grad():
        for images, input_captions, target_captions in dataloader:
            images = images.to(device)
            input_captions = input_captions.to(device)
            target_captions = target_captions.to(device)

            logits = model(images, input_captions)
            loss = criterion(logits.permute(0, 2, 1), target_captions)
            total_loss += loss.item() * images.size(0) # Scale by batch size

            preds = torch.argmax(logits, dim=-1)
            
            # Mask out padding tokens for accurate count
            mask = (target_captions != padding_idx)
            total_correct += (preds == target_captions)[mask].sum().item()
            total_predictions += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0

    print(f"  > {split_name} Loss: {avg_loss:.4f}, {split_name} Accuracy: {accuracy:.4f}")
    
    model.train() # Set model back to training mode
    return avg_loss, accuracy


# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_index['<unk>']"- Starting Experiment: {config['name']} ---")
    
    # Reset Weights: Initialize a brand new model instance
    model = model_class(
        vocab_size=len(tokenizer.vocab), d_embed=128, num_heads=4, n_blocks=4,
        max_length=512, img_size=224, patch_size=16, in_channels=3
    ).to(device)

    # Setup Optimizer and Scheduler based on config
    optimizer = config['optimizer_class'](model.parameters(), **config['kwargs'])
    scheduler = config['scheduler_class'](optimizer, **config['scheduler_kwargs'])
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}



    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # --- (Core Training Loop - Adapted from your deleted code) ---
        for images, input_captions, target_captions in tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs} ({config['name']} Training)"):
            # Move data to device (handle padding/batches correctly based on your final code)
            images = images.to(device)
            input_captions = input_captions.to(device)
            target_captions = target_captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images, input_captions)
            loss = criterion(logits.permute(0, 2, 1), target_captions)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * images.size(0)

        avg_train_loss = total_train_loss / len(train_data.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation (using the existing evaluate_model function)
        val_loss, val_acc = evaluate_model(model, test_data, criterion, device, split_name=config['name'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        scheduler.step()
        
    return history



# Set a seed for reproducibility before starting all experiments
torch.manual_seed(42) 

all_results = {}

print("Starting Comparative Optimizer Training...")
# Run all experiments
for config in optimizer_configs:
    history = run_experiment(
        config, 
        ImageCaptioningModel, 
        train_dataloader, 
        test_dataloader, 
        criterion, 
        device, 
        num_epochs,
        tokenizer
    )
    all_results[config['name']] = history

print("\nAll experiments complete. Results stored in 'all_results'.")

# Your final plotting logic to compare results will go here.    
    evaluate_model(model, test_dataloader, criterion, device, split_name="Test")

    # Optional: Add evaluation on test_dataloader here
    # model.eval()
    # with torch.no_grad():
    #     # ... evaluation logic ...
# -----------------------
# SAVE MODEL CHECKPOINT
# -----------------------
# --- Inside run_experiment function, near the end ---

# Optional: Save the final model state (identified by optimizer name)
final_save_path = f"models/caption_vit_{config['name']}.pt"
    
# Use the 'model' variable that was trained in this experiment instance
if isinstance(model, torch.nn.DataParallel):
    torch.save(model.module.state_dict(), final_save_path)
else:
    torch.save(model.state_dict(), final_save_path)
        
print(f"Model trained with {config['name']} saved to: {final_save_path}")
    
return history

print(f"Model saved to: {save_path}")

print("Training Complete.")

# Report total number of model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal number of trainable parameters: {total_params:,}")


#Visualizing Training and Validation Curves:
import matplotlib.pyplot as plt
epochs_ran = len(train_losses)
epochs_range = range(1, epochs_ran + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

