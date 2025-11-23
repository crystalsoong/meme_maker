# utils/dataset.py
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class MemeCaptionDataset(Dataset):
    """
    Expects manifest JSON: list of {"image": path_or_url, "caption": text, "tone": "<humor>"|"<factual>"}
    Use local image paths (preferred). If image is URL, download first and update manifest.
    """
    def __init__(self, manifest_path, tokenizer, transforms=None, max_length=32):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_path = it["image"]
        # open
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open {img_path}: {e}")
        if self.transforms:
            pixel_values = self.transforms(image)
        else:
            from torchvision import transforms
            t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            pixel_values = t(image)
        # prepend tone token
        caption = f"{it['tone']} {it['caption']}"
        tok = self.tokenizer(caption, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}
