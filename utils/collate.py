# utils/collate.py
import torch

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = input_ids.clone()
    # optionally set pad token ids to -100 if using CrossEntropyLoss ignore_index
    return {"pixel_values": pixel_values, "labels": labels, "attention_mask": attention_mask}
