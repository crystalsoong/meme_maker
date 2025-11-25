# baselines/resnet_baseline.py
"""
Simple baseline: extract ResNet50 pooled features and train a linear classifier to predict
a small set of caption classes. This is a minimal baseline script to show a baseline workflow.
hi my name is alan
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.dataset import MemeCaptionDataset
from transformers import AutoTokenizer

def build_feature_extractor(device):
    r50 = models.resnet50(pretrained=True)
    r50.fc = nn.Identity()
    r50.to(device).eval()
    return r50

def train_baseline(manifest, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    ds = MemeCaptionDataset(manifest, tokenizer, transforms=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]), max_length=32)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    feat = build_feature_extractor(device)
    # dummy classifier: reduce to 128-dim then output 2 classes (for toy test)
    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(2048,128), nn.ReLU(), nn.Linear(128,2))
        def forward(self,x):
            return self.fc(x)
    clf = Clf().to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(2):
        for b in loader:
            pv = b['pixel_values'].to(device)
            with torch.no_grad():
                feats = feat(pv).detach()
            # toy labels: humor if token <humor> present
            labels = torch.tensor([1 if "<humor>" in t else 0 for t in [" ".join(tokenizer.decode(ids).split()) for ids in b['input_ids']]]).to(device)
            logits = clf(feats)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    print("Baseline training finished.")
if __name__ == "__main__":
    train_baseline("data/processed/merged_manifest_train_small.json", device='cuda' if torch.cuda.is_available() else 'cpu')
