# eval/evaluate_metrics.py
"""
Quick evaluation: generates captions for first N examples in manifest and prints them.
Usage:
python eval/evaluate_metrics.py --model_dir outputs/mememaker --manifest data/processed/merged_manifest_val.json --n 20
"""
import argparse
import json
from PIL import Image
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from torchvision import transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--n", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_special_tokens({"bos_token":"<bos>", "eos_token":"<eos>", "pad_token":"<pad>", "additional_special_tokens":["<humor>","<factual>"]})
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(args.manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    for it in items[:args.n]:
        img = Image.open(it["image"]).convert("RGB")
        pv = tf(img).unsqueeze(0).to(device)
        outs = model.generate(pixel_values=pv, max_length=32, num_beams=3)
        txt = tokenizer.decode(outs[0], skip_special_tokens=True)
        print("Image:", it["image"])
        print("Ref:", it["caption"])
        print("Gen:", txt)
        print("-"*40)

if __name__ == "__main__":
    main()
