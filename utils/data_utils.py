# utils/data_utils.py
"""
Utilities:
- copy_design_notes(): copy the uploaded design notes PDF into data/raw/
- build_imgflip_manifest(): build manifest JSON from a CSV/JSON in imgflip folder
- build_coco_manifest_from_hf(): save COCO images and captions into manifest using HF datasets
- download_image(): helper to download images (if manifest contains URLs)
- merge_manifests(): merge multiple manifests into one shuffled manifest
- verify_manifest(): quick sanity checks
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import requests
import pandas as pd

# Path to the uploaded design notes (from your session)
UPLOADED_DESIGN_NOTES = "/mnt/data/Models 2.pdf"

def copy_design_notes(dest_dir="data/raw"):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(UPLOADED_DESIGN_NOTES):
        dst = dest_dir / Path(UPLOADED_DESIGN_NOTES).name
        shutil.copy(UPLOADED_DESIGN_NOTES, dst)
        print(f"Copied design notes to {dst}")
        return str(dst)
    else:
        print("Design notes not found at", UPLOADED_DESIGN_NOTES)
    return None

def build_imgflip_manifest(imgflip_dir, out_manifest="data/processed/imgflip_manifest.json",
                           image_col_candidates=None, caption_col_candidates=None):
    """
    Parse first CSV/JSON found in imgflip_dir and write manifest of {"image":..., "caption":..., "tone":"<humor>"}.
    If image paths are URLs, they will be left as-is (downloader provided).
    """
    imgflip_dir = Path(imgflip_dir)
    if not imgflip_dir.exists():
        raise FileNotFoundError(imgflip_dir)
    candidates = list(imgflip_dir.glob("*.csv")) + list(imgflip_dir.glob("*.json")) + list(imgflip_dir.glob("*.ndjson"))
    if not candidates:
        raise FileNotFoundError("No CSV/JSON found in " + str(imgflip_dir))
    src = candidates[0]
    print("Parsing:", src)
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    else:
        df = pd.read_json(src, lines=(src.suffix.lower() in [".ndjson", ".jsonl"]))
    cols = list(df.columns)
    # guess columns
    img_col = None
    cap_col = None
    if image_col_candidates:
        for c in image_col_candidates:
            if c in cols:
                img_col = c; break
    if caption_col_candidates:
        for c in caption_col_candidates:
            if c in cols:
                cap_col = c; break
    # fallback guesses
    if img_col is None:
        img_col = next((c for c in cols if "img" in c.lower() or "image" in c.lower() or "url" in c.lower()), cols[0])
    if cap_col is None:
        cap_col = next((c for c in cols if "caption" in c.lower() or "text" in c.lower()), cols[1] if len(cols)>1 else cols[0])
    print("Using columns:", img_col, cap_col)
    manifest = []
    for _, row in df.iterrows():
        img = row.get(img_col)
        cap = row.get(cap_col)
        if pd.isna(img) or pd.isna(cap):
            continue
        manifest.append({"image": str(img).strip(), "caption": str(cap).strip(), "tone": "<humor>"})
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(manifest)} records to {out_manifest}")
    return out_manifest

def download_image(url, out_path, timeout=10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            # verify open
            Image.open(out_path).close()
            return str(out_path)
    except Exception as e:
        print("Failed to download", url, "->", e)
    return None

def build_coco_manifest_from_hf(
    output_manifest="data/processed/coco_manifest.json",
    split="train",
    dataset_name="HuggingFaceM4/COCO-Captions"
):
    from datasets import load_dataset
    from pathlib import Path
    import json

    print(f"Loading COCO dataset: {dataset_name}, split: {split}")
    ds = load_dataset(dataset_name, split=split)

    out = []
    img_dir = Path("data/raw/coco_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(ds):
        img = item["image"]       # PIL Image
        caption = item["caption"] # string

        img_path = img_dir / f"coco_{split}_{i}.jpg"
        img.save(img_path)

        out.append({
            "image": str(img_path),
            "caption": caption.strip(),
            "tone": "<factual>"
        })

    Path(output_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} COCO entries → {output_manifest}")
    return output_manifest



def merge_manifests(manifest_paths, out_manifest="data/processed/merged_manifest.json", shuffle=True):
    merged = []
    for p in manifest_paths:
        with open(p, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    if shuffle:
        import random
        random.shuffle(merged)
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged)} records to {out_manifest}")
    return out_manifest

def verify_manifest(manifest_path, n=3):
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(manifest_path)
    with open(p, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Manifest {manifest_path} has {len(items)} items. Showing {n} samples:")
    for i, it in enumerate(items[:n]):
        print(i, {"image": it.get("image"), "caption": it.get("caption")[:120] + ("..." if len(it.get("caption",""))>120 else ""), "tone": it.get("tone")})

if __name__ == "__main__":
    # small CLI helpers
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy_notes", action="store_true")
    parser.add_argument("--imgflip_dir", default=None)
    parser.add_argument("--coco_hf", action="store_true")
    args = parser.parse_args()
    if args.copy_notes:
        copy_design_notes()
    if args.imgflip_dir:
        build_imgflip_manifest(args.imgflip_dir)
    if args.coco_hf:
        build_coco_manifest_from_hf()
