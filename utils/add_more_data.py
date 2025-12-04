# scripts/add_more_public_data.py
"""
Add public datasets (SBU / Conceptual Captions) into your project's manifest.

Usage examples:
python scripts/add_more_public_data.py --dataset sbu --max 200 --append_to data/processed/merged_manifest.json
python scripts/add_more_public_data.py --dataset conceptual --max 500 --append_to data/processed/merged_manifest.json
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import random

# Minimal blacklist (expand as needed)
BLACKLIST = {"kill", "hate", "slur_here"}  # replace slur_here with real tokens if desired

def text_is_safe(text):
    if not text:
        return False
    t = text.lower()
    for bad in BLACKLIST:
        if bad in t:
            return False
    return True

def save_image_from_bytes(bts, out_path):
    try:
        img = Image.open(BytesIO(bts)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=90)
        return str(out_path)
    except Exception:
        return None

def download_url_to_path(url, out_path, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        return save_image_from_bytes(r.content, out_path)
    except Exception:
        return None

def process_record(record, idx, img_dir, dataset_name):
    """
    record: a dict from HF dataset. We attempt to extract an image and single caption.
    """
    # possible caption fields
    caption = None
    for key in ("caption", "captions", "sentence", "sentences", "text"):
        if key in record and record[key]:
            # captions may be list or str
            v = record[key]
            if isinstance(v, list):
                # pick first textual element
                if isinstance(v[0], dict) and "raw" in v[0]:
                    caption = v[0]["raw"]
                else:
                    caption = v[0] if isinstance(v[0], str) else str(v[0])
            elif isinstance(v, dict) and "text" in v:
                caption = v["text"]
            else:
                caption = str(v)
            break

    if not text_is_safe(caption):
        return None

    # possible image info
    # many HF datasets provide 'image' as PIL.Image or dict with 'url' or 'image_url'
    img_path = None
    # first prefer image object
    if "image" in record and record["image"] is not None:
        try:
            pil = record["image"]
            out_path = img_dir / f"{dataset_name}_{idx}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pil.save(out_path)
            img_path = str(out_path)
        except Exception:
            img_path = None

    # fallback to url fields
    if img_path is None:
        for key in ("image_url", "img_url", "url", "photo_url"):
            if key in record and record[key]:
                url = record[key]
                out_path = img_dir / f"{dataset_name}_{idx}.jpg"
                saved = download_url_to_path(url, out_path)
                if saved:
                    img_path = saved
                break

    if img_path is None:
        # some datasets store nested dicts (e.g., {'image': {'url': ...}})
        if "image" in record and isinstance(record["image"], dict):
            for subkey in ("url", "image_url", "img_url"):
                if subkey in record["image"]:
                    out_path = img_dir / f"{dataset_name}_{idx}.jpg"
                    saved = download_url_to_path(record["image"][subkey], out_path)
                    if saved:
                        img_path = saved
                        break

    if img_path is None:
        return None

    return {"image": img_path, "caption": caption.strip(), "tone": "<factual>"}

def add_sbu(max_examples, out_added, append_to):
    """
    SBU Captions via HF is registered as 'sbu' or 'sbu_captions' on some mirrors.
    We'll try a few known ids.
    """
    tried = ["sbu_captions", "sbu"]
    ds = None
    for name in tried:
        try:
            ds = load_dataset(name, split="train")
            print("Loaded SBU dataset id:", name)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load SBU dataset from HF (tried ids: {})".format(tried))

    img_dir = Path("data/raw/sbu_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = {}
        for i, rec in enumerate(ds):
            if max_examples and len(futures) >= max_examples:
                break
            futures[ex.submit(process_record, rec, i, img_dir, "sbu")] = i
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = fut.result()
            if res:
                results.append(res)
    # write
    with open(out_added, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} SBU entries → {out_added}")
    if append_to:
        append_manifest(append_to, out_added)
    return out_added

def add_conceptual(max_examples, out_added, append_to):
    """
    Conceptual Captions: try known HF ids. Note: conceptual caption datasets are large; we sample max_examples.
    """
    tried = ["conceptual_captions", "conceptual-cc", "conceptual-captions"]
    ds = None
    for name in tried:
        try:
            ds = load_dataset(name, split="train")
            print("Loaded Conceptual dataset id:", name)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load Conceptual Captions from HF (tried ids: {})".format(tried))

    img_dir = Path("data/raw/conceptual_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # iterate and process up to max_examples
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = {}
        count = 0
        for i, rec in enumerate(ds):
            if max_examples and count >= max_examples:
                break
            futures[ex.submit(process_record, rec, i, img_dir, "conceptual")] = i
            count += 1
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = fut.result()
            if res:
                results.append(res)
    with open(out_added, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} Conceptual entries → {out_added}")
    if append_to:
        append_manifest(append_to, out_added)
    return out_added

def append_manifest(target_manifest, added_manifest):
    tpath = Path(target_manifest)
    if tpath.exists():
        with open(tpath, "r", encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = []
    with open(added_manifest, "r", encoding="utf-8") as f:
        added = json.load(f)
    before = len(base)
    base.extend(added)
    random.shuffle(base)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
    print(f"Appended {len(added)} entries to {target_manifest} (was {before}, now {len(base)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sbu", "conceptual"], required=True)
    parser.add_argument("--max", type=int, default=500)
    parser.add_argument("--out", default=None)
    parser.add_argument("--append_to", default=None, help="path to existing manifest to append to (optional)")
    args = parser.parse_args()

    if args.dataset == "sbu":
        out_added = args.out or f"data/processed/added_sbu_{args.max}.json"
        add_sbu(args.max, out_added, args.append_to)
    elif args.dataset == "conceptual":
        out_added = args.out or f"data/processed/added_conceptual_{args.max}.json"
        add_conceptual(args.max, out_added, args.append_to)

if __name__ == "__main__":
    main()
