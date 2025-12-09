#!/usr/bin/env python3
"""
Build ImgFlip manifest with STRICT DATA QUALITY FILTERING.

- ONLY accepts captions found in 'boxes' (on-image text).
- Rejects entries relying on external 'title' or 'post' metadata.
- Applies a minimum caption length filter.

Output:
  - Images → data/raw/imgflip575k/images/
  - Manifest → data/processed/imgflip575k_manifest.json (Original Name)
"""
import json
from pathlib import Path
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image
import concurrent.futures
import os 

# --- CONFIGURATION ---
SRC_DIR = Path("imgflip575k/dataset/memes")
OUT_IMG_DIR = Path("data/raw/imgflip575k/images")
# --- REVISED TO USE ORIGINAL FILENAME ---
OUT_MANIFEST = Path("data/processed/imgflip575k_manifest.json") 
# ----------------------------------------
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
MIN_CAPTION_LENGTH = 10 # Only accept captions 10 characters or longer

def download_image_to_file(url: str, out_path: Path, timeout: int = 15):
    """Download an image from url (or handle local path) and save to out_path."""
    try:
        # local file path (relative or absolute)
        if (not url.startswith("http")) and Path(url).exists():
            img = Image.open(url).convert("RGB")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=90)
            return str(out_path)

        # http(s) download
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=90)
        return str(out_path)
    except Exception:
        return None

def process_json_file(json_path: Path):
    """
    Load JSON, apply strict quality filters (boxes only), and return a list of valid entries.
    """
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    entries = []
    
    for idx, item in enumerate(raw[:200]):
        # 1. Find URL (Required for Visual Input)
        url = item.get("url") or item.get("image_url") or item.get("post") or item.get("img")
        if not url:
            continue

        # 2. STRICT CAPTION FILTERING (ONLY accept ON-IMAGE text from 'boxes' or 'texts')
        boxes = item.get("boxes") or item.get("texts") or []
        
        # If no 'boxes' text is found, we SKIP the entry. 
        # This prevents using external noise from 'title' or 'post'.
        if not isinstance(boxes, list) or not boxes:
            continue
            
        caption = " ".join([str(b).strip() for b in boxes if isinstance(b, str)]).strip()

        # 3. MINIMUM LENGTH FILTER
        if not caption or len(caption) < MIN_CAPTION_LENGTH:
            continue

        # 4. Process and Download
        filename = f"{json_path.stem}_{idx}.jpg"
        out_path = OUT_IMG_DIR / filename

        # Skip download if image already exists
        if out_path.exists():
            entries.append({"image": str(out_path), "caption": caption, "tone": "<humor>"})
            continue

        saved = download_image_to_file(str(url), out_path)
        if saved:
            entries.append({"image": saved, "caption": caption, "tone": "<humor>"})
            
    return entries

def main():
    json_files = sorted(SRC_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files under {SRC_DIR}")

    all_entries = []
    # thread pool for IO-bound downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(process_json_file, jf): jf for jf in json_files}
        # Use a progress bar to track files processed
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing JSON files"):
            res = fut.result()
            if res:
                all_entries.extend(res)

    print(f"Successfully processed {len(all_entries)} high-quality meme entries.")
    # write manifest
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved → {OUT_MANIFEST}")

if __name__ == "__main__":
    main()