#!/usr/bin/env python3
"""
Build ImgFlip manifest from JSON files where each JSON is a LIST of meme objects.

Output:
  - Images → data/raw/imgflip575k/images/
  - Manifest → data/processed/imgflip575k_manifest.json
"""
import json
from pathlib import Path
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image
import concurrent.futures

# CONFIG
SRC_DIR = Path("imgflip575k/dataset/memes")           # where your list-JSON files live
OUT_IMG_DIR = Path("data/raw/imgflip575k/images")
OUT_MANIFEST = Path("data/processed/imgflip575k_manifest.json")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def download_image_to_file(url: str, out_path: Path, timeout: int = 15):
    """Download an image from url (or handle local path) and save to out_path. Return str path or None."""
    try:
        # local file path (relative or absolute)
        if (not url.startswith("http")) and Path(url).exists():
            try:
                img = Image.open(url).convert("RGB")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path, format="JPEG", quality=90)
                return str(out_path)
            except Exception:
                return None

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
    Each input JSON is a LIST of meme dicts.
    Return a list of manifest entries (may be empty).
    """
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    entries = []
    for idx, item in enumerate(raw[:50]):   # <-- only take first 50 items
        # find url
        url = item.get("url") or item.get("image_url") or item.get("post") or item.get("img")
        if not url:
            continue

        # caption: prefer boxes -> join, else metadata.title -> post
        boxes = item.get("boxes") or item.get("texts") or []
        if isinstance(boxes, list) and boxes:
            caption = " ".join([str(b).strip() for b in boxes if isinstance(b, str)]).strip()
        else:
            caption = item.get("metadata", {}).get("title") or item.get("post") or item.get("title") or ""
            caption = str(caption).strip()

        if not caption:
            continue

        # create unique filename per item
        filename = f"{json_path.stem}_{idx}.jpg"
        out_path = OUT_IMG_DIR / filename

        # skip if exists
        if out_path.exists():
            entries.append({"image": str(out_path), "caption": caption, "tone": "<humor>"})
            continue

        saved = download_image_to_file(str(url), out_path)
        if saved:
            entries.append({"image": saved, "caption": caption, "tone": "<humor>"})
        # else skip silently for now

    return entries

def main():
    json_files = sorted(SRC_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files under {SRC_DIR}")

    all_entries = []
    # thread pool for IO-bound downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(process_json_file, jf): jf for jf in json_files}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = fut.result()
            if res:
                all_entries.extend(res)

    print(f"Successfully processed {len(all_entries)} meme entries.")
    # write manifest
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"Manifest saved → {OUT_MANIFEST}")

if __name__ == "__main__":
    main()
