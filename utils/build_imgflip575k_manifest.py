#!/usr/bin/env python3
import json
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import concurrent.futures

"""
Creates a full manifest + downloads images for the ImgFlip575K dataset.

INPUT FOLDER (from Kaggle zip extract):
    /home/users/cs785/meme_maker/imgflip575k/dataset/memes/

OUTPUT:
    Images → data/raw/imgflip575k/images/
    Manifest → data/processed/imgflip575k_manifest.json
"""

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DATASET_DIR = Path("imgflip575k/dataset/memes")
OUT_IMG_DIR = Path("data/raw/imgflip575k/images")
OUT_MANIFEST = Path("data/processed/imgflip575k_manifest.json")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# Helper: download image safely
# --------------------------------------------------------------------
def download_image_to_file(url, out_path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(out_path)
        return str(out_path)
    except Exception:
        return None


# --------------------------------------------------------------------
# Process a single JSON meme file
# --------------------------------------------------------------------
def process_meme_json(json_path):
    try:
        with open(json_path, "r") as f:
            item = json.load(f)

        url = item.get("url")
        text_boxes = item.get("boxes", [])

        if not url:
            print("no url")
            return None

        # Build caption string from text boxes
        caption = " ".join([str(t).strip() for t in text_boxes if isinstance(t, str)])
        caption = caption.strip()

        # Unique filename
        filename = json_path.stem + ".jpg"
        out_path = OUT_IMG_DIR / filename

        # Skip if already downloaded
        if out_path.exists():
            return {
                "image": str(out_path),
                "caption": caption,
                "tone": "<humor>"
            }

        # Download image
        saved_path = download_image_to_file(url, out_path)
        if saved_path is None:
            return None

        return {
            "image": saved_path,
            "caption": caption,
            "tone": "<humor>"
        }

    except Exception:
        return None


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    print("Scanning for meme JSON files...")
    json_files = sorted(list(DATASET_DIR.glob("*.json")))
    print(f"Found {len(json_files)} meme items.")

    results = []

    # Multi-threaded image downloading (MUCH faster)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for result in tqdm(executor.map(process_meme_json, json_files), total=len(json_files)):
            if result is not None:
                results.append(result)
            else: print("nothing found")

    print(f"Successfully processed: {len(results)} memes.")

    # Save manifest
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Manifest saved → {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
