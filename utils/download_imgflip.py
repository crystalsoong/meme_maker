import os
from pathlib import Path
import zipfile
import subprocess
import build_imgflip575k_manifest

raw_dir = Path("data/raw/imgflip")
raw_dir.mkdir(parents=True, exist_ok=True)

print("Downloading ImgFlip575K from Kaggle…")

cmd = [
    "kaggle", "datasets", "download",
    "-d", "schesa/imgflip575k",
    "-p", str(raw_dir)
]

subprocess.run(cmd, check=True)
print("Download complete.")

zip_files = list(raw_dir.glob("*.zip"))
if not zip_files:
    raise FileNotFoundError("No zip file downloaded!")

zip_path = zip_files[0]
print("Unzipping:", zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(raw_dir)

print("Unzip complete.")

csvs = list(raw_dir.glob("*.csv"))
if not csvs:
    raise FileNotFoundError("No CSV found in ImgFlip folder!")

csv_path = csvs[0]
print("Building manifest from:", csv_path)

manifest = build_imgflip_manifest(
    imgflip_dir=raw_dir,
    out_manifest="data/processed/imgflip_manifest.json",
    image_col_candidates=["image_url", "img_url", "url"],
    caption_col_candidates=["text_0", "text", "caption"]
)

print("Saved manifest:", manifest)
