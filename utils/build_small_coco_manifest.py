# utils/build_small_coco_manifest.py

import json
from pathlib import Path
from datasets import load_dataset

def build_small_coco_manifest(
    dataset_name="abisee/coco_captions_small",
    split="train",
    out="data/processed/coco_small_manifest.json",
):
    print(f"Downloading COCO subset: {dataset_name} ({split})")

    ds = load_dataset(dataset_name, split=split)

    img_dir = Path("data/raw/coco_small_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i, item in enumerate(ds):
        img = item["image"]        # Already a PIL image
        captions = item["captions"]

        save_path = img_dir / f"coco_small_{i}.jpg"
        img.save(save_path)

        for c in captions:
            manifest.append({
                "image": str(save_path),
                "caption": c["text"].strip(),
                "tone": "<factual>"
            })

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(manifest)} captions → {out}")
    return out

if __name__ == "__main__":
    build_small_coco_manifest()
