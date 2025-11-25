import json
from pathlib import Path

DATA_DIR = Path("data/raw/hateful_memes")
OUT = Path("data/processed/non_hateful_manifest.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

splits = ["train.jsonl", "dev.jsonl", "test.jsonl"]
records = []

for split in splits:
    path = DATA_DIR / split
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)

            # only NON-HATEFUL SAMPLES
            if item.get("label") == 0:
                img_name = item["img"]

                # FIX: JSONL has "img/img123.png", strip leading "img/"
                img_name = img_name.replace("img/", "", 1)

                img_path = DATA_DIR / "img" / img_name

                if img_path.exists():
                    records.append({
                        "image": str(img_path),
                        "caption": item["text"].strip(),
                        "tone": "<humor>"
                    })
                else:
                    print("Missing image:", img_path)

with open(OUT, "w") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(records)} non-hateful entries → {OUT}")
