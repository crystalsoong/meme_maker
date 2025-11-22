# MemeMaker

Multimodal image captioning project to generate meme-style captions.

## Quickstart

1. Prepare environment (see deploy/requirements.txt)
2. Run dataset prep: `python utils/data_utils.py --prepare ...`
3. Train: `python trainers/train_mememaker.py --config utils/configs.py`

## Files
- `data/` - dataset and annotations
- `models/` - model code
- `trainers/` - training scripts
- `deploy/` - app and deployment requirements