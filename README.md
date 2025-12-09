**MemeMaker**
Meme Maker is an image-captioning model that generates humorous, meme-style captions from input images. It combines a Vision Transformer–style encoder with a transformer decoder to produce context-aware, funny one-liners.

**What it does**
The Meme Maker Model takes inputted images and produces funny captions in an end-to-end pipeline: it preprocesses and augments image data, encodes images with a patch-based vision encoder, and autoregressively decodes captions using a transformer decoder trained on a large meme-caption dataset. The project uses train/val/test splits, configurable optimizers and schedulers, checkpointing, and early stopping; evaluation and diagnostic tools such as loss/accuracy plotting, first-token statistics, BLEU/CIDEr hooks; inference support with multiple decoding strategies, such as deterministic greedy, nucleus sampling, beam search. It also saves tokenizer mappings and experiment configs to ensure reproducibility and easy deployment.


**Quick Start**
a Quick Start section that concisely explains how to run your project,


**Video Links**
a Video Links section with direct links to your demo and technical walkthrough videos,

**Evaluation**
an Evaluation section that presents any quantitative results, accuracy metrics, or qualitative outcomes from testing,

**Individual Contributions**
Crystal Soong and Alan Lu contributed equally to the brainstorming, development, testing, and documentation of the Meme Maker Model. The work was split evenly and both partners worked together to complete the project. 