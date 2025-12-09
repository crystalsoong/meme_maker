**MemeMaker**
Meme Maker is an image-captioning model that generates humorous, meme-style captions from input images. It combines a Vision Transformer–style encoder with a transformer decoder to produce context-aware, funny one-liners.

**What it does**
The Meme Maker Model takes inputted images and produces funny captions in an end-to-end pipeline: it preprocesses and augments image data, encodes images with a patch-based vision encoder, and autoregressively decodes captions using a transformer decoder trained on a large meme-caption dataset. The project uses train/val/test splits, configurable optimizers and schedulers, checkpointing, and early stopping; evaluation and diagnostic tools such as loss/accuracy plotting, first-token statistics, BLEU/CIDEr hooks; inference support with multiple decoding strategies, such as deterministic greedy, nucleus sampling, beam search. It also saves tokenizer mappings and experiment configs to ensure reproducibility and easy deployment.


**Quick Start**
1. Clone the Repository
2. Install any dependencies 
- ```pip install torch torchvision torchaudio numpy matplotlib Pillow tqdm```
- ```pip install pycocoevalcap```

If training the model:
1. Run ```python3 utils/build_imgflip575k_manifest.py``` to get the raw and processed data used to train the model
2. Go to the ```real_train_vit.py``` file within the models folder and turn TRAIN to True
3. Run ```python3 models/real_train_vit.py``` to train the model

If evaluating the model:
1. Go to the ```real_train_vit.py``` file within the models folder and turn TRAIN to False
2. Run ```python3 models/real_train_vit.py``` to run and evaluate the model

The Model Evaluation corresponding meme images used in the 10 example outputs given to the model for testing is found in the ```eval/image_examples``` folder
The training and validation curve plots for each experiment are found in the ```eval/plots``` folder

To test the model with your own image, import the image into the ```test_images``` folder and run ```python3 eval/predict_image.py test_images/***YOUR IMAGE NAME***``` to generate a corresponding caption

**Video Links**
Project Demo: This is a 3–5-minute video that shows what your project does and why that matters. Think of this as the video you would show a non-specialist to pitch what you built. There is no reason to show any code in this video – you can use slides with visualizations or diagrams to provide motivation, show the running application, show experimental results, etc. This video should be included in your repository and linked prominently in your README file.
Technical Walkthrough: This is a 5–10-minute video that shows and discusses how your project works. Think of this as the video you would show a fellow ML engineer to explain how you accomplished what you did. This video should help orient a grader to understand how your code works and where the machine learning concepts are being applied. It should also help a grader understand what was challenging about the project and where the significant technical contributions can be found. Like the project demo, this video should be stored in your repository and clearly linked in your README file.

**Evaluation**
an Evaluation section that presents any quantitative results, accuracy metrics, or qualitative outcomes from testing,

**Individual Contributions**
Crystal Soong and Alan Lu contributed equally to the brainstorming, development, testing, and documentation of the Meme Maker Model. The work was split evenly and both partners worked together to complete the project. 