# Project Setup and Installation Guide

This guide provides clear instructions for setting up the environment, preparing the data, and running the Image Captioning Model project.

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

Model Predictions:
FINISH