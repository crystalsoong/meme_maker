# ATTRIBUTION

This project acknowledges and provides detailed attribution for all external resources, code components, libraries, and datasets used, as well as the assistance provided by AI models.

## 1. AI-Assisted Code Generation and Debugging

The dataset initialization, dataset processing, and debugging of this project was done with the assistance of Google Gemini and ChatGPT.

## 2. External Libraries

| Library | Purpose | License / Source |
| :--- | :--- | :--- |
| **PyTorch (`torch`)** | Core framework for tensor operations, neural network layers (`nn.Module`), training loop, and GPU acceleration. | [PyTorch License](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| **Torchvision (`torchvision`)**| Provides image transformations, utilities, and access to pre-trained Vision Transformer (ViT) weights for Transfer Learning. | [Torchvision License](https://github.com/pytorch/vision/blob/main/LICENSE) |
| **NumPy (`numpy`)** | Numerical operations, especially for utility functions like metric smoothing. | [NumPy License](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| **PIL (`Pillow`)** | Image loading and manipulation (e.g., `Image.open`). | [Pillow License](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| **Tqdm (`tqdm`)** | Progress bar visualization for training loops. | [Tqdm License](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| **Matplotlib (`matplotlib`)**| Plotting of training and validation loss/accuracy history. | [Matplotlib License](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| **Pycocoevalcap** | Used for computing standard sequence metrics: CIDEr and BLEU-4. | [Pycocoevalcap GitHub](https://github.com/salaniz/pycocoevalcap) |

## 3. Dataset and Data Processing

| Resource | Purpose | Attribution |
| :--- | :--- | :--- |
| **`imgflip575k_manifest.json`** | Primary dataset containing image file paths and corresponding meme captions/tones. | Custom subset/reformatting of publicly available meme dataset (Source details should be added if publicly hosted). |
| **Image Preprocessing Constants** | Image normalization mean and standard deviation values. | Standard values derived from the ImageNet dataset, commonly used for image classification/generation tasks. |