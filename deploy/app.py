# deploy/app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms

MODEL_DIR = "outputs/mememaker"  # update to your checkpoint after training

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.add_special_tokens({"bos_token":"<bos>", "eos_token":"<eos>", "pad_token":"<pad>", "additional_special_tokens":["<humor>","<factual>"]})
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def generate_memes_pil(image: Image.Image, tone: str = "<humor>", n: int = 3, temp: float = 0.8, top_p: float = 0.9):
    if image is None:
        return ["No image provided."] * n
    img = image.convert("RGB")
    pv = eval_transform(img).unsqueeze(0).to(device)
    # We don't pass decoder prompt tokens explicitly: the model has been trained with tone tokens in captions.
    # To encourage tone conditioning you may fine-tune with tone prefixes and also prompt by seeding decoder input.
    outputs = model.generate(pixel_values=pv, max_length=32, do_sample=True, temperature=temp, top_p=top_p, num_return_sequences=n)
    decoded = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
    return decoded

iface = gr.Interface(fn=generate_memes_pil,
                     inputs=[gr.Image(type="pil"), gr.Radio(["<humor>", "<factual>"], label="Tone"), gr.Slider(1,5, value=3, label="Num captions"), gr.Slider(0.3,1.2, value=0.8, label="Temperature")],
                     outputs=[gr.Textbox() for _ in range(3)],
                     title="MemeMaker: Meme Caption Generator",
                     description="Upload an image and receive meme-style captions.")
if __name__ == "__main__":
    iface.launch(debug=True)
