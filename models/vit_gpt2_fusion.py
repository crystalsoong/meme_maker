# models/vit_gpt2_fusion.py
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
import torch

def build_model_and_extractor(tokenizer, encoder_name="google/vit-base-patch16-224-in21k", decoder_name="gpt2", device=None):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
    # resize tokenizer embeddings
    model.decoder.resize_token_embeddings(len(tokenizer))
    # set config tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.convert_tokens_to_ids("<bos>")
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    # generation defaults
    model.config.max_length = 32
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 2
    # feature extractor for ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor
