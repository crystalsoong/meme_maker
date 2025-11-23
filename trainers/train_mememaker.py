# trainers/train_mememaker.py
"""
Train script for MemeMaker.
Example:
python trainers/train_mememaker.py --train_manifest data/processed/merged_manifest_train.json --val_manifest data/processed/merged_manifest_val.json --output_dir outputs/mememaker --epochs 3
"""
import argparse
import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils.dataset import MemeCaptionDataset
from utils.collate import collate_fn
from models.vit_gpt2_fusion import build_model_and_extractor
from torchvision import transforms
import wandb
from transformers.integrations import WandbCallback
from transformers import EarlyStoppingCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--output_dir", default="outputs/mememaker")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    special = {"bos_token":"<bos>", "eos_token":"<eos>", "pad_token":"<pad>", "additional_special_tokens":["<humor>","<factual>"]}
    tokenizer.add_special_tokens(special)
    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.08,0.08,0.08,0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # datasets
    train_ds = MemeCaptionDataset(args.train_manifest, tokenizer, transforms=train_transforms, max_length=32)
    val_ds = MemeCaptionDataset(args.val_manifest, tokenizer, transforms=eval_transforms, max_length=32)
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat = build_model_and_extractor(tokenizer, device=device)
    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=500,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=200,
        predict_with_generate=True,
        remove_unused_columns=False,
        save_total_limit=3,
        load_best_model_at_end=True,      # enable best model saving
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0                 # gradient clipping
    )
    # initialize W&B
    wandb.init(project="mememaker", name="mememaker_run")
    # ensure training_args logs to wandb by default (Trainer handles it), add callbacks list
    callbacks = [WandbCallback]
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        callbacks=[WandbCallback, EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved model to", args.output_dir)

if __name__ == "__main__":
    main()
