"""
CPU-friendly trainer script (no GPU required).
Defaults to small CPU-friendly base models:
- model_preset 'gpt2' -> "gpt2"
- model_preset 'ko_small' -> "skt/kogpt2-base-v2" (if available)

This script uses HuggingFace Trainer on CPU. Training will be slow but workable for small datasets.
"""
import argparse, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_preset", type=str, choices=["gpt2","ko_small"], default="gpt2")
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()

def build_prompt(ex):
    inst = ex.get("instruction","")
    ctx = ex.get("context","")
    prompt = ex.get("prompt","")
    response = ex.get("response","")
    parts = []
    if inst: parts.append(inst)
    if ctx: parts.append("Context:\\n"+ctx)
    parts.append("User: "+prompt)
    parts.append("Assistant: "+response)
    return "\\n\\n".join(parts)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    preset_map = {"gpt2":"gpt2", "ko_small":"skt/kogpt2-base-v2"}
    base = args.base_model if args.base_model else preset_map.get(args.model_preset,"gpt2")
    print("Using base model:", base)
    ds = load_dataset("json", data_files=args.train_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    model = AutoModelForCausalLM.from_pretrained(base)
    # resize token embeddings if added pad
    model.resize_token_embeddings(len(tokenizer))
    def preprocess(ex):
        full = build_prompt(ex)
        return tokenizer(full, truncation=True, max_length=args.max_length)
    tokenized = ds.map(preprocess, remove_columns=ds.column_names)
    tokenized.set_format(type="torch")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete, saved to", args.output_dir)

if __name__=='__main__':
    main()
