"""
Trainer CLI supporting two model presets:
- koalpaca: a hypothetical KoAlpaca-compatible base (small/medium models)
- llama2-ko: Llama-2 Korean model (e.g., beomi/llama-2-ko-7b or similar)

Usage:
python trainer.py --model_type koalpaca --base_model <model_name> --train_file demo/sample_train_rag.jsonl --output_dir models/lora_out --epochs 3 --per_device_batch_size 4

Notes:
- Ensure CUDA + bitsandbytes + accelerate configured for large models.
- This script chooses target_modules based on model architecture (OPT/LLAMA).
"""
import argparse, os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, choices=["koalpaca","llama2-ko"], default="koalpaca")
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()

def build_prompt(example):
    # build SFT prompt including instruction, context, prompt
    inst = example.get("instruction","")
    ctx = example.get("context","")
    prompt = example.get("prompt","")
    # Compose: instruction \n context \n user: prompt \n assistant:
    parts = []
    if inst: parts.append(inst.strip())
    if ctx: parts.append("Context:\n"+ctx.strip())
    parts.append("User: " + prompt.strip())
    parts.append("Assistant: " + example.get("response","").strip())
    return "\n\n".join(parts)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...", args.train_file)
    ds = load_dataset("json", data_files=args.train_file)["train"]

    print("Loading tokenizer for", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})

    # load model (try 8-bit if possible)
    use_8bit = False
    try:
        import bitsandbytes as bnb
        use_8bit = True
    except Exception:
        use_8bit = False

    model_kwargs = {}
    if use_8bit:
        print("Loading model in 8-bit")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True, device_map="auto")
    else:
        print("Loading model in fp16 if possible")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    # prepare model
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print("prepare_model_for_kbit_training:", e)

    # choose target_modules heuristically
    if args.model_type == "llama2-ko":
        target_modules = ["q_proj","v_proj"]
    else:
        target_modules = ["q_proj","v_proj","k_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def preprocess(ex):
        full = build_prompt(ex)
        tok = tokenizer(full, truncation=True, max_length=args.max_length)
        return tok

    print("Tokenizing...")
    tokenized = ds.map(preprocess, remove_columns=ds.column_names)
    tokenized.set_format(type="torch")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.lr,
        fp16=not use_8bit and torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    print("Saving...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done")

if __name__ == "__main__":
    main()
