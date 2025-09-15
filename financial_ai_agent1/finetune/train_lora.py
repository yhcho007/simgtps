import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import Trainer
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help='HuggingFace model id or local path')
    parser.add_argument('--dataset_path', type=str, required=True, help='SFT json file (list of {"instruction","input","output"})')
    parser.add_argument('--output_dir', type=str, default='./lora_out')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization with bitsandbytes')
    return parser.parse_args()

def make_prompt(item):
    # Simple instruction -> output format. Adapt to your tokenizer/model special tokens.
    inst = item.get('instruction','')
    inp = item.get('input','')
    if inp:
        prompt = f"### 지시문:\n{inst}\n\n### 입력:\n{inp}\n\n### 응답:\n"
    else:
        prompt = f"### 지시문:\n{inst}\n\n### 응답:\n"
    return prompt

def main():
    args = parse_args()

    # Load dataset (expects a JSON list)
    raw = load_dataset('json', data_files=args.dataset_path)['train']
    # Combine: prompt + output, but for causal LM we will create input_ids = prompt+output and labels = same with -100 for prompt
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_examples(example):
        prompt = make_prompt(example)
        target = example.get('output','')
        full = prompt + target
        tokenized_full = tokenizer(full, truncation=True, max_length=1024)
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=1024)
        # labels: -100 for prompt tokens
        labels = tokenized_full['input_ids'].copy()
        prompt_len = len(tokenized_prompt['input_ids'])
        labels[:prompt_len] = [-100]*prompt_len
        return {'input_ids': tokenized_full['input_ids'], 'attention_mask': tokenized_full['attention_mask'], 'labels': labels}

    tokenized = raw.map(build_examples, remove_columns=raw.column_names)

    # Model load
    if args.use_4bit:
        # 4-bit pathway (requires bitsandbytes)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=True,
            device_map="auto",
            quantization_config=None  # let transformers decide; you can pass BitsAndBytesConfig if needed
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    # PEFT LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj"] if "llama" in args.base_model or "vicuna" in args.base_model else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # TrainingArguments -> use Trainer for simplicity
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=not args.use_4bit,
        save_strategy="epoch",
        push_to_hub=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    # Save only PEFT weights (lightweight)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()