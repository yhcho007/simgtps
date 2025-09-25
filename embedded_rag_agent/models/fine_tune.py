import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model_name = 'beomi/KoAlpaca-Polyglot-12.8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    dataset = load_dataset('json', data_files=args.dataset)

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=512)
    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir='./models/local_models/ft',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        save_steps=100,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train']
    )
    trainer.train()
    trainer.save_model('./models/local_models/ft')

if __name__ == '__main__':
    main()
