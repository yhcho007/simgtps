import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

class LocalLLM:
    def __init__(self, base_model: str = None, lora_weights: str = None, device: str = None):
        self.base_model = base_model or os.getenv('BASE_MODEL')
        self.lora_weights = lora_weights or os.getenv('LORA_WEIGHTS')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if self.lora_weights:
            # load base + apply peft
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            base = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")
            self.model = PeftModel.from_pretrained(base, self.lora_weights)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

