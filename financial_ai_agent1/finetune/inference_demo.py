import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def guess_target_modules(model_name: str):
    """
    모델 이름에 따라 LoRA target_modules 추천
    """
    name = model_name.lower()
    if "llama" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gptj" in name or "gpt-j" in name:
        return ["q_proj", "v_proj"]
    elif "gptneo" in name or "neo" in name:
        return ["q_proj", "v_proj"]
    elif "bloom" in name:
        return ["query_key_value"]
    else:
        # 기본값
        return ["q_proj", "v_proj"]


def load_model(base_model: str, lora_path: str = None, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model: {base_model} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    if lora_path:
        print(f"Applying LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA target_modules (추정):", guess_target_modules(base_model))

    return tokenizer, model


def infer(prompt: str, tokenizer, model, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    BASE_MODEL = "MLP-KTLim/llama-3-Korean-SFT-1.8B"  # 추천 한국어 SFT 모델
    LORA_WEIGHTS = "./finetune/lora_adapter"  # train_lora.py 결과물 (없으면 None)

    tokenizer, model = load_model(BASE_MODEL, LORA_WEIGHTS)

    while True:
        q = input("\n질문을 입력하세요 (종료: exit): ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        result = infer(q, tokenizer, model)
        print("\n=== 응답 ===")
        print(result)