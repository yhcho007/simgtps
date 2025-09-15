import json
from typing import List, Dict
import argparse
import os

"""
입력 예시(자유 형식): 여러 JSON 파일 또는 CSV로부터 로드하여 아래 SFT 형식으로 변환.
SFT format (for causal LM + PEFT training):
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]

OpenAI fine-tune JSONL:
{"prompt": "<PROMPT>", "completion": " <COMPLETE>"}
"""

def convert_to_sft(items: List[Dict], out_path: str):
    """
    items: list of dicts containing keys 'instruction','input','output' (input optional)
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved SFT dataset to {out_path}")

def convert_to_openai_jsonl(items: List[Dict], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            prompt = it.get('instruction', '')
            if it.get('input'):
                prompt += "\n\n" + it['input']
            # OpenAI expects completion to start with a space and end with stop token.
            completion = " " + it.get('output', '')
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + '\n')
    print(f"Saved OpenAI JSONL to {out_path}")

def sample_items():
    return [
        {"instruction": "다음 채무 내역을 분석하고 상환 우선순위를 제안해줘.",
         "input": "채권자: A은행, 잔액: 5,000,000원, 이자율: 12% 연, 월 상환액: 150,000원\n채권자: B카드, 잔액: 2,000,000원, 이자율: 20% 연, 월 상환액: 80,000원",
         "output": "요약: 총 부채 7,000,000원... (예시 응답)"}
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sft','openai'], default='sft')
    parser.add_argument('--out', default='finetune_dataset.json')
    args = parser.parse_args()

    items = sample_items()  # 실제로는 CSV/DB에서 로드하여 items 생성
    if args.mode == 'sft':
        convert_to_sft(items, args.out)
    else:
        convert_to_openai_jsonl(items, args.out)