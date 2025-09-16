# AI Finance Agent Demo v4 (CPU-friendly)

This v4 package configures the system to support CPU-only training and inference for small models.

Key changes:
- `agent/trainer_cpu.py`: CPU-friendly trainer using HuggingFace Trainer. Select preset `gpt2` or `ko_small`.
- `agent/app.py`: invokes `trainer_cpu.py` for training jobs (non-blocking).
- `agent/generate_train_jsonl.py`: pipeline to build RAG+SFT training JSONL.
- Docker setup updated to CPU-only images (no GPU required).

Quick start (local, CPU):
1. Create venv and install:
   ```
   cd agent
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate training data:
   ```
   python generate_train_jsonl.py --docs agent/data/raw_texts.jsonl --questions demo/questions.jsonl --output demo/generated_train_cpu.jsonl --top_k 3
   ```
3. Train (example):
   ```
   python trainer_cpu.py --model_preset gpt2 --train_file demo/generated_train_cpu.jsonl --output_dir models/cpu_out --epochs 1 --per_device_batch_size 1
   ```
4. Start agent:
   ```
   uvicorn app:app --reload --port 8000
   ```
5. Use frontend or curl to call `/chat` and `/train`.

Notes:
- CPU training is slow; use very small datasets and small base models.
- For Korean-specific models, `skt/kogpt2-base-v2` may be used but check tokenizer compatibility.
- Remove personal data before training.

