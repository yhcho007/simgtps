# AI Finance Agent Demo v3 — Full Usage Guide

This distribution includes:
- Frontend (Next.js) demo
- Gateway (Express) proxy
- Agent (FastAPI) with ingestion, chat, training trigger endpoints
- Trainer scripts supporting KoAlpaca and Llama-2-ko model types using HuggingFace + PEFT (LoRA)
- GPU-enabled Dockerfile (nvidia/cuda) and docker-compose.gpu.yml
- Pipeline automation script `generate_train_jsonl.py` to create RAG+SFT JSONL training data from document corpus

Prerequisites
- GPU server with NVIDIA drivers and Docker + nvidia-docker runtime (for GPU containers)
- Python 3.10+ for local running
- Adequate disk and memory for model downloads

Quick local run (CPU / small models)
1. Create virtualenv and install:
   ```bash
   cd agent
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare training data:
   ```bash
   python generate_train_jsonl.py --docs agent/data/raw_texts.jsonl --questions demo/questions.jsonl --output demo/generated_train.jsonl --top_k 3
   ```
3. Train (example using a small KoAlpaca-style base):
   ```bash
   python trainer.py --model_type koalpaca --base_model username/koalpaca-small --train_file demo/generated_train.jsonl --output_dir models/lora_out --epochs 1 --per_device_batch_size 1
   ```
   Adjust `--base_model` to a real model ID you have access to.

Docker (GPU)
1. Build and run with nvidia runtime:
   ```bash
   docker compose -f docker-compose.gpu.yml up --build
   ```
2. Trigger training via API:
   ```bash
   curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"base_model":"facebook/opt-1.3b","train_file":"demo/generated_train.jsonl","output_dir":"models/lora_out","epochs":1,"batch_size":1}'
   ```

Notes and safety
- Always remove PII before training. This demo does NOT automatically mask PII.
- Training large models requires significant GPU memory and proper accelerate config.
- This code is provided as a PoC. Harden for production: add authentication, logging, monitoring, retry/backoff, and secure storage for models.

