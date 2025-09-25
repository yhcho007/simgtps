# Embedded Multimodal RAG Agent (Offline / Air-gapped Ready) - Scaffold

## Overview
This repository is a scaffold for running an **offline, embedded-model** based multimodal RAG (Retrieval-Augmented Generation) chatbot suitable for air-gapped networks. It includes:
- Local embedding model hooks (sentence-transformers or local ONNX/Torch models)
- Milvus vector search integration for RAG (assumes Milvus is available inside the network)
- Multimodal response support (PDF, images) through file attachments + metadata
- Fine-tuning orchestration hooks for embedding model, with a simple "leaderboard" to track resulting models and promote best models to production
- Simple Flask API (app.py) exposing endpoints for chat, fine-tune, model status, and manual promotion
- Postman / curl examples for testing

> NOTE: This is a scaffold and NOT a drop-in full production system. It's intentionally detailed with comments to guide adaptation for your environment and local models (PyTorch, ONNX, or other frameworks).

## Files
- `app.py` - Flask server that exposes endpoints for chat, upload documents, start fine-tune job, check leaderboard, and promote models.
- `agent.py` - Core agent code implementing RAG, multimodal packaging, and model selection logic.
- `models/embedder.py` - Local embedding interface (supports sentence-transformers, ONNX or other local model).
- `models/fine_tune.py` - Script to orchestrate fine-tuning embeddings (placeholder; adapt to your training infra).
- `vector/milvus_client.py` - Minimal Milvus client wrapper for indexing and search.
- `storage/leaderboard.json` - Example leaderboard that tracks fine-tune jobs and metrics.
- `tests/postman_examples.txt` - curl + Postman collection fragments for testing endpoints.
- `sample_client_android.md` - Notes for integrating with Android or web front-ends.
- `requirements.txt` - Python dependencies (choose to install within your closed network).
- `LICENSE` - MIT

## How to adapt for fully offline use
1. Provide local models: put PyTorch/ONNX weights inside `models/local_models/` and set configuration in `models/embedder.py`.
2. Install Milvus within your internal network (or embedded vector DB like FAISS on disk if Milvus isn't allowed). Adjust `vector/milvus_client.py` accordingly.
3. Replace the placeholder generator in `agent.py` with a local LLM (e.g. Llama family using llama.cpp, GGML, or a local server) that you can call via subprocess or local socket.
4. For multimodal responses, store PDFs/images on internal file shares and return metadata + file paths from the API for the client to fetch.
5. For fine-tuning, adapt `models/fine_tune.py` to your training cluster or to use CPU-only PyTorch with small datasets. The example uses file-based "jobs".

## Quick start (example)
1. Install dependencies from `requirements.txt` (inside your offline env).
2. Start Milvus (or adapt to FAISS).
3. Run `python3 app.py` to start the Flask API.
4. Use the included curl examples in `tests/postman_examples.txt` to test operations.

## Security & Privacy notes
- Since the environment is air-gapped, ensure model files and Milvus are stored on secure storage.
- Do not enable outbound connections from the machine that runs models unless intentionally required.

