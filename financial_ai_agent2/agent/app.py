from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, threading, os, json
from chunking import chunk_korean_text
from embeddings import embed_texts
from vector_db import FaissDB
import numpy as np

app = FastAPI()
DB = FaissDB(dim=384)
train_proc = {"running":False, "last":None}

class TrainReq(BaseModel):
    model_preset: str = "gpt2"
    train_file: str = "demo/generated_train_cpu.jsonl"
    output_dir: str = "models/lora_out"
    epochs: int = 1
    batch_size: int = 1

@app.post("/train")
def start_train(req: TrainReq):
    if train_proc["running"]:
        raise HTTPException(status_code=400, detail="Training already running")
    cmd = ["python", "trainer_cpu.py", "--model_preset", req.model_preset, "--train_file", req.train_file, "--output_dir", req.output_dir, "--epochs", str(req.epochs), "--per_device_batch_size", str(req.batch_size)]
    def _run():
        train_proc["running"]=True
        try:
            subprocess.run(cmd, check=True)
            train_proc["last"]={"status":"finished","output_dir":req.output_dir}
        except Exception as e:
            train_proc["last"]={"status":"error","error":str(e)}
        train_proc["running"]=False
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status":"started"}

@app.get("/train/status")
def status():
    return train_proc

class ChatReq(BaseModel):
    text: str

@app.post("/chat")
def chat(req: ChatReq):
    chunks = chunk_korean_text(req.text)
    vecs = embed_texts(chunks)
    q = np.mean(vecs, axis=0, keepdims=True)
    hits = DB.search(q, k=3)
    context = [h.get('text', h.get('content','')) for h in (hits[0] if hits else [])]
    # If a trained model exists, we could load and run it; here we return mock answer
    answer = "[CPU Demo answer] 질문: {}\\n상세: {}".format(req.text, context[:2])
    return {"answer":answer, "context":context}
