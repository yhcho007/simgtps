"""Simple synchronous model server template using HuggingFace Transformers.
- Serves POST /generate with JSON {prompt, max_tokens}
- This is intended for small-scale or internal use. For production scale, use Triton/TF-Serving/custom optimized infra.
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

app = FastAPI()
model = None
tokenizer = None

@app.post('/generate')
def generate(req: GenRequest):
    inputs = tokenizer(req.prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {'text': text}

@app.get('/health')
def health():
    return {'status':'ok'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    uvicorn.run(app, host='0.0.0.0', port=args.port)
