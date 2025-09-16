"""
generate_train_jsonl.py - CPU friendly pipeline to create RAG+SFT JSONL

Usage:
python generate_train_jsonl.py --docs agent/data/raw_texts.jsonl --questions demo/questions.jsonl --output demo/generated_train_cpu.jsonl --top_k 3
"""
import argparse, json
from chunking import chunk_korean_text
from embeddings import embed_texts
import numpy as np
import faiss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()

def load_jsonl(path):
    items=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def main():
    args = parse_args()
    docs = load_jsonl(args.docs)
    chunks=[]; metas=[]
    for d in docs:
        chs = chunk_korean_text(d.get('text',''))
        for c in chs:
            metas.append({'text':c})
            chunks.append(c)
    if not chunks:
        print("no chunks"); return
    vecs = embed_texts(chunks).astype('float32')
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    qs = load_jsonl(args.questions)
    out=[]
    for q in qs:
        qvec = embed_texts([q.get('prompt','')]).astype('float32')
        faiss.normalize_L2(qvec)
        D,I = idx.search(qvec, args.top_k)
        ctxs=[]
        for ii in I[0]:
            if 0<=ii<len(metas): ctxs.append(metas[ii]['text'])
        out.append({
            "id": q.get("id"),
            "instruction": q.get("instruction","금융 관련 질문에 답변하세요."),
            "context": "\n\n".join(ctxs),
            "prompt": q.get("prompt"),
            "response": q.get("response","")
        })
    with open(args.output,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", args.output)

if __name__=='__main__':
    main()
