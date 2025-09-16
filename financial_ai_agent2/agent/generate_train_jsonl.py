"""
generate_train_jsonl.py

Usage:
python generate_train_jsonl.py --docs agent/data/raw_texts.jsonl --questions demo/questions.jsonl --output demo/generated_train.jsonl --top_k 3

- docs: lines of {"id":.., "text": ...}
- questions: lines of {"id":.., "prompt": ..., "response": optional}
This script will:
- chunk & embed docs
- build a FAISS index in-memory
- for each question, find top_k relevant doc chunks and include them as 'context' in output JSONL
"""
import argparse, json, os
from chunking import chunk_korean_text
from embeddings import embed_texts
import numpy as np
import faiss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--questions", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
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
    chunks=[]
    metas=[]
    for d in docs:
        txt = d.get('text','')
        chs = chunk_korean_text(txt)
        for i,c in enumerate(chs):
            metas.append({'doc_id':d.get('id'), 'text':c})
            chunks.append(c)
    if not chunks:
        print("No chunks found, exiting")
        return
    print(f"Embedding {len(chunks)} chunks...")
    vecs = embed_texts(chunks)
    vecs = np.array(vecs).astype('float32')
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    questions = load_jsonl(args.questions)
    out=[]
    for q in questions:
        qvec = embed_texts([q.get('prompt','')])
        qvec = qvec.astype('float32')
        faiss.normalize_L2(qvec)
        D,I = index.search(qvec, args.top_k)
        ctxs=[]
        for idx in I[0]:
            if 0 <= idx < len(metas):
                ctxs.append(metas[idx]['text'])
        out_obj = {
            "id": q.get("id"),
            "instruction": q.get("instruction","금융 관련 질문에 답변하세요."),
            "context": "\\n\\n".join(ctxs),
            "prompt": q.get("prompt"),
            "response": q.get("response","")
        }
        out.append(out_obj)
    print(f"Writing {len(out)} items to {args.output}")
    with open(args.output,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\\n")

if __name__ == '__main__':
    main()
