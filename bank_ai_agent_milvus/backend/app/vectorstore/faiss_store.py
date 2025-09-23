"""FAISS-based vector store wrapper using sentence-transformers for embeddings.
- add(id, text): add a document to the index and persist
- search(query, k): return top-k nearest documents

Persisted artifacts:
- index.faiss (binary)
- meta.pkl (list of metadata dicts)

Note: FAISS requires consistent embedding dimension; use the same model for building & querying.
"""
from sentence_transformers import SentenceTransformer
import faiss, os, numpy as np, pickle

class FaissVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2', persist_path='./data/faiss_store'):
        self.model = SentenceTransformer(model_name)
        self.persist_path = persist_path
        os.makedirs(self.persist_path, exist_ok=True)
        self.index_path = os.path.join(self.persist_path, 'index.faiss')
        self.meta_path = os.path.join(self.persist_path, 'meta.pkl')
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.meta = pickle.load(f)
        else:
            self.index = None
            self.meta = []

    def add(self, id: str, text: str):
        # Compute embedding
        emb = self.model.encode([text])
        if self.index is None:
            d = emb.shape[1]
            # Use flat L2 index for simplicity. For production, consider IVF/PQ for scale.
            self.index = faiss.IndexFlatL2(d)
            self.index.add(emb)
            self.meta = [{'id': id, 'text': text}]
        else:
            self.index.add(emb)
            self.meta.append({'id': id, 'text': text})
        # persist to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f)

    def search(self, query: str, k=5):
        emb = self.model.encode([query])
        if self.index is None:
            return []
        D, I = self.index.search(emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results
