"""Build FAISS index from docs in docs/policy/*.txt
- Reads text files, generates embeddings, and stores index under data/faiss_store
"""
import os
from app.vectorstore.faiss_store import FaissVectorStore
from app.vectorstore.milvus_store import MilvusClient

if __name__ == '__main__':
    # store = FaissVectorStore()
    store = MilvusClient()
    docs_dir = os.path.join(os.getcwd(), '..', '..', 'docs', 'policy')
    if not os.path.exists(docs_dir):
        print('No docs found at', docs_dir)
    else:
        for fname in os.listdir(docs_dir):
            if fname.endswith('.txt'):
                path = os.path.join(docs_dir, fname)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                store.add(id=fname, text=text)
                print('Added', fname)
    if isinstance(store, FaissVectorStore):
        print('FAISS build complete')
    else:
        print('Milvus build complete')
