from sentence_transformers import SentenceTransformer
import numpy as np
# lightweight multilingual model
EMB_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def embed_texts(texts):
    return EMB_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True)
