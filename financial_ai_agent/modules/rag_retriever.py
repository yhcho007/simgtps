import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_retriever(persist_dir="data/vector_store"):
    if not os.path.exists(persist_dir):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever