from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def query_db(q):
    vectordb = Chroma(persist_directory="data/vector_store", embedding_function=OpenAIEmbeddings())
    return vectordb.similarity_search(q, k=3)