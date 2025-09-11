from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def recommend_products(query):
    vectordb = Chroma(persist_directory="data/vector_store", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_type="similarity", k=3)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4"), retriever=retriever, chain_type="stuff")
    return qa.run(query)