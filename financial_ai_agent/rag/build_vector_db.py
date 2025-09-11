import os
import shutil
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def build_vector_db():
    persist_dir = os.path.join("data", "vector_store")

    # 기존 DB 삭제 (중복 방지)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("🗑️ 기존 Vector DB 삭제 완료")

    # 문서 로드
    loader = TextLoader(os.path.join("data", "product_docs", "products.txt"), encoding="utf-8")
    docs = loader.load()

    # 다국어 한국어 지원 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 새로운 벡터 DB 생성
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("✅ 새로운 Vector DB build success (multilingual embeddings 사용)")


if __name__ == "__main__":
    build_vector_db()