# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "bank_faq_collection")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME",
                                          "text-embedding-3-small")  # OpenAI embedding model name

    # Milvus 컬렉션 필드 설정
    MILVUS_DIM: int = 1536  # OpenAI text-embedding-3-small의 기본 차원
    MILVUS_NLIST: int = 128  # 인덱스 파라미터

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
