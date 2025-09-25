# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "bank_faq_collection")

    # 임베딩 모델 관련 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    # --- 폐쇄망 & 로컬 임베딩 관련 추가 설정 ---
    FORCE_LOCAL_EMBEDDING: bool = os.getenv("FORCE_LOCAL_EMBEDDING",
                                            "False").lower() == "true"  # True면 OpenAI 키가 있어도 로컬 모델 사용 강제
    LOCAL_EMBEDDING_MODEL_PATH: str = os.getenv("LOCAL_EMBEDDING_MODEL_PATH",
                                                "./models/snunlp-SKT-KR-KoBERT-Large-vocab")  # 로컬 모델 저장 경로
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"  # 로컬 임베딩 모델에서 GPU 사용 여부

    # Milvus 컬렉션 필드 설정 (KoBERT 모델 차원 768에 맞춰 조정)
    MILVUS_DIM: int = 768  # KoBERT Large 모델의 임베딩 차원
    MILVUS_NLIST: int = 128  # 인덱스 파라미터 (IVF_FLAT 기준, 데이터셋 크기에 따라 튜닝 필요)

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
