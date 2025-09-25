# backend/app/core/config.py
import os
import platform
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Vector DB 선택 설정 ---
    @property
    def SELECTED_VECTOR_DB(self) -> str:
        os_name = platform.system().lower()
        if os_name == "windows":
            return os.getenv("VECTOR_DB", "chroma")
        elif os_name == "linux":
            return os.getenv("VECTOR_DB", "milvus")
        return os.getenv("VECTOR_DB", "chroma")  # 기본값은 윈도우 환경에 맞춤

    # Milvus 설정
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "bank_faq_collection")
    MILVUS_NLIST: int = 128
    MILVUS_NPROBE: int = 10

    # ChromaDB 설정
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_data")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "bank_faq_collection")

    # 임베딩 모델 관련 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    FORCE_LOCAL_EMBEDDING: bool = os.getenv("FORCE_LOCAL_EMBEDDING", "False").lower() == "true"
    LOCAL_EMBEDDING_MODEL_PATH: str = os.getenv("LOCAL_EMBEDDING_MODEL_PATH",
                                                "./backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab")
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    MILVUS_DIM: int = 768

    # --- PostgreSQL DB 설정 ---
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "agent_db")

    # --- Local LLM 설정 --- (폐쇄망 환경에 맞춰 수정)
    # 로컬 LLM을 위한 API 엔드포인트. 예: Ollama, vLLM, Hugging Face Text Generation Inference
    LOCAL_LLM_API_URL: str = os.getenv("LOCAL_LLM_API_URL",
                                       "http://localhost:8000/v1/chat/completions")  # 실제 LLM 서비스의 엔드포인트로 변경
    LOCAL_LLM_MODEL_NAME: str = os.getenv("LOCAL_LLM_MODEL_NAME", "llama2")  # 사용 중인 로컬 LLM 모델명

    # --- 구글 로그인 (OAuth2) 설정 ---
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_super_secret_key_for_jwt_and_sessions_CHANGEME_IN_PRODUCTION!")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()