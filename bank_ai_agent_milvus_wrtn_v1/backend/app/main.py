# backend/app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from app.api import milvus as milvus_router
from app.core.config import settings
from app.services.milvus_vector_store import MilvusVectorStore
from app.services.faq_loader import load_faqs_to_milvus
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 Milvus 클라이언트 초기화 및 FAQ 로드
    logger.info("애플리케이션 시작: Milvus 클라이언트 초기화 및 FAQ 로드 중...")
    try:
        milvus_store = MilvusVectorStore()
        # 컬렉션이 없으면 생성 및 FAQ 로드
        if not await milvus_store.check_collection_exists():
            await milvus_store.create_collection()
            await load_faqs_to_milvus(milvus_store)
        else:
            logger.info("Milvus 컬렉션이 이미 존재합니다. 데이터 로드를 건너뜁니다.")
        logger.info("Milvus 클라이언트 초기화 및 FAQ 로드 완료!")
    except Exception as e:
        logger.error(f"애플리케이션 시작 중 오류 발생: {e}", exc_info=True)
    yield
    # 종료 시 Milvus 클라이언트 정리 (필요시)
    logger.info("애플리케이션 종료.")

app = FastAPI(
    title="은행 AI Agent 시스템",
    description="Milvus 벡터 데이터베이스와 통합된 AI Agent 백엔드.",
    version="1.0.0",
    lifespan=lifespan
)

# Milvus 관련 라우터 등록
app.include_router(milvus_router.router, prefix="/milvus", tags=["milvus"])

@app.get("/")
async def read_root():
    return {"message": "어서오세요! 은행 AI Agent API 입니다. /docs 에서 API 문서를 확인하세요!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

