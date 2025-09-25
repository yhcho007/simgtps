# backend/app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from app.api import milvus as milvus_router
from app.api import agent as agent_router  # 새롭게 추가될 에이전트 라우터
from app.core.config import settings
from app.services.milvus_vector_store import MilvusVectorStore
from app.services.faq_loader import load_faqs_to_milvus
from app.core.embeddings import get_embedding_model  # 임베딩 모델 가져오기
from fastapi.staticfiles import StaticFiles # 추가
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 앱 시작 시 필요한 전역 리소스 초기화
# 비동기 context manager 사용하여 앱 시작/종료 시 작업 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- 애플리케이션 시작 ---")

    # 1. 임베딩 모델 로드 (폐쇄망 환경을 위해 로컬 모델 우선 로드)
    try:
        global_embedding_model = get_embedding_model()  # 전역에서 사용할 모델 로드
        logger.info(
            f"임베딩 모델 로드 완료: {global_embedding_model.__class__.__name__}, 차원: {global_embedding_model.dimension}")
    except Exception as e:
        logger.error(f"임베딩 모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
        # 모델 로드 실패 시 애플리케이션 시작을 막을 수도 있습니다.
        # raise RuntimeError("임베딩 모델 로드 실패, 애플리케이션 종료.") from e

    # 2. Milvus 클라이언트 초기화 및 FAQ 로드
    logger.info("Milvus 클라이언트 초기화 및 FAQ 로드 중...")
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
        logger.error(f"Milvus 초기화 또는 FAQ 로드 중 오류 발생: {e}", exc_info=True)

    yield  # 여기서 애플리케이션이 요청을 처리합니다.

    # 앱 종료 시 정리 작업
    logger.info("--- 애플리케이션 종료 ---")


app = FastAPI(
    title="폐쇄망 멀티모달 AI Agent 시스템 (은행 특화)",
    description="Milvus, 로컬 임베딩 모델, 다중 RAG 소스 및 멀티모달 응답을 지원하는 AI Agent 백엔드.",
    version="2.0.0",
    lifespan=lifespan
)

# --- 정적 파일 서비스 추가 ---
# /static 경로로 접근 시 backend/app/data/documents 폴더의 파일들을 제공합니다.
app.mount("/static", StaticFiles(directory="backend/app/data/documents"), name="static")

# Milvus 관련 라우터 등록
app.include_router(milvus_router.router, prefix="/milvus", tags=["milvus"])
# AI Agent 핵심 라우터 등록 (새로운 기능)
app.include_router(agent_router.router, prefix="/agent", tags=["ai_agent"])


@app.get("/")
async def read_root():
    return {"message": "어서오세요! 폐쇄망 멀티모달 AI Agent API 입니다. /docs 에서 API 문서를 확인하세요!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)