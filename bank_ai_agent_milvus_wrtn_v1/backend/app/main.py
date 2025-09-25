# backend/app/main.py
import platform
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse  # CORS 테스트용
from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 추가
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Optional, Dict, Any

from app.api import agent as agent_router
from app.api import auth as auth_router
from app.api import feedback as feedback_router
from app.api import dashboard as dashboard_router
from app.core.config import settings
from app.core.common_vector_store import AbstractVectorStore
from app.core.database import connect_to_db, close_db_connection  # DB 연결 모듈 임포트
from app.services.milvus_vector_store import MilvusVectorStore
from app.services.chroma_vector_store import ChromaVectorStore
from app.services.faq_loader import load_faqs_to_milvus
from app.services.session_manager import SessionManager
from app.services.stats_manager import StatsManager
from app.services.background_tasks import start_background_fine_tuning, start_session_db_saver
from app.core.embeddings import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

global_vector_store: Optional[AbstractVectorStore] = None
session_manager: Optional[SessionManager] = None
stats_manager: Optional[StatsManager] = None


# Dependency Injection을 위한 헬퍼 함수
async def get_vector_store() -> AbstractVectorStore:
    if global_vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return global_vector_store


async def get_session_manager() -> SessionManager:
    if session_manager is None:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    return session_manager


async def get_stats_manager() -> StatsManager:
    if stats_manager is None:
        raise HTTPException(status_code=500, detail="Stats manager not initialized")
    return stats_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- 애플리케이션 시작 ---")

    # 0. PostgreSQL DB 연결
    await connect_to_db()

    # 1. 임베딩 모델 로드
    try:
        global_embedding_model = get_embedding_model()
        logger.info(
            f"임베딩 모델 로드 완료: {global_embedding_model.__class__.__name__}, 차원: {global_embedding_model.dimension}")
    except Exception as e:
        logger.error(f"임베딩 모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
        raise RuntimeError("임베딩 모델 로드 실패, 애플리케이션 종료.") from e

    # 2. Vector Store 초기화 (OS에 따라 선택)
    global global_vector_store
    collection_name = ""
    if settings.SELECTED_VECTOR_DB == "milvus":
        global_vector_store = MilvusVectorStore()
        logger.info("Milvus Vector Store 초기화 예정...")
        collection_name = settings.MILVUS_COLLECTION_NAME
    elif settings.SELECTED_VECTOR_DB == "chroma":
        global_vector_store = ChromaVectorStore()
        logger.info("ChromaDB Vector Store 초기화 예정...")
        collection_name = settings.CHROMA_COLLECTION_NAME
    else:
        raise ValueError(f"알 수 없는 VECTOR_DB: {settings.SELECTED_VECTOR_DB}")

    # Vector Store 연결 확인 및 FAQ 로드
    try:
        await global_vector_store.check_connection()
        if not await global_vector_store.check_collection_exists(collection_name):
            logger.info(f"{settings.SELECTED_VECTOR_DB} 컬렉션 '{collection_name}'이 존재하지 않습니다. 새로 생성합니다.")
            await global_vector_store.create_collection(collection_name, global_embedding_model.dimension)
            await load_faqs_to_milvus(global_vector_store, global_embedding_model)
        else:
            logger.info(f"{settings.SELECTED_VECTOR_DB} 컬렉션 '{collection_name}'이 이미 존재합니다. 데이터 로드를 건너뜜.")
        logger.info(f"{settings.SELECTED_VECTOR_DB} 클라이언트 초기화 및 FAQ 로드 완료!")
    except Exception as e:
        logger.error(f"{settings.SELECTED_VECTOR_DB} 초기화 또는 FAQ 로드 중 오류 발생: {e}", exc_info=True)
        raise RuntimeError(f"{settings.SELECTED_VECTOR_DB} 초기화 실패, 애플리케이션 종료.") from e

    # 3. Session Manager 및 Stats Manager 초기화 (DB 로드)
    global session_manager, stats_manager
    session_manager = SessionManager()
    stats_manager = StatsManager()
    stats_manager.set_session_manager(session_manager)  # 순환 참조 주의 (DI가 더 적합)

    await session_manager.load_all_sessions_from_db()
    await stats_manager.load_all_feedback_from_db()  # 기존 피드백 로드하여 통계 계산

    logger.info(
        f"초기 세션 {len(session_manager.get_all_active_sessions())}개, 피드백 {len(stats_manager.get_all_feedback())}개 로드 완료.")

    # 4. 백그라운드 Fine-tuning 및 DB 저장 작업 시작
    logger.info("백그라운드 Fine-tuning 및 세션 DB 저장 스케줄러 시작...")
    asyncio.create_task(start_background_fine_tuning(stats_manager))  # stats_manager 전달
    asyncio.create_task(start_session_db_saver(session_manager))  # session_manager 전달

    yield  # 여기서 애플리케이션이 요청을 처리합니다.

    # 5. 애플리케이션 종료 시 정리 작업
    logger.info("--- 애플리케이션 종료 ---")
    await close_db_connection()  # DB 연결 종료


app = FastAPI(
    title="폐쇄망 멀티모달 & 지능형 AI Agent 시스템",
    description="OS별 VectorDB, 로컬 임베딩, RAG, 멀티모달, 로그인, 대시보드, DB 연동을 지원하는 AI Agent.",
    version="4.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 추가 (프론트엔드 연동을 위해)
# 실제 배포 시에는 allowed_origins를 프론트엔드 도메인으로 한정해야 합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서비스 추가 (멀티모달 응답용)
app.mount("/static", StaticFiles(directory="backend/app/data/documents"), name="static")

# 라우터 등록
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(feedback_router.router, prefix="/feedback", tags=["feedback"])
app.include_router(dashboard_router.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(agent_router.router, prefix="/agent", tags=["ai_agent"],
                   dependencies=[Depends(get_current_user)])  # 에이전트 API는 로그인 필요


@app.get("/", response_class=HTMLResponse)  # CORS 테스트용 HTML 응답 추가
async def read_root():
    return """
    <html>
        <head>
            <title>AI Agent Backend</title>
        </head>
        <body>
            <h1>Welcome to the AI Agent Backend!</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
            <p>For React frontend, access <a href="http://localhost:3000">http://localhost:3000</a></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
