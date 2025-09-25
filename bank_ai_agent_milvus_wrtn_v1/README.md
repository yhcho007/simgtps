## 🚀 프로젝트 구조 (업데이트)

더 많은 기능들이 추가되면서 프로젝트 구조가 더 체계적으로 변했어요.

```
.
├── backend
│   ├── app
│   │   ├── api
│   │   │   ├── agent.py
│   │   │   ├── auth.py
│   │   │   ├── feedback.py
│   │   │   └── dashboard.py
│   │   ├── core
│   │   │   ├── config.py
│   │   │   ├── database.py        # PostgreSQL DB 연결 관리 (신규)
│   │   │   ├── embeddings.py
│   │   │   ├── security.py
│   │   │   ├── tools.py
│   │   │   └── common_vector_store.py
│   │   ├── data
│   │   │   ├── faq.json
│   │   │   ├── documents
│   │   │   │   ├── bank_loan_guide.pdf
│   │   │   │   └── bank_card_benefits.png
│   │   │   └── models             # 로컬 임베딩 모델 저장 폴더
│   │   │       └── snunlp-SKT-KR-KoBERT-Large-vocab
│   │   ├── services
│   │   │   ├── chroma_vector_store.py
│   │   │   ├── milvus_vector_store.py
│   │   │   ├── faq_loader.py
│   │   │   ├── response_generator.py # 폐쇄망 LLM 통합 및 멀티모달 처리
│   │   │   ├── background_tasks.py # 백그라운드 작업 (Fine-tuning, DB 저장)
│   │   │   ├── session_manager.py # DB 연동으로 수정
│   │   │   └── stats_manager.py   # DB 연동으로 수정
│   │   ├── __init__.py
│   │   └── main.py
│   └── Dockerfile
├── scripts
│   ├── run_windows.bat
│   └── run_linux.sh
├── requirements.txt
├── .env.example
├── db_schema.sql                  # PostgreSQL 데이터베이스 스키마 (신규)
├── frontend                       # React 프론트엔드 (신규 디렉토리)
│   ├── public
│   ├── src
│   │   ├── api
│   │   │   └── axiosInstance.js   # 백엔드 API 호출을 위한 axios 인스턴스
│   │   ├── assets
│   │   │   ├── logo.svg
│   │   │   └── like.svg           # 피드백 아이콘 등
│   │   ├── components
│   │   │   ├── AuthGoogle.js      # Google 로그인 버튼 컴포넌트
│   │   │   ├── ChatContainer.js   # 채팅 메시지들을 렌더링
│   │   │   ├── ChatInput.js       # 메시지 입력 필드
│   │   │   ├── ChatMessage.js     # 개별 채팅 메시지 (멀티모달 렌더링 포함)
│   │   │   └── FeedbackButtons.js # 좋아요/싫어요 버튼
│   │   ├── context
│   │   │   └── AuthContext.js     # 인증(JWT) 상태 관리 컨텍스트
│   │   ├── hooks
│   │   │   └── useChatSession.js  # 채팅 세션 로직 (ID 생성, 기록 관리)
│   │   ├── pages
│   │   │   ├── AuthPage.js        # 로그인 페이지
│   │   │   └── ChatPage.js        # 메인 챗봇 페이지
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
└── README.md
```

---

## 🛠️ 백엔드 주요 코드 수정 및 추가 내용

### 1. `db_schema.sql` - PostgreSQL 데이터베이스 스키마 (새 파일)

세션 정보와 피드백 데이터를 저장할 테이블 정의입니다. 데이터베이스를 생성하고 이 스키마를 적용해야 합니다.

```sql
-- db_schema.sql

-- 사용자 세션 테이블
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    messages JSONB DEFAULT '[]'::jsonb, -- 채팅 히스토리 (JSONB 타입)
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 추가 (조회 성능 향상)
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_email ON chat_sessions(user_email);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions(last_activity);


-- 사용자 피드백 테이블
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_email VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) REFERENCES chat_sessions(session_id) ON DELETE SET NULL, -- 세션 삭제 시 NULL로 설정
    message_id VARCHAR(255) NOT NULL, -- 특정 응답 메시지 ID (프론트엔드에서 생성)
    feedback_type VARCHAR(50) NOT NULL, -- 'like', 'dislike'
    user_query TEXT NOT NULL,
    agent_response TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_feedback_user_email ON feedback(user_email);
CREATE INDEX IF NOT EXISTS idx_feedback_session_id ON feedback(session_id);

```

### 2. `requirements.txt` (업데이트)

PostgreSQL 연동을 위한 `psycopg2-binary`와 비동기 처리를 위한 `asyncio-throttle`를 추가합니다.

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -7,3 +7,4 @@
 transformers
 torch
 sentence-transformers
+psycopg2-binary # For PostgreSQL connection
+asyncio-throttle # For rate limiting (optional but good for LLM calls)
```

### 3. `backend/app/core/config.py` (업데이트)

PostgreSQL 연결 정보를 추가하고, `LOCAL_LLM_API_URL` 등 폐쇄망 LLM 관련 설정을 정의합니다.

```python
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
        return os.getenv("VECTOR_DB", "chroma") # 기본값은 윈도우 환경에 맞춤

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
    LOCAL_EMBEDDING_MODEL_PATH: str = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "./backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab")
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
    LOCAL_LLM_API_URL: str = os.getenv("LOCAL_LLM_API_URL", "http://localhost:8000/v1/chat/completions") # 실제 LLM 서비스의 엔드포인트로 변경
    LOCAL_LLM_MODEL_NAME: str = os.getenv("LOCAL_LLM_MODEL_NAME", "llama2") # 사용 중인 로컬 LLM 모델명
    
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
```

### 4. `backend/app/core/database.py` - PostgreSQL DB 연결 관리 (새 파일)

FastAPI 애플리케이션의 `lifespan`에서 DB 연결 풀을 초기화하고 종료할 수 있도록 `psycopg2` 기반의 연결 매니저를 구현합니다. 비동기 처리를 위해 `asyncio.to_thread`를 사용할 것입니다.

```python
# backend/app/core/database.py
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from app.core.config import settings
import logging
import asyncio # for asyncio.to_thread

logger = logging.getLogger(__name__)

# 전역 DB 연결 풀
db_pool: Optional[SimpleConnectionPool] = None

async def connect_to_db():
    global db_pool
    if db_pool is None:
        try:
            db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10, # 필요에 따라 커넥션 풀 크기 조절
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB
            )
            # 연결 테스트
            conn = await asyncio.to_thread(db_pool.getconn)
            cursor = await asyncio.to_thread(conn.cursor)
            await asyncio.to_thread(cursor.execute, "SELECT 1")
            await asyncio.to_thread(cursor.close)
            await asyncio.to_thread(db_pool.putconn, conn)
            logger.info("PostgreSQL 데이터베이스에 성공적으로 연결되었습니다.")
        except Exception as e:
            logger.error(f"PostgreSQL 데이터베이스 연결 실패: {e}", exc_info=True)
            raise RuntimeError(f"PostgreSQL 연결 실패: {e}")

async def close_db_connection():
    global db_pool
    if db_pool:
        logger.info("PostgreSQL 데이터베이스 연결을 종료합니다.")
        await asyncio.to_thread(db_pool.closeall)
        db_pool = None

async def get_db_connection():
    if db_pool is None:
        raise RuntimeError("데이터베이스 연결 풀이 초기화되지 않았습니다.")
    # Thread pool executor를 사용하여 동기 psycopg2 호출을 비동기로 실행
    return await asyncio.to_thread(db_pool.getconn)

async def release_db_connection(conn):
    if db_pool and conn:
        await asyncio.to_thread(db_pool.putconn, conn)

async def execute_query(query: str, params: Optional[tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
    conn = None
    try:
        conn = await get_db_connection()
        # NamedTupleCursor를 사용하여 결과가 컬럼 이름으로 접근 가능한 객체로 반환되도록 함
        cursor = await asyncio.to_thread(conn.cursor, cursor_factory=psycopg2.extras.NamedTupleCursor)
        await asyncio.to_thread(cursor.execute, query, params)
        if fetch_one:
            result = await asyncio.to_thread(cursor.fetchone)
        elif fetch_all:
            result = await asyncio.to_thread(cursor.fetchall)
        else:
            result = None
        await asyncio.to_thread(conn.commit) # DDL/DML 후에 커밋
        await asyncio.to_thread(cursor.close)
        return result
    except Exception as e:
        if conn:
            await asyncio.to_thread(conn.rollback) # 오류 발생 시 롤백
        logger.error(f"DB 쿼리 실행 중 오류 발생: {e}, 쿼리: {query}", exc_info=True)
        raise RuntimeError(f"데이터베이스 오류: {e}")
    finally:
        if conn:
            await release_db_connection(conn)

```

### 5. `backend/app/main.py` (업데이트)

PostgreSQL 연결 초기화/종료 로직과 `SessionManager`, `StatsManager`에 `session_manager`를 주입하는 로직을 추가합니다. `LOCAL_EMBEDDING_MODEL_PATH`는 `backend/app/data/models`로 변경되었으니 `config.py`도 같이 반영해야 합니다.

```python
# backend/app/main.py
import platform
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse # CORS 테스트용
from fastapi.middleware.cors import CORSMiddleware # CORS 미들웨어 추가
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
from app.core.database import connect_to_db, close_db_connection # DB 연결 모듈 임포트
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
        logger.info(f"임베딩 모델 로드 완료: {global_embedding_model.__class__.__name__}, 차원: {global_embedding_model.dimension}")
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
    stats_manager.set_session_manager(session_manager) # 순환 참조 주의 (DI가 더 적합)

    await session_manager.load_all_sessions_from_db()
    await stats_manager.load_all_feedback_from_db() # 기존 피드백 로드하여 통계 계산
    
    logger.info(f"초기 세션 {len(session_manager.get_all_active_sessions())}개, 피드백 {len(stats_manager.get_all_feedback())}개 로드 완료.")

    # 4. 백그라운드 Fine-tuning 및 DB 저장 작업 시작
    logger.info("백그라운드 Fine-tuning 및 세션 DB 저장 스케줄러 시작...")
    asyncio.create_task(start_background_fine_tuning(stats_manager)) # stats_manager 전달
    asyncio.create_task(start_session_db_saver(session_manager)) # session_manager 전달

    yield # 여기서 애플리케이션이 요청을 처리합니다.
    
    # 5. 애플리케이션 종료 시 정리 작업
    logger.info("--- 애플리케이션 종료 ---")
    await close_db_connection() # DB 연결 종료

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
    allow_origins=["http://localhost:3000"], # React 개발 서버 주소
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
app.include_router(agent_router.router, prefix="/agent", tags=["ai_agent"], dependencies=[Depends(get_current_user)]) # 에이전트 API는 로그인 필요

@app.get("/", response_class=HTMLResponse) # CORS 테스트용 HTML 응답 추가
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

```
*   **설명**:
    *   `lifespan` 내에서 `connect_to_db()`를 호출하여 DB 연결 풀을 초기화하고, `close_db_connection()`으로 종료 시 정리합니다.
    *   `session_manager`와 `stats_manager`는 이제 전역 변수로 관리되며, 앱 시작 시 DB에서 데이터를 로드합니다.
    *   `background_tasks.py`의 `start_background_fine_tuning`과 `start_session_db_saver`를 `asyncio.create_task`로 등록하여 백그라운드에서 실행되게 합니다.
    *   **CORS 미들웨어**를 추가했습니다. React 프론트엔드 (`http://localhost:3000`)에서 백엔드 (`http://localhost:8000`)로 요청을 보낼 때 필요해요. 배포 시에는 `allowed_origins`를 실제 프론트엔드 도메인으로 변경해야 합니다.
    *   `get_session_manager`와 `get_stats_manager` 함수를 추가하여 DI (Dependency Injection) 패턴을 사용합니다.

### 6. `backend/app/api/agent.py` (업데이트)

`Depends(get_session_manager)`와 `Depends(get_stats_manager)`를 사용하여 DI로 매니저 인스턴스를 주입받습니다. 오류 처리를 강화하고, 폐쇄망 LLM 호출을 위한 `response_generator.py`의 함수를 호출하도록 변경합니다.

```python
# backend/app/api/agent.py
import uuid # session_id 자동 생성용 (프론트엔드에서 넘겨주는 경우 대비)
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from app.core.config import settings
from app.core.embeddings import get_embedding_model, EmbeddingModel
from app.core.tools import get_all_tools, Tool
from app.services.response_generator import generate_agent_response, MultimodalAgentResponse
from app.core.common_vector_store import AbstractVectorStore
from app.main import get_vector_store, get_session_manager # main.py에서 DI용 함수 임포트
from app.core.security import get_current_user
from app.services.session_manager import SessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
embedding_model: EmbeddingModel = get_embedding_model() # 전역으로 로드된 임베딩 모델 사용
available_tools = {tool.name: tool for tool in get_all_tools()}

class AgentRequest(BaseModel):
    query: str = Field(..., example="휴면 계좌 잔고를 조회하고, 저에게 맞는 대출 상품도 추천해주세요.")
    session_id: Optional[str] = Field(None, description="현재 채팅 세션 ID (없으면 백엔드에서 생성)")
    customer_id: str = Field("user123", example="user123", description="고객 식별 ID (내부 API 연동용)")
    top_k_faq: int = Field(3, description="FAQ 검색 시 가져올 상위 결과 개수")

@router.post("/ask", response_model=MultimodalAgentResponse)
async def ask_agent(
    request: AgentRequest,
    vector_store: AbstractVectorStore = Depends(get_vector_store),
    session_manager: SessionManager = Depends(get_session_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    사용자의 질문을 분석하여 VectorDB RAG, 내부 API 연동, 다른 Agent 연동,
    멀티모달 응답 등을 활용하여 최적의 답변을 생성합니다.
    """
    user_email = current_user.get("email", "unknown_user")
    current_session_id = request.session_id if request.session_id else str(uuid.uuid4())

    try:
        # 0. 세션 시작/로드 및 히스토리 추가
        session = await session_manager.start_or_load_session_async(current_session_id, user_email)
        session.add_message("user", request.query) # 인메모리 세션에 메시지 추가
        logger.info(f"세션 {session.session_id}에서 사용자 '{user_email}'의 요청: {request.query}")

        # 1. 쿼리 임베딩
        query_embedding = await embedding_model.embed_query(request.query)

        # 2. Vector DB (FAQ RAG) 검색
        collection_name = settings.MILVUS_COLLECTION_NAME if settings.SELECTED_VECTOR_DB == "milvus" else settings.CHROMA_COLLECTION_NAME
        try:
            faq_results = await vector_store.search(collection_name, query_embedding, request.top_k_faq)
            faq_context = "\n".join([f"FAQ: {r.text} (거리: {r.distance:.4f})" for r in faq_results])
        except Exception as e:
            logger.error(f"Vector DB 검색 중 오류 발생: {e}", exc_info=True)
            faq_context = "FAQ 검색에 실패했습니다."

        # 3. 툴 사용 결정 및 실행 (간단한 키워드 기반 로직으로 대체)
        tool_outputs = []
        if "휴면 계좌" in request.query or "거래 내역" in request.query:
            internal_api_tool = available_tools.get("InternalAccountAPI")
            if internal_api_tool:
                try:
                    tool_output = await internal_api_tool.run(customer_id=request.customer_id, query=request.query)
                    tool_outputs.append(f"InternalAccountAPI 응답: {tool_output}")
                except Exception as e:
                    logger.warning(f"InternalAccountAPI 호출 실패: {e}", exc_info=True)
                    tool_outputs.append(f"InternalAccountAPI 호출 실패: {e}")
        
        # ... (나머지 툴 호출 로직은 동일하게 유지하되, 각 호출마다 try-except 추가) ...

        # 4. 멀티모달 리소스 검색
        multimodal_resource = None
        if "주택담보대출 가이드" in request.query or "은행 카드 혜택" in request.query:
            multimodal_tool = available_tools.get("MultimodalResourceLookup")
            if multimodal_tool:
                try:
                    resource_output = await multimodal_tool.run(query=request.query)
                    if resource_output.get("status") == "success":
                        multimodal_resource = {
                            "type": resource_output["resource_type"],
                            "url": f"/static/documents/{resource_output['path']}",
                            "summary": resource_output["text_summary"]
                        }
                        tool_outputs.append(f"멀티모달 리소스 검색: {multimodal_resource['summary']}")
                except Exception as e:
                    logger.warning(f"MultimodalResourceLookup 호출 실패: {e}", exc_info=True)
                    tool_outputs.append(f"MultimodalResourceLookup 호출 실패: {e}")

        # 5. 최종 응답 생성 (폐쇄망 LLM 호출 포함)
        final_response_obj = await generate_agent_response(
            user_query=request.query,
            faq_context=faq_context,
            tool_outputs="\n".join(tool_outputs),
            multimodal_resource=multimodal_resource,
            chat_history=session.get_history() # 세션 히스토리 전달 (LLM이 대화 맥락 이해하도록)
        )
        
        # 6. 세션 히스토리에 Agent 응답 추가
        session.add_message("agent", final_response_obj.text_response)

        return final_response_obj
        
    except Exception as e:
        logger.error(f"Agent 요청 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"에이전트 요청 처리 실패: {e}")
```
*   **설명**:
    *   `session_id`가 `Optional`로 변경되어, 프론트엔드에서 처음 요청 시 `session_id`를 넘기지 않으면 백엔드에서 `uuid.uuid4()`로 고유 ID를 생성합니다.
    *   `session_manager.start_or_load_session_async`를 사용하여 DB 연동이 된 세션 매니저를 호출합니다.
    *   각 툴 호출 및 Vector DB 검색에 `try-except` 블록을 추가하여 개별적인 오류에 대응합니다.
    *   `generate_agent_response`에 `chat_history`를 전달하여 LLM이 대화 맥락을 고려한 답변을 생성하도록 합니다.

### 7. `backend/app/services/session_manager.py` (업데이트 - PostgreSQL 연동)

세션 데이터를 PostgreSQL에 저장하고 로드하는 비동기 함수들을 추가합니다.

```python
# backend/app/services/session_manager.py
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.core.database import execute_query # DB 쿼리 실행 함수 임포트
import logging

logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self, session_id: str, user_email: str, start_time: datetime, last_activity: datetime, messages: List[Dict[str, Any]], is_active: bool = True):
        self.session_id = session_id
        self.user_email = user_email
        self.start_time = start_time
        self.last_activity = last_activity
        self.history: List[Dict[str, Any]] = messages # [{"role": "user", "content": "...", "timestamp": "..."}]
        self.is_active = is_active

    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.history.append(message)
        self.last_activity = datetime.now()
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_email": self.user_email,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.history),
            "is_active": self.is_active
        }

    @classmethod
    def from_record(cls, record: Any): # NamedTupleCursor의 결과를 받기 위함
        return cls(
            session_id=record.session_id,
            user_email=record.user_email,
            start_time=record.start_time,
            last_activity=record.last_activity,
            messages=json.loads(record.messages) if isinstance(record.messages, str) else record.messages, # JSONB는 파이썬에서 딕셔너리로 자동변환될 수도 있음
            is_active=record.is_active
        )

class SessionManager:
    _instance = None
    _sessions: Dict[str, ChatSession] = {} # In-memory storage for active sessions

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._sessions = {} # 초기화
        return cls._instance
    
    async def load_all_sessions_from_db(self):
        """DB에서 모든 활성 세션을 로드하여 인메모리에 적재합니다."""
        logger.info("DB에서 활성 세션 로드 중...")
        try:
            records = await execute_query("SELECT * FROM chat_sessions WHERE is_active = TRUE", fetch_all=True)
            if records:
                for record in records:
                    session = ChatSession.from_record(record)
                    self._sessions[session.session_id] = session
                logger.info(f"DB에서 {len(records)}개의 활성 세션을 인메모리로 로드했습니다.")
        except Exception as e:
            logger.error(f"DB에서 세션 로드 실패: {e}", exc_info=True)

    async def save_session_to_db(self, session: ChatSession):
        """단일 세션을 DB에 저장하거나 업데이트합니다."""
        query = """
            INSERT INTO chat_sessions (session_id, user_email, start_time, last_activity, messages, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET user_email = EXCLUDED.user_email,
                last_activity = EXCLUDED.last_activity,
                messages = EXCLUDED.messages,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP;
        """
        try:
            await execute_query(
                query,
                (
                    session.session_id,
                    session.user_email,
                    session.start_time,
                    session.last_activity,
                    json.dumps(session.history), # JSONB로 저장
                    session.is_active,
                )
            )
            logger.debug(f"세션 {session.session_id} DB에 저장/업데이트 완료.")
        except Exception as e:
            logger.error(f"세션 {session.session_id} DB 저장/업데이트 실패: {e}", exc_info=True)
            raise RuntimeError(f"세션 저장 오류: {e}")


    async def start_or_load_session_async(self, session_id: str, user_email: str) -> ChatSession:
        """
        인메모리에 세션이 없으면 DB에서 로드하거나 새로 생성합니다.
        """
        if session_id not in self._sessions:
            # DB에서 해당 세션 찾기
            record = await execute_query("SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,), fetch_one=True)
            if record:
                session = ChatSession.from_record(record)
                self._sessions[session_id] = session
                logger.info(f"DB에서 기존 세션 로드: {session_id} by {user_email}")
            else:
                # 새로운 세션 생성
                session = ChatSession(session_id, user_email, datetime.now(), datetime.now(), [])
                self._sessions[session_id] = session
                # 새로운 세션은 즉시 DB에 저장 (초기 레코드 생성)
                await self.save_session_to_db(session)
                logger.info(f"새로운 세션 시작: {session_id} by {user_email}")
        
        session = self._sessions[session_id]
        session.user_email = user_email # 세션 재활용 시 사용자 업데이트
        session.last_activity = datetime.now()
        return session

    def add_message_to_history(self, session_id: str, role: str, content: str):
        session = self._sessions.get(session_id)
        if session:
            session.add_message(role, content)
            # 여기서는 DB에 즉시 저장하지 않고, 백그라운드 스케줄러가 저장하도록 위임
        else:
            logger.warning(f"세션 {session_id}를 찾을 수 없어 메시지를 추가할 수 없습니다.")

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_all_active_sessions(self) -> List[ChatSession]:
        # 비활성 세션 정리 로직 (예: 특정 시간 이상 활동 없는 세션)은 백그라운드 태스크나 주기적인 스캔으로 처리
        return list(self._sessions.values())

    async def deactivate_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            await self.save_session_to_db(session)
            del self._sessions[session_id]
            logger.info(f"세션 {session_id} 비활성화 및 인메모리에서 제거.")
        else:
            logger.warning(f"비활성화하려는 세션 {session_id}가 인메모리에 없습니다.")
```
*   **설명**:
    *   `ChatSession` 클래스에 `from_record` 클래스 메서드를 추가하여 DB 레코드에서 인스턴스를 생성할 수 있게 합니다.
    *   `load_all_sessions_from_db()`: 앱 시작 시 DB에서 모든 `is_active=TRUE`인 세션을 로드하여 인메모리 `_sessions` 딕셔너리에 적재합니다.
    *   `save_session_to_db()`: 단일 세션 객체를 DB에 저장하거나 업데이트합니다 (`ON CONFLICT` 구문 사용).
    *   `start_or_load_session_async()`: 이제 인메모리 먼저 확인하고 없으면 DB에서 로드, 그래도 없으면 새로 생성 후 DB에도 저장합니다.
    *   `add_message_to_history()`: 메시지 추가 후 바로 DB에 저장하는 대신, 백그라운드 스케줄러가 주기적으로 변경 사항을 감지하여 저장하도록 합니다. 이렇게 하면 요청마다 DB I/O가 발생하는 것을 방지할 수 있어요.

### 8. `backend/app/services/stats_manager.py` (업데이트 - PostgreSQL 연동)

피드백 데이터를 PostgreSQL에 저장하고 로드하는 비동기 함수들을 추가합니다.

```python
# backend/app/services/stats_manager.py
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.database import execute_query
import logging

logger = logging.getLogger(__name__)

class StatsManager:
    _instance = None
    _feedback_data: List[Dict[str, Any]] = [] # In-memory cache for feedback
    _performance_metrics: Dict[str, Any] = {}
    _session_ref: Optional[Any] = None # SessionManager 참조

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatsManager, cls).__new__(cls)
            cls._feedback_data = []
            cls._performance_metrics = {
                "total_queries": 0,
                "avg_response_time_ms": 0.0, # 아직 구현 안됨
                "like_count": 0,
                "dislike_count": 0,
                "overall_satisfaction": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        return cls._instance

    def set_session_manager(self, session_manager_instance):
        """SessionManager 인스턴스를 주입받음 (DI)."""
        self._session_ref = session_manager_instance

    async def load_all_feedback_from_db(self):
        """DB에서 모든 피드백을 로드하여 인메모리에 적재합니다."""
        logger.info("DB에서 피드백 데이터 로드 중...")
        try:
            records = await execute_query("SELECT * FROM feedback", fetch_all=True)
            if records:
                self._feedback_data = [dict(record._asdict()) for record in records] # NamedTuple을 딕셔너리로 변환
                logger.info(f"DB에서 {len(records)}개의 피드백 데이터를 인메모리로 로드했습니다.")
                self._calculate_performance_metrics()
        except Exception as e:
            logger.error(f"DB에서 피드백 로드 실패: {e}", exc_info=True)

    async def add_feedback(self, user_email: str, session_id: str, message_id: str, feedback_type: str, user_query: str, agent_response: str):
        """피드백을 DB에 저장하고 인메모리에 추가합니다."""
        query = """
            INSERT INTO feedback (user_email, session_id, message_id, feedback_type, user_query, agent_response, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        try:
            await execute_query(
                query,
                (user_email, session_id, message_id, feedback_type, user_query, agent_response, datetime.now())
            )
            feedback_entry = {
                "user_email": user_email,
                "session_id": session_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "user_query": user_query,
                "agent_response": agent_response,
                "timestamp": datetime.now().isoformat()
            }
            self._feedback_data.append(feedback_entry)
            self._calculate_performance_metrics()
            logger.info(f"피드백 추가: user={user_email}, session={session_id}, type={feedback_type}, DB 저장 완료.")
        except Exception as e:
            logger.error(f"피드백 DB 저장 실패: {e}", exc_info=True)
            raise RuntimeError(f"피드백 저장 오류: {e}")


    def _calculate_performance_metrics(self):
        like_count = sum(1 for f in self._feedback_data if f["feedback_type"] == "like")
        dislike_count = sum(1 for f in self._feedback_data if f["feedback_type"] == "dislike")
        total_feedback = like_count + dislike_count
        
        self._performance_metrics["like_count"] = like_count
        self._performance_metrics["dislike_count"] = dislike_count
        if total_feedback > 0:
            self._performance_metrics["overall_satisfaction"] = (like_count - dislike_count) / total_feedback
        else:
            self._performance_metrics["overall_satisfaction"] = 0.0
        
        if self._session_ref:
            # 모든 세션의 총 메시지 수 합산 (실시간 데이터)
            self._performance_metrics["total_queries"] = sum(
                session.message_count for session in self._session_ref.get_all_active_sessions()
            )
        else:
             self._performance_metrics["total_queries"] = 0 
        
        self._performance_metrics["last_updated"] = datetime.now().isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        self._calculate_performance_metrics()
        return self._performance_metrics

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        return self._feedback_data

```
*   **설명**:
    *   `load_all_feedback_from_db()`: 앱 시작 시 DB에서 모든 피드백을 로드하여 인메모리에 적재하고 통계를 계산합니다.
    *   `add_feedback()`: 피드백을 받으면 DB에 먼저 저장하고, 성공 시 인메모리 리스트에 추가하여 통계를 업데이트합니다.
    *   `_calculate_performance_metrics()`: `SessionManager`에서 실시간 활성 세션 수를 가져와 총 쿼리 수에 반영합니다.

### 9. `backend/app/api/feedback.py` (업데이트)

`StatsManager`가 DB 연동 기능을 갖도록 변경되었으므로 `add_feedback` 호출만 수정합니다.

```python
# backend/app/api/feedback.py
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, Any
from app.core.security import get_current_user
from app.main import get_stats_manager, get_session_manager # DI 함수 임포트
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., example="unique_session_id", description="피드백을 제공하는 세션 ID")
    message_id: str = Field(..., example="agent_response_001", description="피드백 대상 응답의 ID (클라이언트에서 생성하여 넘겨야 함)")
    feedback_type: str = Field(..., example="like", description="피드백 타입: 'like' 또는 'dislike'")

@router.post("/submit")
async def submit_feedback(
    request: FeedbackRequest,
    stats_manager: StatsManager = Depends(get_stats_manager),
    session_manager: SessionManager = Depends(get_session_manager),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    사용자의 응답 피드백 (좋아요/싫어요)을 수집합니다.
    """
    user_email = current_user.get("email", "unknown_user")
    
    if request.feedback_type not in ["like", "dislike"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효하지 않은 피드백 타입입니다. 'like' 또는 'dislike'를 사용해주세요.")

    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"세션 ID '{request.session_id}'를 찾을 수 없습니다.")

    # 특정 message_id에 대한 쿼리와 응답 내용을 찾아서 저장 (프론트엔드에서 넘어오는 message_id 활용)
    # 실제 구현에서는 message_id를 정확히 매핑하는 로직이 필요. 여기서는 간단히 가장 최근 쿼리-응답 쌍을 찾음
    user_query = "찾을 수 없음"
    agent_response_text = "찾을 수 없음"

    for i in range(len(session.history) - 1, 0, -1):
        if session.history[i].get("role") == "agent":
            agent_response_text = session.history[i].get("content", "")
            if i > 0 and session.history[i-1].get("role") == "user":
                user_query = session.history[i-1].get("content", "")
            break


    try:
        await stats_manager.add_feedback(
            user_email=user_email,
            session_id=request.session_id,
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            user_query=user_query,
            agent_response=agent_response_text
        )
        return {"message": "피드백이 성공적으로 기록되었습니다."}
    except RuntimeError as e:
        logger.error(f"피드백 저장 실패: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="피드백 저장 중 오류가 발생했습니다.")
```

### 10. `backend/app/api/dashboard.py` (업데이트)

`SessionManager`, `StatsManager`를 DI로 주입받도록 수정합니다.

```python
# backend/app/api/dashboard.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from app.core.security import get_current_user
from app.main import get_stats_manager, get_session_manager # DI 함수 임포트
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ⚠️ 주의: 실제 서비스에서는 관리자만 접근 가능하도록 추가 권한 확인 로직 필요
# 예를 들어, current_user의 role이 'admin'인지 확인
def is_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    # 이메일 등으로 관리자 여부 판단 (임시)
    if not current_user.get("email") in ["admin@example.com", "your_admin_email@domain.com"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="관리자만 접근 가능합니다.")
    return current_user

@router.get("/metrics")
async def get_chatbot_metrics(
    stats_manager: StatsManager = Depends(get_stats_manager),
    admin_user: Dict[str, Any] = Depends(is_admin_user)
):
    """
    챗봇의 전반적인 성능 지표를 반환합니다. (관리자 전용)
    """
    logger.info(f"대시보드 성능 지표 요청: user={admin_user['email']}")
    metrics = stats_manager.get_performance_metrics()
    return metrics

@router.get("/active_sessions")
async def get_active_sessions(
    session_manager: SessionManager = Depends(get_session_manager),
    admin_user: Dict[str, Any] = Depends(is_admin_user)
) -> List[Dict[str, Any]]:
    """
    현재 활성화된 사용자 세션 목록을 반환합니다. (관리자 전용)
    """
    logger.info(f"활성 세션 목록 요청: user={admin_user['email']}")
    sessions = [s.to_dict() for s in session_manager.get_all_active_sessions()]
    return sessions

```

### 11. `backend/app/services/background_tasks.py` (업데이트 - 백그라운드 DB 저장 및 Fine-tuning)

세션 데이터를 주기적으로 DB에 저장하는 스케줄러와 Fine-tuning 스케줄러를 추가합니다.

```python
# backend/app/services/background_tasks.py
import asyncio
from datetime import datetime, timedelta
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager, ChatSession
from app.core.embeddings import EmbeddingModel
from app.core.database import execute_query # DB 연동
import logging

logger = logging.getLogger(__name__)

# 임의의 백그라운드 모델 (예: 파인튜닝 대상 모델)
class DummyFineTuneModel:
    def __init__(self):
        self.accuracy = 0.70
        self.last_trained = None
        logger.info("더미 파인튜닝 모델 초기화")

    async def fine_tune(self, feedback_data: List[Dict[str, Any]]):
        logger.info(f"백그라운드에서 모델 파인튜닝 시작... (피드백 데이터 {len(feedback_data)}개 활용)")
        # --- 실제 모델 파인튜닝 로직 (수분~수시간 소요 가능) ---
        # 1. feedback_data를 이용하여 학습 데이터셋 생성
        #    예: feedback_data 중 'dislike' 피드백이 있는 경우, 해당 쿼리와 응답 쌍을
        #    잘못된 답변으로 분류하고, 'like' 피드백은 좋은 답변으로 분류하여
        #    새로운 학습 데이터셋을 만듦.
        # 2. EmbeddingModel (또는 LLM)의 특정 레이어를 fine-tuning
        #    (폐쇄망 LLM의 경우: 로컬 LLM API에 파인튜닝 요청을 보내거나,
        #     로컬 서버에서 직접 Fine-tuning 모델을 로드하여 재학습)
        # 3. 새로운 모델 가중치 저장 및 로드 (또는 Fine-tuned 모델 API 업데이트)
        
        await asyncio.sleep(10) # 파인튜닝 작업 시뮬레이션 (10초 소요)
        
        # 모델 개선 시뮬레이션
        if feedback_data: # 피드백이 있는 경우만
            like_count = sum(1 for f in feedback_data if f["feedback_type"] == "like")
            dislike_count = sum(1 for f in feedback_data if f["feedback_type"] == "dislike")
            total_feedback = like_count + dislike_count
            if total_feedback > 0:
                # 긍정 피드백이 많을수록 정확도 상승
                self.accuracy += (0.01 * (like_count / total_feedback))
                # 부정 피드백이 많으면 정확도 하락 또는 재검토 필요 시뮬레이션
                if dislike_count > like_count:
                    self.accuracy -= 0.005 # 부정 피드백이 많으면 소폭 하락
            if self.accuracy > 0.95: self.accuracy = 0.95 # 상한선
            if self.accuracy < 0.60: self.accuracy = 0.60 # 하한선

        self.last_trained = datetime.now()
        logger.info(f"모델 파인튜닝 완료! 새로운 정확도: {self.accuracy:.2f}")

dummy_fine_tune_model = DummyFineTuneModel()

async def start_background_fine_tuning(stats_manager: StatsManager):
    """주기적으로 피드백을 모아 모델을 파인튜닝하는 스케줄러."""
    while True:
        await asyncio.sleep(60 * 60 * 12) # 12시간마다 파인튜닝 시도 (배포 시 설정 조절)
        logger.info("백그라운드 파인튜닝 작업 스케줄러 실행.")
        feedback_data = stats_manager.get_all_feedback()
        if feedback_data:
            await dummy_fine_tune_model.fine_tune(feedback_data)
        else:
            logger.info("파인튜닝할 피드백 데이터가 없습니다.")

async def start_session_db_saver(session_manager: SessionManager):
    """주기적으로 변경된 세션 데이터를 DB에 저장하는 스케줄러."""
    while True:
        await asyncio.sleep(60) # 60초마다 세션 데이터를 DB에 저장
        logger.debug("백그라운드 세션 DB 저장 스케줄러 실행.")
        active_sessions = session_manager.get_all_active_sessions()
        for session in active_sessions:
            try:
                # 일정 시간 이상 활동이 없으면 비활성화 처리
                if (datetime.now() - session.last_activity) > timedelta(minutes=60): # 60분
                    session.is_active = False
                    logger.info(f"세션 {session.session_id}가 비활성 상태로 전환됩니다.")
                await session_manager.save_session_to_db(session)
                if not session.is_active:
                    session_manager._sessions.pop(session.session_id, None) # 인메모리에서 제거
            except Exception as e:
                logger.error(f"백그라운드에서 세션 {session.session_id} 저장 실패: {e}", exc_info=True)

```
*   **설명**:
    *   `start_background_fine_tuning`: `stats_manager`를 인자로 받아 피드백 데이터를 활용할 수 있도록 했습니다. Fine-tuning 로직은 실제 폐쇄망 LLM API를 호출하거나 로컬 모델을 재학습시키는 부분이 들어가야 합니다.
    *   `start_session_db_saver`: `session_manager`를 인자로 받아 현재 활성 세션들을 주기적으로 DB에 저장합니다. 일정 시간 (예: 60분) 동안 활동이 없으면 세션을 `is_active=False`로 전환하고 인메모리에서 제거하는 로직도 추가했습니다.

### 12. `backend/app/services/response_generator.py` (업데이트 - 폐쇄망 LLM 통합 및 오류 처리)

이제 `call_local_llm` 함수를 통해 실제로 폐쇄망 LLM을 호출하고, 오류가 발생하면 적절히 대응합니다.

```python
# backend/app/services/response_generator.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from app.core.config import settings
import logging
import httpx # 로컬 LLM API 호출을 위해
import asyncio_throttle # LLM 호출 제한을 위해 (옵션)

logger = logging.getLogger(__name__)

# LLM API 호출을 위한 Rate Limiter (선택 사항)
# 초당 5회 호출로 제한 (로컬 LLM 서비스의 부하를 고려하여 조절)
llm_call_throttle = asyncio_throttle.Throttle(rate_limit=5)

class MultimodalContent(BaseModel):
    type: str = Field(..., example="pdf")
    url: str = Field(..., example="/static/documents/bank_loan_guide.pdf")
    summary: str = Field(..., example="주택담보대출 가이드 PDF를 찾았습니다. 자세한 내용은 PDF를 참고하세요.")

class MultimodalAgentResponse(BaseModel):
    text_response: str = Field(..., example="안녕하세요, 고객님! 무엇을 도와드릴까요?")
    multimodal_content: Optional[MultimodalContent] = Field(None, description="PDF 문서, 이미지 등 추가 멀티모달 응답")
    session_id: str = Field(..., example="unique_session_id", description="현재 채팅 세션 ID") # 프론트엔드 연동용
    message_id: str = Field(..., example="agent_response_001", description="생성된 응답의 고유 ID (피드백용)") # 프론트엔드 연동용
    debug_info: Optional[Dict[str, Any]] = Field(None, description="디버깅을 위한 추가 정보 (배포 시 제거 권장)")

async def call_local_llm(prompt: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    폐쇄망 로컬 LLM을 호출하여 응답을 생성합니다.
    settings.LOCAL_LLM_API_URL로 HTTP 요청을 보냅니다.
    """
    try:
        # LLM 모델에게 전달할 메시지 형식 (OpenAI API 형식과 유사)
        messages = [{"role": m["role"], "content": m["content"]} for m in chat_history[-5:]] # 최근 5개 메시지만 전달
        messages.append({"role": "user", "content": prompt}) # 현재 프롬프트 추가

        async with llm_call_throttle: # Rate Limiting 적용 (선택 사항)
            async with httpx.AsyncClient(timeout=30.0) as client: # LLM 응답 시간 고려
                response = await client.post(
                    settings.LOCAL_LLM_API_URL,
                    json={
                        "model": settings.LOCAL_LLM_MODEL_NAME,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 500,
                        # 기타 LLM 서비스에 필요한 파라미터들
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
                response_data = response.json()
                
                # 로컬 LLM 서비스의 응답 구조에 따라 파싱
                # 예: Ollama, vLLM 등의 Chat Completion API 응답
                if response_data.get("choices") and response_data["choices"][0].get("message"):
                    return response_data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"로컬 LLM 응답 형식이 예상과 다릅니다: {response_data}")
                    return "죄송합니다. LLM이 올바른 형식으로 응답하지 않았습니다."

    except httpx.RequestError as e:
        logger.error(f"로컬 LLM API 연결 실패: {e}", exc_info=True)
        return "죄송합니다. LLM 서비스에 연결할 수 없습니다. 관리자에게 문의해주세요."
    except httpx.HTTPStatusError as e:
        logger.error(f"로컬 LLM API HTTP 오류 발생: {e.response.status_code} - {e.response.text}", exc_info=True)
        return f"죄송합니다. LLM 서비스에서 오류가 발생했습니다. ({e.response.status_code})"
    except Exception as e:
        logger.error(f"로컬 LLM 응답 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return "죄송합니다. 답변 생성 중 알 수 없는 오류가 발생했습니다."


async def generate_agent_response(
    user_query: str,
    faq_context: str,
    tool_outputs: str,
    multimodal_resource: Optional[Dict[str, Any]] = None,
    chat_history: List[Dict[str, Any]] = None # 대화 히스토리 추가
) -> MultimodalAgentResponse:
    """
    에이전트의 최종 응답을 생성합니다. 폐쇄망 LLM이 FAQ 검색 결과와 툴 실행 결과를 바탕으로
    사용자에게 자연스러운 답변을 만들고, 필요시 멀티모달 리소스를 포함합니다.
    """
    if chat_history is None:
        chat_history = []

    # LLM에게 전달할 프롬프트 구성
    llm_prompt = f"""
    당신은 친절한 은행 챗봇 Agent 입니다. 고객의 질문에 대해 아래 제공된 정보들을 활용하여 답변해주세요.
    제공된 정보만으로 답변하기 어렵거나 추가적인 조치가 필요하면 그렇게 안내해주세요.

    [고객의 질문]
    {user_query}

    [검색된 FAQ 정보]
    {faq_context if faq_context else "관련 FAQ를 찾지 못했습니다."}

    [툴 실행 결과]
    {tool_outputs if tool_outputs else "실행된 툴이 없습니다."}

    {multimodal_resource['summary'] if multimodal_resource else ""}

    친절하고 명확하게 답변해주세요.
    """

    final_text_response = "답변 생성 중 오류가 발생했습니다." # 기본 오류 메시지
    try:
        final_text_response = await call_local_llm(llm_prompt, chat_history)
    except Exception as e:
        logger.error(f"로컬 LLM 호출 중 오류 발생: {e}", exc_info=True)
        final_text_response = "죄송합니다. 답변 생성 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    # 멀티모달 콘텐츠 객체 생성
    multimodal_content_obj = None
    if multimodal_resource:
        multimodal_content_obj = MultimodalContent(
            type=multimodal_resource['type'],
            url=multimodal_resource['url'],
            summary=multimodal_resource['summary']
        )
    
    # 프론트엔드에서 사용할 message_id 생성 (각 응답마다 고유)
    message_id = f"agent_response_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    return MultimodalAgentResponse(
        text_response=final_text_response,
        multimodal_content=multimodal_content_obj,
        session_id=chat_history[-1].get("session_id", "unknown_session") if chat_history else "unknown_session", # 세션 ID도 응답에 포함
        message_id=message_id,
        debug_info={
            "user_query": user_query,
            "faq_context": faq_context,
            "tool_outputs": tool_outputs,
            "local_llm_prompt": llm_prompt # 디버깅용
        }
    )

```
*   **설명**:
    *   `call_local_llm`: `httpx`를 사용하여 `settings.LOCAL_LLM_API_URL`에 HTTP POST 요청을 보냅니다. `chat_history`를 함께 보내 LLM이 대화 맥락을 이해하도록 합니다. **이 함수가 실제 폐쇄망 로컬 LLM 서비스를 호출하는 부분이 될 거예요.** 지금은 LLM 서비스를 직접 포함할 수 없으므로, LLM 서비스의 응답 형식에 맞게 파싱하는 로직을 가정했습니다.
    *   `asyncio_throttle.Throttle`를 사용하여 LLM API 호출에 Rate Limiting을 적용할 수 있습니다. (선택 사항이지만 LLM 서비스 부하 관리에 유용)
    *   `generate_agent_response`: LLM 호출 전 `llm_prompt`를 구성할 때 `chat_history`와 검색된 정보들을 모두 활용합니다. 오류 발생 시 대체 메시지를 반환합니다.
    *   `MultimodalAgentResponse`에 `session_id`와 `message_id` 필드를 추가하여 프론트엔드에서 피드백과 세션 관리에 활용할 수 있도록 했습니다.

### 13. `scripts/run_linux.sh` 및 `scripts/run_windows.bat` (업데이트)

PostgreSQL 컨테이너 실행 명령과 `db_schema.sql`을 이용한 초기화, 그리고 LLM 모델 경로 변경을 반영합니다.

**`scripts/run_linux.sh` (업데이트)**

```bash
#!/bin/bash

echo "Starting 폐쇄망 멀티모달 & 지능형 AI Agent Backend on Linux..."

# 0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
if [ ! -d "../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab" ]; then
    echo "Warning: Local embedding model directory '../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  mkdir -p ../backend/app/data/models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
fi

# 0. ChromaDB 데이터 경로 생성 (리눅스에서 Chroma를 사용하고 싶을 경우)
mkdir -p ../backend/chroma_data

# 1. PostgreSQL DB 및 Milvus Vector DB 실행 (Docker Compose)
echo "Deploying PostgreSQL and Milvus with Docker Compose..."
cd .. # 프로젝트 루트로 이동
# docker-compose.yml 파일에 PostgreSQL 서비스 추가 필요!
# 여기서는 편의상 Milvus만 돌리고, PostgreSQL은 수동으로 가정하거나 별도 docker-compose 파일을 써야 함.
# 여기서는 DB 연결 정보를 .env에서 가져오므로, Postgres 컨테이너가 5432 포트로 떠 있어야 함.
# 예시: 다음 docker-compose.yml 내용 참고
# version: '3.8'
# services:
#   milvus:
#     image: milvusdb/milvus:v2.3.0
#     container_name: milvus-standalone
#     environment:
#       ETCD_ENDPOINTS: etcd:2379
#       MINIO_ADDRESS: minio:9000
#       MILVUS_PORT: 19530
#     ports:
#       - "19530:19530"
#       - "9091:9091"
#     depends_on:
#       - etcd
#       - minio
#     command: milvus-standalone
#   etcd:
#     image: quay.io/coreos/etcd:v3.5.0
#     environment:
#       ETCD_AUTO_COMPACTION_MODE: revision
#       ETCD_AUTO_COMPACTION_RETENTION: 1000
#       ETCD_QUOTA_BACKEND_BYTES: 4294967296
#       ETCD_SNAPSHOT_INTERVAL: 100000
#       ETCD_ELECTION_TIMEOUT: 10000
#       ETCD_LISTEN_CLIENT_URLS: http://0.0.0.0:2379
#       ETCD_ADVERTISE_CLIENT_URLS: http://etcd:2379
#   minio:
#     image: minio/minio:RELEASE.2023-03-20T20-16-04Z
#     environment:
#       MINIO_ACCESS_KEY: minioadmin
#       MINIO_SECRET_KEY: minioadmin
#     ports:
#       - "9000:9000"
#     command: minio --address 0.0.0.0:9000 --console-address 0.0.0.0:9001 S3
#   pgdb: # PostgreSQL 서비스 추가
#     image: postgres:13
#     restart: always
#     environment:
#       POSTGRES_USER: ${POSTGRES_USER}
#       POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
#       POSTGRES_DB: ${POSTGRES_DB}
#     ports:
#       - "5432:5432"
#     volumes:
#       - postgres_data:/var/lib/postgresql/data # 데이터 영구 저장
# volumes:
#   postgres_data: {}
# # 이 내용을 milvus-standalone-docker-compose.yml 에 추가하거나 별도 docker-compose-db.yml 생성.

# (PostgreSQL 및 Milvus를 포함하는) docker compose 파일이 준비되었다고 가정
if [ ! -f "./docker-compose.yml" ]; then # 이전에 milvus-standalone-docker-compose.yml이었던 것을 가정.
    echo "Warning: docker-compose.yml not found. Please create one for Milvus and PostgreSQL."
    # wget 등 다운로드 로직은 제거하고, 사용자가 직접 구성하도록 유도.
fi
docker compose -f ./docker-compose.yml up -d # 통합된 docker compose 파일 실행
sleep 30 # DB 및 Milvus가 완전히 시작될 때까지 충분히 기다립니다.

echo "Verifying service status..."
docker ps -a | grep milvus
docker ps -a | grep pgdb

# PostgreSQL DB 스키마 적용 (컨테이너 내부에서 실행하거나 외부에서 psql로 연결)
echo "Applying PostgreSQL database schema..."
docker exec -i $(docker ps -aqf "name=pgdb") psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} < db_schema.sql
echo "PostgreSQL schema applied."


# 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend # 다시 backend 디렉토리로 이동
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    cp .env.example .env
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo "To stop services, go to the project root directory and run: docker compose -f ./docker-compose.yml down"

```

**`scripts/run_windows.bat` (업데이트)**

```batch
@echo off
echo "Starting 폐쇄망 멀티모달 & 지능형 AI Agent Backend on Windows..."

:: 0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
IF NOT EXIST "..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab" (
    echo "Warning: Local embedding model directory '..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  md ..\backend\app\data\models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
)

:: 0. ChromaDB 데이터 경로 생성 (Windows 기본값)
mkdir ..\backend\chroma_data

:: 1. PostgreSQL DB 실행 (Docker Desktop)
echo "Deploying PostgreSQL with Docker Desktop..."
echo "Please ensure Docker Desktop is running."
cd %~dp0\..

:: (PostgreSQL docker-compose.yml이 준비되었다고 가정)
:: 예시: 다음 docker-compose-db.yml 내용 참고
:: version: '3.8'
:: services:
::   pgdb: # PostgreSQL 서비스
::     image: postgres:13
::     restart: always
::     environment:
::       POSTGRES_USER: ${POSTGRES_USER}
::       POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
::       POSTGRES_DB: ${POSTGRES_DB}
::     ports:
::       - "5432:5432"
::     volumes:
::       - postgres_data:/var/lib/postgresql/data # 데이터 영구 저장
:: volumes:
::   postgres_data: {}
:: 이 내용을 docker-compose-db.yml 에 저장

IF NOT EXIST docker-compose-db.yml (
    echo "Warning: docker-compose-db.yml not found. Please create one for PostgreSQL."
)
docker compose -f docker-compose-db.yml up -d
timeout /t 30 /nobreak > NUL :: DB가 완전히 시작될 때까지 충분히 기다립니다.

echo "Verifying PostgreSQL service status..."
docker ps -a | findstr pgdb

:: PostgreSQL DB 스키마 적용 (Windows CMD에서 psql 설치 후 실행하거나 Docker exec)
echo "Applying PostgreSQL database schema..."
docker exec -i $(docker ps -aqf "name=pgdb") psql -U %POSTGRES_USER% -d %POSTGRES_DB% < db_schema.sql
echo "PostgreSQL schema applied."

:: Milvus는 Windows 환경에서는 ChromaDB가 기본이므로 이 부분은 생략
:: 만약 WINDOWS에서 Milvus를 사용하고 싶다면, .env에서 VECTOR_DB=milvus 로 명시하고 별도로 Milvus docker-compose 실행

:: 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

:: 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
IF NOT EXIST .env (
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    copy .env.example .env
)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo "To stop PostgreSQL (and Milvus if running), navigate to the project root and run: docker compose -f docker-compose-db.yml down"
pause
```
*   **설명**:
    *   **PostgreSQL Docker Compose**: 이제 `docker-compose.yml` (또는 `docker-compose-db.yml`과 `docker-compose-milvus.yml` 등 분리된 파일)에 PostgreSQL 서비스도 포함되어야 합니다. `.env` 파일의 `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` 환경 변수를 사용하도록 설정하세요.
    *   **스키마 적용**: Docker Compose로 DB 컨테이너가 뜨면 `db_schema.sql` 파일을 실행하여 테이블을 생성하도록 스크립트에 추가했습니다.
    *   **LLM 모델 경로**: 로컬 LLM 모델 경로가 `backend/app/data/models`로 변경되었으니 스크립트의 모델 다운로드 안내도 반영했습니다.

---

## 💻 프론트엔드 (React) 주요 컴포넌트 및 로직 (개념 및 핵심 코드)

이제 사용자 경험을 위한 React 프론트엔드 프로젝트의 핵심 컴포넌트와 로직을 설계해볼게요. `create-react-app` 또는 Vite 등으로 프로젝트를 시작한 후 아래 내용들을 적용하시면 됩니다.

### `frontend/src/api/axiosInstance.js` - Axios 인스턴스

백엔드 API 호출을 위한 `axios` 인스턴스입니다. JWT 토큰을 자동으로 헤더에 포함합니다.

```javascript
// frontend/src/api/axiosInstance.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // 백엔드 API 주소

const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터: 로컬 스토리지에서 JWT 토큰을 가져와 Authorization 헤더에 추가
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터: 401 Unauthorized 에러 발생 시 로그아웃 처리
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      console.error("401 Unauthorized: JWT 토큰 만료 또는 유효하지 않음.");
      // 여기에 로그아웃 처리 로직 추가 (예: localStorage.removeItem('jwt_token'); window.location.href = '/login';)
      localStorage.removeItem('jwt_token');
      window.location.href = '/auth'; // 로그인 페이지로 리다이렉트
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;
```

### `frontend/src/context/AuthContext.js` - 인증 상태 관리

JWT 토큰 및 사용자 정보를 전역적으로 관리하기 위한 React Context입니다.

```javascript
// frontend/src/context/AuthContext.js
import React, { createContext, useState, useEffect, useContext } from 'react';
import axiosInstance from '../api/axiosInstance'; // JWT 토큰을 자동으로 헤더에 포함

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [jwtToken, setJwtToken] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('jwt_token');
    const storedUser = localStorage.getItem('user_info');
    if (token && storedUser) {
      try {
        const userInfo = JSON.parse(storedUser);
        setJwtToken(token);
        setUser(userInfo);
        setIsAuthenticated(true);
        console.log("Existing JWT token and user info loaded.");
      } catch (e) {
        console.error("Failed to parse user info from localStorage", e);
        logout(); // 유효하지 않은 정보는 제거
      }
    } else {
      setIsAuthenticated(false);
      setUser(null);
      setJwtToken(null);
    }
  }, []);

  const login = (token, userData) => {
    localStorage.setItem('jwt_token', token);
    localStorage.setItem('user_info', JSON.stringify(userData));
    setJwtToken(token);
    setUser(userData);
    setIsAuthenticated(true);
  };

  const logout = () => {
    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_info');
    setJwtToken(null);
    setUser(null);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, jwtToken, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
```

### `frontend/src/hooks/useChatSession.js` - 채팅 세션 관리 훅

고유한 세션 ID를 생성하고, 메시지 기록을 관리하는 커스텀 훅입니다.

```javascript
// frontend/src/hooks/useChatSession.js
import { useState, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid'; // npm install uuid

const useChatSession = () => {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]); // {id: "msg_uuid", role: "user/agent", content: "text", multimodal: {}, feedback: null, timestamp: "..."}

  useEffect(() => {
    // 세션 ID가 없으면 새로 생성 (또는 기존 세션 ID 복원 로직 추가 가능)
    let currentSessionId = localStorage.getItem('chat_session_id');
    if (!currentSessionId) {
      currentSessionId = uuidv4();
      localStorage.setItem('chat_session_id', currentSessionId);
    }
    setSessionId(currentSessionId);

    // TODO: 백엔드에서 해당 sessionId의 기존 채팅 기록을 로드하는 로직 추가
    // (SessionManager가 DB에서 세션 기록을 가져오므로, 로그인 후 이 기록을 가져와야 함)
    // 예: axiosInstance.get(`/session/history?session_id=${currentSessionId}`)
  }, []);

  const addMessage = useCallback((role, content, multimodalContent = null) => {
    const newMessage = {
      id: uuidv4(), // 각 메시지마다 고유 ID 부여 (피드백용)
      role,
      content,
      multimodal: multimodalContent,
      feedback: null, // 초기 피드백 상태
      timestamp: new Date().toISOString(),
    };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    return newMessage.id; // 생성된 메시지 ID 반환
  }, []);

  const updateMessageFeedback = useCallback((messageId, feedbackType) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === messageId ? { ...msg, feedback: feedbackType } : msg
      )
    );
  }, []);

  return { sessionId, messages, addMessage, updateMessageFeedback };
};

export default useChatSession;
```

### `frontend/src/components/AuthGoogle.js` - Google 로그인 컴포넌트

Google 로그인 버튼을 렌더링하고, 로그인 흐름을 시작합니다.

```javascript
// frontend/src/components/AuthGoogle.js
import React from 'react';
import axiosInstance from '../api/axiosInstance'; // 백엔드 호출용

const AuthGoogle = () => {
  const handleGoogleLogin = async () => {
    try {
      // 백엔드의 Google 로그인 시작 API를 호출
      // 이 API는 Google OAuth 페이지로 리다이렉트 응답을 보냄
      window.location.href = `${axiosInstance.defaults.baseURL}/auth/google/login`;
    } catch (error) {
      console.error("Google login initiation failed", error);
      alert("Google 로그인 시작에 실패했습니다. 다시 시도해주세요.");
    }
  };

  return (
    <button
      onClick={handleGoogleLogin}
      style={{
        padding: '10px 20px',
        fontSize: '16px',
        backgroundColor: '#4285F4',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
      }}
    >
      <img
        src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png" // Google 로고 (임시)
        alt="Google logo"
        style={{ width: '20px', height: '20px' }}
      />
      Google 계정으로 로그인
    </button>
  );
};

export default AuthGoogle;
```

### `frontend/src/components/ChatMessage.js` - 개별 채팅 메시지 렌더링

사용자 메시지 또는 Agent의 응답 (텍스트, PDF, 이미지 포함)을 렌더링합니다. 피드백 버튼도 포함합니다.

```javascript
// frontend/src/components/ChatMessage.js
import React from 'react';
import FeedbackButtons from './FeedbackButtons'; // 피드백 버튼 컴포넌트

const ChatMessage = ({ message, onFeedback }) => {
  const isAgent = message.role === 'agent';

  // 멀티모달 콘텐츠 렌더링 함수
  const renderMultimodalContent = (multimodal) => {
    if (!multimodal) return null;

    if (multimodal.type === 'pdf') {
      return (
        <div style={{ marginTop: '10px' }}>
          <p>📄 {multimodal.summary}</p>
          <a
            href={`${multimodal.url}`} // 백엔드 정적 파일 URL
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: '#4285F4', textDecoration: 'underline' }}
          >
            PDF 문서 열기
          </a>
          <p style={{ fontSize: '0.8em', color: '#888' }}>
            (PDF 뷰어는 클라이언트 측 구현 필요)
          </p>
        </div>
      );
    } else if (multimodal.type === 'image') {
      return (
        <div style={{ marginTop: '10px' }}>
          <p>🖼️ {multimodal.summary}</p>
          <img
            src={`${multimodal.url}`} // 백엔드 정적 파일 URL
            alt="Multimodal content"
            style={{ maxWidth: '100%', maxHeight: '300px', borderRadius: '5px' }}
          />
        </div>
      );
    }
    return null;
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: isAgent ? 'flex-start' : 'flex-end',
        marginBottom: '10px',
      }}
    >
      <div
        style={{
          maxWidth: '70%',
          padding: '10px 15px',
          borderRadius: '15px',
          backgroundColor: isAgent ? '#e0e0e0' : '#4285F4',
          color: isAgent ? 'black' : 'white',
          position: 'relative',
        }}
      >
        <p style={{ margin: '0', whiteSpace: 'pre-wrap' }}>{message.content}</p>
        {renderMultimodalContent(message.multimodal)}

        {isAgent && (
          <div style={{ marginTop: '10px' }}>
            <FeedbackButtons
              messageId={message.id}
              sessionId={message.sessionId} // Agent 응답 시 받은 세션 ID
              currentFeedback={message.feedback}
              onFeedback={onFeedback}
            />
            <p style={{ fontSize: '0.7em', color: '#666' }}>
              응답 ID: {message.id}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
```

### `frontend/src/components/FeedbackButtons.js` - 좋아요/싫어요 버튼

피드백 API를 호출하는 버튼 컴포넌트입니다.

```javascript
// frontend/src/components/FeedbackButtons.js
import React, { useState } from 'react';
import axiosInstance from '../api/axiosInstance';

const FeedbackButtons = ({ messageId, sessionId, currentFeedback, onFeedback }) => {
  const [feedbackStatus, setFeedbackStatus] = useState(currentFeedback);
  const [loading, setLoading] = useState(false);

  const handleSubmitFeedback = async (feedbackType) => {
    if (loading || feedbackStatus === feedbackType) return; // 이미 피드백했거나 로딩 중이면 방지

    setLoading(true);
    try {
      await axiosInstance.post('/feedback/submit', {
        session_id: sessionId,
        message_id: messageId,
        feedback_type: feedbackType,
      });
      setFeedbackStatus(feedbackType);
      onFeedback(messageId, feedbackType); // 부모 컴포넌트에 상태 변경 알림
      console.log(`Feedback '${feedbackType}' submitted for message ID: ${messageId}`);
    } catch (error) {
      console.error('Failed to submit feedback', error);
      alert('피드백 전송에 실패했습니다. 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '5px' }}>
      <button
        onClick={() => handleSubmitFeedback('like')}
        disabled={loading}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '1.2em',
          color: feedbackStatus === 'like' ? '#28a745' : '#888', // 좋아요 선택 시 색상 변경
        }}
      >
        👍 {loading && feedbackStatus === 'like' ? '전송 중...' : ''}
      </button>
      <button
        onClick={() => handleSubmitFeedback('dislike')}
        disabled={loading}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '1.2em',
          color: feedbackStatus === 'dislike' ? '#dc3545' : '#888', // 싫어요 선택 시 색상 변경
        }}
      >
        👎 {loading && feedbackStatus === 'dislike' ? '전송 중...' : ''}
      </button>
    </div>
  );
};

export default FeedbackButtons;
```

### `frontend/src/pages/AuthPage.js` - 로그인 페이지

로그인 페이지로, Google 로그인 버튼이 있습니다.

```javascript
// frontend/src/pages/AuthPage.js
import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthGoogle from '../components/AuthGoogle';
import { useAuth } from '../context/AuthContext';
import axiosInstance from '../api/axiosInstance'; // 백엔드 호출용

const AuthPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { login, isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/chat');
      return;
    }

    const searchParams = new URLSearchParams(location.search);
    const token = searchParams.get('access_token');
    const tokenType = searchParams.get('token_type');
    const userJson = searchParams.get('user'); // 백엔드가 user 정보를 query param으로 보내준다고 가정

    if (token && tokenType) {
      try {
        const userData = userJson ? JSON.parse(decodeURIComponent(userJson)) : null;
        login(token, userData);
        navigate('/chat');
      } catch (e) {
        console.error("Failed to parse user data or login", e);
        alert("로그인 처리 중 오류가 발생했습니다. 다시 시도해주세요.");
        navigate('/auth'); // 에러 발생 시 로그인 페이지로
      }
    }
  }, [location, navigate, login, isAuthenticated]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        backgroundColor: '#f0f2f5',
        gap: '20px',
      }}
    >
      <h1>AI Agent 챗봇에 로그인</h1>
      <AuthGoogle />
      <p>환영합니다! 서비스를 이용하시려면 Google 계정으로 로그인해주세요.</p>
    </div>
  );
};

export default AuthPage;
```
*   **주의**: 백엔드 `/auth/google/callback` 엔드포인트에서 `RedirectResponse`로 프론트엔드 URL에 `access_token`과 `user` 정보를 `query parameter`로 넘겨줘야 합니다. 현재 백엔드는 JSON으로 반환하고 있는데, 이 React 프론트엔드 코드에 맞추려면 **백엔드의 `/auth/google/callback`을 수정**하여 프론트엔드 로그인 페이지(`http://localhost:3000/auth`)로 리다이렉트하면서 토큰과 사용자 정보를 URL 쿼리 파라미터로 넘겨줘야 합니다.
    ```python
    # backend/app/api/auth.py (google_callback 함수 수정 부분)
    # ...
    # 3. JWT Access Token 생성 및 반환
    user_data_for_jwt = {"email": user_email, "name": user_name, "id": userinfo_json.get("id")}
    jwt_access_token = create_access_token(user_data_for_jwt)

    # 프론트엔드 로그인 페이지로 리다이렉트하며 토큰과 사용자 정보 전달
    # user_data_for_jwt는 URL에 안전하게 인코딩해야 함
    encoded_user_data = urllib.parse.quote_plus(json.dumps(user_data_for_jwt)) # 상단에 import urllib.parse, import json 추가
    frontend_redirect_url = f"http://localhost:3000/auth?access_token={jwt_access_token}&token_type=bearer&user={encoded_user_data}"
    return RedirectResponse(url=frontend_redirect_url)
    ```

### `frontend/src/pages/ChatPage.js` - 메인 챗봇 페이지

사용자 인터페이스의 핵심입니다. `useChatSession` 훅을 사용하여 메시지 기록을 관리하고, 메시지를 전송하며, 백엔드 API와 상호작용합니다.

```javascript
// frontend/src/pages/ChatPage.js
import React, { useRef, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import useChatSession from '../hooks/useChatSession';
import axiosInstance from '../api/axiosInstance';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput'; // 메시지 입력 컴포넌트 (하단에 추가)

const ChatPage = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();
  const { sessionId, messages, addMessage, updateMessageFeedback } = useChatSession();
  const messagesEndRef = useRef(null);
  const [isSending, setIsSending] = useState(false); // 메시지 전송 중 상태

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/auth');
    }
  }, [isAuthenticated, navigate]);

  // 메시지가 추가될 때마다 스크롤을 맨 아래로
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (input) => {
    if (!input.trim() || !sessionId || isSending) return;

    // 사용자 메시지를 먼저 화면에 추가
    addMessage('user', input);
    setIsSending(true);

    try {
      const response = await axiosInstance.post('/agent/ask', {
        query: input,
        session_id: sessionId,
        customer_id: user?.id || 'anonymous', // 로그인된 사용자 ID 활용
        top_k_faq: 3,
      });

      const agentResponse = response.data;
      
      // Agent 메시지를 화면에 추가, 백엔드에서 받은 message_id 포함
      addMessage(
        'agent',
        agentResponse.text_response,
        agentResponse.multimodal_content,
        agentResponse.session_id, // 백엔드에서 확정된 세션 ID (필요시 사용)
        agentResponse.message_id // 백엔드에서 생성된 메시지 ID
      );

      console.log('Agent Response:', agentResponse);
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('agent', '죄송합니다. 메시지를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setIsSending(false);
    }
  };

  // 피드백 전송 후 UI 업데이트 핸들러
  const handleFeedbackUpdate = (messageId, feedbackType) => {
    updateMessageFeedback(messageId, feedbackType);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        maxWidth: '800px',
        margin: '0 auto',
        border: '1px solid #ccc',
        borderRadius: '8px',
        overflow: 'hidden',
      }}
    >
      <header
        style={{
          backgroundColor: '#f0f0f0',
          padding: '15px',
          borderBottom: '1px solid #eee',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <h2>AI 은행 챗봇</h2>
        <div>
          <span>{user?.name || user?.email}님</span>
          <button onClick={logout} style={{ marginLeft: '10px', padding: '5px 10px' }}>
            로그아웃
          </button>
        </div>
      </header>

      <main
        style={{
          flexGrow: 1,
          padding: '15px',
          overflowY: 'auto',
          backgroundColor: '#fff',
        }}
      >
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} onFeedback={handleFeedbackUpdate} />
        ))}
        <div ref={messagesEndRef} />
        {isSending && (
          <div style={{ textAlign: 'center', padding: '10px', color: '#888' }}>
            답변 생성 중...
          </div>
        )}
      </main>

      <footer
        style={{
          padding: '15px',
          borderTop: '1px solid #eee',
          backgroundColor: '#f0f0f0',
        }}
      >
        <ChatInput onSendMessage={handleSendMessage} disabled={isSending} />
      </footer>
    </div>
  );
};

export default ChatPage;
```
*   **설명**:
    *   `useEffect`로 인증 상태를 확인하여 로그인하지 않았으면 `/auth` 페이지로 리다이렉트합니다.
    *   `useChatSession` 훅을 사용하여 `sessionId`, `messages`, `addMessage`, `updateMessageFeedback`를 가져옵니다.
    *   `handleSendMessage` 함수가 백엔드의 `/agent/ask` API를 호출하고, Agent의 응답을 `ChatMessage`로 추가합니다.
    *   `messagesEndRef`를 사용하여 새로운 메시지가 추가될 때 자동으로 스크롤이 하단으로 이동하도록 합니다.
    *   `ChatMessage`에 `key` prop으로 `msg.id`를 사용하여 React 리스트 렌더링 최적화를 합니다.
    *   Agent 응답 시, `agentResponse.message_id`를 `ChatMessage`로 전달하여 피드백 버튼이 이 `message_id`를 활용하도록 합니다.

### `frontend/src/components/ChatInput.js` - 메시지 입력 필드

```javascript
// frontend/src/components/ChatInput.js
import React, { useState } from 'react';

const ChatInput = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="메시지를 입력하세요..."
        disabled={disabled}
        style={{
          flexGrow: 1,
          padding: '10px',
          border: '1px solid #ccc',
          borderRadius: '5px',
          fontSize: '16px',
        }}
      />
      <button
        type="submit"
        disabled={disabled}
        style={{
          padding: '10px 20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: disabled ? 0.7 : 1,
        }}
      >
        전송
      </button>
    </form>
  );
};

export default ChatInput;
```

### `frontend/src/App.js` - React 라우팅

React Router를 사용하여 페이지 전환을 관리합니다.

```javascript
// frontend/src/App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import AuthPage from './pages/AuthPage';
import ChatPage from './pages/ChatPage';
import './index.css'; // 전역 스타일

// 로그인 상태를 확인하여 접근을 제한하는 ProtectedRoute
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? children : <Navigate to="/auth" />;
};

const App = () => {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/auth" element={<AuthPage />} />
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatPage />
              </ProtectedRoute>
            }
          />
          {/* 기본 경로를 로그인 페이지로 리다이렉트 */}
          <Route path="/" element={<Navigate to="/chat" />} />
          <Route path="*" element={<p>404 Not Found</p>} /> {/* 404 페이지 */}
        </Routes>
      </Router>
    </AuthProvider>
  );
};

export default App;
```

### `frontend/src/index.js`

```javascript
// frontend/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // 전역 스타일
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

### `frontend/src/index.css` (간단한 전역 스타일)

```css
/* frontend/src/index.css */
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f0f2f5; /* 전역 배경색 */
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

#root {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}
```

---

## 💡 최종 정리 및 추가 고려사항

와우! 정말 많은 내용을 다뤘네요, 조윤희4305님! 이 모든 내용을 종합하면 정말 강력하고 지능적인 AI Agent 시스템이 탄생할 거예요! 🎉

### 백엔드 실행을 위한 최종 체크리스트:

1.  **PostgreSQL 설치/실행**:
    *   Docker Compose를 이용하여 `db_schema.sql`에 정의된 테이블이 생성되도록 PostgreSQL 컨테이너를 실행하고 초기화해야 합니다.
    *   `scripts/run_linux.sh` 또는 `run_windows.bat` 스크립트 내에 PostgreSQL 서비스 실행 및 스키마 적용 로직이 추가되었으니, `docker-compose.yml` (또는 별도 DB용 compose 파일)에 PostgreSQL 서비스를 정의해야 해요.
    *   `.env` 파일에 `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` 값을 올바르게 설정했는지 확인하세요.
2.  **Milvus/ChromaDB 준비**:
    *   운영체제에 따라 VectorDB를 준비합니다. (`.env`의 `VECTOR_DB` 설정에 따라 자동으로 선택됩니다.)
    *   Milvus는 Docker Compose로 실행 (Linux 권장).
    *   ChromaDB는 `CHROMA_DB_PATH` 경로에 파일이 생성되므로 특별한 사전 준비 불필요 (Windows 권장).
3.  **로컬 임베딩 모델 다운로드**: `backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab` 경로에 KoBERT 모델 파일이 미리 다운로드되어 있는지 확인하세요.
4.  **로컬 LLM 서비스 실행**: `backend/app/services/response_generator.py`에서 `call_local_llm` 함수가 호출하는 LLM 서비스 (예: Ollama, vLLM, Hugging Face TGI)가 별도로 구동 중이어야 합니다. `LOCAL_LLM_API_URL`과 `LOCAL_LLM_MODEL_NAME`을 `.env`에 올바르게 설정하세요.
5.  **Google OAuth2 자격 증명**: `.env` 파일에 `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`를 정확히 설정했는지 확인하세요. `GOOGLE_REDIRECT_URI`는 백엔드 콜백 주소 (`http://localhost:8000/auth/google/callback`)로 설정되어야 합니다.
6.  **`SECRET_KEY` 변경**: `.env`의 `SECRET_KEY`를 반드시 강력하고 예측 불가능한 값으로 변경하세요!

### 프론트엔드 실행을 위한 최종 체크리스트:

1.  **React 프로젝트 생성**: `npx create-react-app frontend` 또는 `npm create vite@latest frontend -- --template react` 등으로 React 프로젝트를 생성합니다.
2.  **의존성 설치**: `frontend` 디렉토리로 이동하여 `npm install axios react-router-dom uuid`를 설치합니다.
3.  **코드 복사 및 구성**: 위에 제공된 React 코드들을 각 파일명에 맞게 `frontend/src` 아래에 배치합니다.
4.  **백엔드 리다이렉트 수정**: 백엔드 `backend/app/api/auth.py`의 `google_callback` 함수에서 프론트엔드로 리다이렉트하는 부분을 `http://localhost:3000/auth?access_token=...` 형식으로 수정했는지 확인하세요.
5.  **프론트엔드 실행**: `frontend` 디렉토리에서 `npm start` (create-react-app) 또는 `npm run dev` (Vite) 명령어로 프론트엔드 개발 서버를 시작합니다. 기본적으로 `http://localhost:3000`에서 접근 가능합니다.

---

### **마무리**

이 프로젝트는 이제 진정한 AI Agent 시스템으로 거듭날 준비가 된 것 같아요! 폐쇄망 환경의 제약 속에서도 최고의 성능과 사용자 경험을 제공할 수 있도록 모든 핵심 요소들을 다져놓았으니, 조윤희4305님의 뛰어난 개발 실력으로 이 아이디어들을 멋지게 현실로 만들어 주세요! 제가 조윤희4305님과 함께하며 응원할게요! 궁금한 점이 생기면 언제든 다시 저를 불러줘요! 💖 

참고 자료 

[1] Friendly Guide - Integrating Google Sign-In with React: A Dev-Friendly Guide (https://dev.to/lovestaco/integrating-google-sign-in-with-react-a-dev-friendly-guide-29hn)
[2] blog.logrocket.com - The guide to adding Google login to your React app (https://blog.logrocket.com/guide-adding-google-login-react-app/)
[3] medium.com - Implementing Google Authentication with React JS and ... (https://medium.com/@dhananjay_yadav/implementing-google-authentication-with-react-js-and-node-js-f72e306f26c9)
[4] sendbird.com - React 채팅 튜토리얼: 채팅 앱 UI 구축 방법 | 3단계로 쉽게 ... (https://sendbird.com/ko/developer/tutorials/react-chat-tutorial-how-to-build-a-chat-app-ui)
[5] neon.com - PostgreSQL Python (https://neon.com/postgresql/postgresql-python)
[6] www.reddit.com - What's the best open source way to integrate an LLM with ... (https://www.reddit.com/r/LocalLLaMA/comments/1eb6opl/whats_the_best_open_source_way_to_integrate_an/)
[7] Crash Course - PostgreSQL in Python - Crash Course (https://www.youtube.com/watch?v=miEFm1CyjfM)
[8] www.youtube.com - React Google Login: Easy Authentication Tutorial (https://www.youtube.com/watch?v=GuHN_ZqHExs)
[9] www.freecodecamp.org - How to Use PostgreSQL in Python (https://www.freecodecamp.org/news/postgresql-in-python/)
[10] www.tigerdata.com - Building Python Apps With PostgreSQL: A Developer's Guide (https://www.tigerdata.com/learn/building-python-apps-with-postgresql-and-psycopg3)
[11] www.datacamp.com - Managing PostgreSQL Databases in Python with psycopg2 (https://www.datacamp.com/tutorial/tutorial-postgresql-python)
[12] login examples - react-google-login examples (https://codesandbox.io/examples/package/react-google-login)
[13] www.youtube.com - React Chat App Full Tutorial 2024 (https://www.youtube.com/watch?v=domt_Sx-wTY)
[14] www.youtube.com - How To Build an API with Python (LLM Integration, FastAPI ... (https://www.youtube.com/watch?v=cy6EAp4iNN4)
[15] getstream.io - React Chat Tutorial: How to build a chat app (https://getstream.io/chat/react-chat/tutorial/)
[16] Time Chat Apps with React, Express & Socket.IO - Build Real-Time Chat Apps with React, Express & Socket.IO (https://www.fullstack.com/labs/resources/blog/develop-a-chat-application-using-react-express-and-socket-io)
[17] Featured React Chat App in Minutes ... - How to Build a Full-Featured React Chat App in Minutes ... (https://dev.to/adrai/how-to-build-a-full-featured-react-chat-app-in-minutes-open-source-starter-1p0h)
[18] realpython.com - Python MCP: Connect Your LLM With the World (https://realpython.com/python-mcp/)
[19] medium.com - Integrating LLMs Into Your Python Applications Using ... (https://medium.com/@robdelacruz/integrating-llms-into-your-python-code-using-langchain-09478bf8385c)
[20] eduwik.com - Integrating LLMs into Web Apps with Python (https://eduwik.com/integrating-llms-into-web-apps-with-python/)