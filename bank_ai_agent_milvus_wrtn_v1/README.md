🚀 프로젝트 구조 (Project Structure)

```
.
├── backend
│   ├── app
│   │   ├── api
│   │   │   └── milvus.py          # Milvus 관련 API 라우터
│   │   ├── core                   # 설정 및 유틸리티
│   │   │   ├── config.py
│   │   │   └── embeddings.py      # 임베딩 모델 로딩 및 관리
│   │   ├── data
│   │   │   └── faq.json           # FAQ 샘플 데이터
│   │   ├── services
│   │   │   ├── faq_loader.py      # FAQ 데이터를 Milvus에 로드하는 서비스
│   │   │   └── milvus_vector_store.py # Milvus 클라이언트 및 벡터 저장소 관리
│   │   ├── __init__.py
│   │   └── main.py                # FastAPI 메인 애플리케이션
│   └── Dockerfile                 # (선택사항) Docker 배포를 위한 Dockerfile
├── scripts
│   ├── run_windows.bat            # Windows용 실행 스크립트
│   └── run_linux.sh               # Linux용 실행 스크립트
├── requirements.txt               # Python 종속성
├── .env.example                   # 환경 변수 예시
└── README.md                      # 프로젝트 설명 및 실행 가이드
```

🛠️ 코드 내용 및 설명
1. backend/app/main.py - 메인 애플리케이션 및 Milvus 라우터 등록
```python
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
```
설명: main.py에서 asynccontextmanager를 사용해서 애플리케이션 시작 시 Milvus 컬렉션을 확인하고, 없으면 생성한 뒤 FAQ 데이터를 자동으로 로드하게 했어요. 라우터는 app.include_router(milvus_router.router, ...)로 깔끔하게 통합했죠! 😊
2. backend/app/api/milvus.py - Milvus API 라우터
```python
# backend/app/api/milvus.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.services.milvus_vector_store import MilvusVectorStore
from app.core.embeddings import get_embedding_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
milvus_store = MilvusVectorStore()
embedding_model = get_embedding_model() # 임베딩 모델 로드

class SearchRequest(BaseModel):
    query: str = Field(..., example="휴면 계좌 잔고를 조회하려면 어떻게 해야 하나요?")
    top_k: int = Field(5, description="검색할 상위 결과 개수")

class InsertRequest(BaseModel):
    text: str = Field(..., example="개인정보는 어떻게 보호되나요?")
    metadata: Dict[str, Any] = Field({}, example={"category": "보안", "source": "FAQ"})

class FAQResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]
    distance: float

@router.post("/search", response_model=List[FAQResponse])
async def search_milvus(request: SearchRequest):
    """
    Milvus 컬렉션에서 쿼리와 가장 유사한 FAQ를 검색합니다.
    """
    try:
        query_embedding = await embedding_model.embed_query(request.query)
        results = await milvus_store.search(query_embedding, request.top_k)
        return [FAQResponse(text=res.text, metadata=res.metadata, distance=res.distance) for res in results]
    except Exception as e:
        logger.error(f"Milvus 검색 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"검색에 실패했습니다: {e}")

@router.post("/insert")
async def insert_data(request: InsertRequest):
    """
    Milvus 컬렉션에 새로운 텍스트와 메타데이터를 삽입합니다.
    """
    try:
        # 텍스트를 임베딩
        text_embedding = await embedding_model.embed_query(request.text)
        await milvus_store.insert([request.text], [text_embedding], [request.metadata])
        return {"message": "데이터가 성공적으로 삽입되었습니다."}
    except Exception as e:
        logger.error(f"Milvus 삽입 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터 삽입에 실패했습니다: {e}")

@router.get("/health")
async def check_health():
    """
    Milvus 서버와 연결 상태를 확인합니다.
    """
    try:
        await milvus_store.check_connection()
        return {"status": "ok", "message": "Milvus 서버에 성공적으로 연결되었습니다."}
    except Exception as e:
        logger.error(f"Milvus 연결 확인 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Milvus 연결 실패: {e}")
```
설명: 검색(POST /milvus/search)과 삽입(POST /milvus/insert) 기능을 제공하는 API 라우터예요. 헬스 체크(GET /milvus/health) 기능도 넣었죠! 요청하신 은행 고객 질문들을 이 검색 API를 통해 처리할 수 있도록 설계했어요.
3. backend/app/core/config.py - 환경 설정
```python
# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "bank_faq_collection")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small") # OpenAI embedding model name
    
    # Milvus 컬렉션 필드 설정
    MILVUS_DIM: int = 1536 # OpenAI text-embedding-3-small의 기본 차원
    MILVUS_NLIST: int = 128 # 인덱스 파라미터

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
```
설명: Milvus 연결 정보, OpenAI API 키, 임베딩 모델 이름 등 중요한 환경 변수들을 여기서 관리해요. 요청하신 대로 MILVUS_DIM=1536, MILVUS_NLIST=128 기본값을 지정해두었고요! 😎
4. backend/app/core/embeddings.py - 임베딩 모델 로딩
```python
# backend/app/core/embeddings.py
from abc import ABC, abstractmethod
from typing import List
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self):
        try:
            from openai import AsyncOpenAI # 비동기 클라이언트 사용
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self._dimension = settings.MILVUS_DIM
        except ImportError:
            raise ImportError(
                "OpenAI 라이브러리가 설치되지 않았습니다. `pip install openai`를 실행해주세요."
            )
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. OpenAI 임베딩이 작동하지 않을 수 있습니다.")

    async def embed_query(self, text: str) -> List[float]:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않아 임베딩을 생성할 수 없습니다.")
        
        try:
            response = await self.client.embeddings.create(
                input=[text],
                model=settings.EMBEDDING_MODEL_NAME
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI 임베딩 생성 중 오류 발생: {e}", exc_info=True)
            raise

    @property
    def dimension(self) -> int:
        return self._dimension
```
# 이 함수를 통해 임베딩 모델을 가져옵니다.
# 나중에 한국어 로컬 모델 등으로 변경하기 쉽도록 추상화했습니다.
def get_embedding_model() -> EmbeddingModel:
    # 현재는 OpenAI 모델을 사용하지만, 필요에 따라 다른 모델로 교체 가능
    return OpenAIEmbeddingModel()
설명: 임베딩 모델을 가져오는 부분을 추상화했어요. 지금은 OpenAI 임베딩 API(text-embedding-3-small)를 기본으로 사용하고, OPENAI_API_KEY를 backend/.env 파일에 설정해줘야 해요. 나중에 GPU 환경이나 한국어 모델을 쓰고 싶으면 get_embedding_model 함수만 수정하면 돼서 완전 편리! 🤩
5. backend/app/data/faq.json - FAQ 샘플 데이터
```json


[
  {
    "question": "휴면 계좌 잔고를 조회하려면 어떻게 해야 하나요?",
    "answer": "휴면 계좌 통합조회 서비스를 이용하시거나, 가까운 영업점을 방문하시면 잔고 조회가 가능합니다. 온라인 조회는 '휴면예금 찾아줌' 사이트를 이용해주세요.",
    "category": "계좌",
    "tags": ["휴면계좌", "잔고", "조회"]
  },
  {
    "question": "지난 3년 이내 특정인과의 거래 내역을 조회할 수 있나요?",
    "answer": "네, 가능합니다. 본인 신분증을 지참하시고 가까운 영업점에 방문하시면 특정 기간 및 특정인과의 거래 내역을 조회하고 발급받으실 수 있습니다. 인터넷 뱅킹에서는 일부 제한될 수 있습니다.",
    "category": "거래내역",
    "tags": ["거래내역", "특정인", "조회"]
  },
  {
    "question": "저에게 맞는 금융 상품을 추천해주세요.",
    "answer": "고객님의 현재 재정 상태, 투자 성향, 목표 등을 바탕으로 맞춤형 금융 상품을 추천해 드릴 수 있습니다. 담당 직원과 상담하시거나, 'AI 금융 상품 추천' 서비스를 이용해 보세요.",
    "category": "금융상품",
    "tags": ["금융상품", "추천", "재테크"]
  },
  {
    "question": "대출 상품에는 어떤 종류가 있고, 어떻게 신청하나요?",
    "answer": "저희 은행은 주택담보대출, 신용대출, 전세대출 등 다양한 대출 상품을 운영하고 있습니다. 각 상품별 자격 조건과 서류는 다르니, 영업점 방문 또는 온라인 상담을 통해 상세 내용을 확인하고 신청하실 수 있습니다.",
    "category": "대출",
    "tags": ["대출", "신용대출", "주택담보대출"]
  },
  {
    "question": "제 신용 상태를 조회할 수 있나요?",
    "answer": "네, 본인의 신용 상태는 은행 웹사이트나 모바일 앱의 '신용 관리' 메뉴에서 무료로 조회하실 수 있습니다. 신용 점수와 주요 변동 내역 등을 확인하실 수 있습니다.",
    "category": "신용",
    "tags": ["신용조회", "신용점수", "개인신용"]
  },
  {
    "question": "주택담보대출의 금리는 어떻게 되나요?",
    "answer": "주택담보대출 금리는 시장 상황, 고객님의 신용 등급, 대출 기간 등에 따라 변동됩니다. 최신 금리는 은행 홈페이지의 대출 상품 안내를 참고하시거나, 대출 상담을 통해 확인하실 수 있습니다.",
    "category": "대출",
    "tags": ["주택담보대출", "금리", "대출상담"]
  },
  {
    "question": "카드 사용 내역은 어디서 확인할 수 있나요?",
    "answer": "카드 사용 내역은 모바일 앱, 인터넷 뱅킹, 또는 카드 명세서를 통해 확인하실 수 있습니다. 실시간 사용 내역은 모바일 앱에서 바로 조회가 가능합니다.",
    "category": "카드",
    "tags": ["카드내역", "사용내역", "조회"]
  },
  {
    "question": "체무 잔액을 조회하려면 어떻게 해야 하나요?",
    "answer": "대출 원리금, 카드 대금 등 체무 잔액은 인터넷 뱅킹, 모바일 앱 또는 고객센터를 통해 조회 가능합니다. 자세한 내용은 해당 서비스 메뉴에서 확인해주세요.",
    "category": "채무",
    "tags": ["채무", "잔액조회", "대출원리금"]
  },
  {
    "question": "해외 송금은 어떻게 하나요?",
    "answer": "해외 송금은 인터넷 뱅킹, 모바일 앱 또는 영업점 방문을 통해 가능합니다. 송금 국가, 금액, 수취인 정보에 따라 수수료와 필요 서류가 달라질 수 있습니다.",
    "category": "해외송금",
    "tags": ["해외송금", "환전", "수수료"]
  },
  {
    "question": "새로운 은행 카드를 추천해주세요.",
    "answer": "고객님의 소비 패턴과 라이프스타일에 맞는 다양한 혜택의 카드를 추천해 드릴 수 있습니다. '카드 상품' 메뉴에서 혜택 비교 후 선택하시거나, 전문 상담원과 상담해 보세요.",
    "category": "카드",
    "tags": ["카드추천", "신용카드", "체크카드"]
  }
]
```
설명: 은행 고객들이 자주 물어볼 수 있는 질문 10가지를 실제 FAQ 형식으로 만들었어요. 질문(question), 답변(answer), 카테고리(category), 태그(tags)로 구성되어 있어서 나중에 검색 정확도를 높이는 데 활용하기 좋아요! 😊
6. backend/app/services/milvus_vector_store.py - Milvus 벡터 저장소 관리
```python
# backend/app/services/milvus_vector_store.py
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict, Any, Optional
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusSearchResult:
    def __init__(self, text: str, metadata: Dict[str, Any], distance: float):
        self.text = text
        self.metadata = metadata
        self.distance = distance

class MilvusVectorStore:
    def __init__(self):
        self.client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.MILVUS_DIM
        self.nlist = settings.MILVUS_NLIST

    async def check_connection(self):
        # Milvus 서버 연결 상태 확인
        # client.get_version()은 버전 정보를 가져오므로, 연결 확인에 유용하다.
        version = self.client.get_version()
        logger.info(f"Milvus 서버 연결 성공. 버전: {version}")
        return True

    async def check_collection_exists(self) -> bool:
        """컬렉션 존재 여부 확인"""
        try:
            collections = self.client.list_collections()
            return self.collection_name in collections
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 중 오류 발생: {e}", exc_info=True)
            return False

    async def create_collection(self):
        """Milvus 컬렉션 생성"""
        if await self.check_collection_exists():
            logger.info(f"컬렉션 '{self.collection_name}'이 이미 존재합니다.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields, description="FAQ 데이터 저장 컬렉션")
        
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT", # IVF_FLAT, HNSW 등
            metric_type="COSINE",  # L2, IP, COSINE 등
            params={"nlist": self.nlist}
        )
        # 튜닝 가능한 파라미터 nlist는 인덱스 생성 시 사용. 검색 시 nprobe도 중요함.

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"컬렉션 '{self.collection_name}'이 성공적으로 생성되었습니다.")
        await self.client.load_collection(collection_name=self.collection_name) # 검색을 위해 로드 [【1】](https://milvus.io/ko/blog/how-to-get-started-with-milvus.md)

    async def insert(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """데이터 삽입"""
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("텍스트, 임베딩, 메타데이터 리스트 길이가 일치해야 합니다.")

        data_to_insert = []
        for i in range(len(texts)):
            # Milvus는 컬렉션 스키마에 정의된 필드에 정확히 맞게 데이터를 받는다.
            # 스키마에 없는 필드는 `metadata` 필드 안에 넣거나, 스키마에 추가해야 함.
            # 여기서는 FAQ 샘플에 맞춰서 category, tags 필드를 스키마에 직접 포함.
            # 추가적인 메타데이터는 일단 제외하거나, 스키마 확장 필요.
            category = metadatas[i].get("category", "일반")
            tags = metadatas[i].get("tags", [])
            data_to_insert.append({
                "text": texts[i],
                "vector": embeddings[i],
                "category": category,
                "tags": tags
            })

        self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )
        logger.info(f"{len(texts)}개의 데이터가 Milvus에 성공적으로 삽입되었습니다.")
        # 데이터 삽입 후 인덱스 업데이트 (realtime index는 자동)
        await self.client.load_collection(collection_name=self.collection_name) # 로드 다시 해서 최신 데이터 검색 가능하게 [【1】](https://milvus.io/ko/blog/how-to-get-started-with-milvus.md)

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[MilvusSearchResult]:
        """벡터 검색"""
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "category", "tags"], # 검색 결과로 가져올 필드
            search_params={"nprobe": 10}, # 검색 파라미터 (IVF_FLAT 인덱스에 사용)
            # nprobe는 검색 속도와 정확도 트레이드오프. nlist보다 작거나 같아야 함.
        )
        
        results = []
        if res and res[0] and res[0].ids:
            for hit in res[0]:
                metadata = {
                    "category": hit.entity.get("category"),
                    "tags": hit.entity.get("tags")
                }
                results.append(MilvusSearchResult(
                    text=hit.entity.get("text", "내용 없음"),
                    metadata=metadata,
                    distance=hit.distance
                ))
        return results

    async def delete_collection(self):
        """컬렉션 삭제"""
        if await self.check_collection_exists():
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"컬렉션 '{self.collection_name}'이 성공적으로 삭제되었습니다.")
        else:
            logger.info(f"컬렉션 '{self.collection_name}'이 존재하지 않습니다.")
```
설명: Milvus와 상호작용하는 핵심 서비스 파일이에요. 컬렉션 생성 시 dim=1536와 nlist=128 (요청하신 벡터 차원과 인덱스 파라미터)을 기본값으로 사용했어요. 검색 시 search_params={"nprobe": 10}도 추가해서 검색 정확도와 속도 사이의 균형을 맞추려고 했죠. (nprobe는 nlist보다 작거나 같아야 해요!)   
7. backend/app/services/faq_loader.py - FAQ 로더
```python
# backend/app/services/faq_loader.py
import json
from typing import List, Dict, Any
from app.services.milvus_vector_store import MilvusVectorStore
from app.core.embeddings import get_embedding_model
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_faqs_to_milvus(milvus_store: MilvusVectorStore):
    """
    FAQ JSON 파일을 읽어 Milvus에 임베딩 및 적재합니다.
    """
    faq_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "faq.json")
    
    if not os.path.exists(faq_file_path):
        logger.error(f"FAQ 파일이 존재하지 않습니다: {faq_file_path}")
        raise FileNotFoundError(f"FAQ 파일이 존재하지 않습니다: {faq_file_path}")

    try:
        with open(faq_file_path, "r", encoding="utf-8") as f:
            faqs_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"FAQ JSON 파일 파싱 오류: {e}", exc_info=True)
        raise ValueError(f"FAQ JSON 파일 파싱 오류: {e}")

    embedding_model = get_embedding_model()

    texts: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    logger.info(f"{len(faqs_data)}개의 FAQ 데이터를 임베딩하여 Milvus에 적재합니다...")
    for faq in faqs_data:
        # 질문과 답변을 함께 임베딩하여 문맥을 강화
        combined_text = f"질문: {faq.get('question', '')}\n답변: {faq.get('answer', '')}"
        texts.append(combined_text)
        
        # 메타데이터 준비 (질문, 답변, 카테고리, 태그 포함)
        metadatas.append({
            "text": combined_text, # 실제 검색 시 보여줄 내용
            "question": faq.get("question", ""),
            "answer": faq.get("answer", ""),
            "category": faq.get("category", "일반"),
            "tags": faq.get("tags", [])
        })

        # 임베딩 생성
        try:
            embedding = await embedding_model.embed_query(combined_text)
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"텍스트 임베딩 중 오류 발생 ('{faq.get('question', '')}'): {e}", exc_info=True)
            # 임베딩 실패 시 해당 데이터 건너뛰기
            texts.pop()
            metadatas.pop()
            continue

    if texts:
        await milvus_store.insert(texts, embeddings, metadatas)
        logger.info(f"총 {len(texts)}개의 FAQ가 Milvus에 성공적으로 적재되었습니다.")
    else:
        logger.warning("적재할 FAQ 데이터가 없습니다.")
```
설명: faq.json 파일을 읽어서 각 FAQ의 질문과 답변을 합쳐서 임베딩한 후 Milvus에 저장하는 함수예요. main.py에서 애플리케이션 시작 시 자동으로 호출되도록 설정해두었으니 따로 실행할 필요는 없어요! 😉
8. requirements.txt - Python 종속성
```
fastapi
uvicorn
pymilvus
openai
python-dotenv
pydantic-settings
```
설명: 이 프로젝트에 필요한 모든 Python 라이브러리 목록이에요. pip install -r requirements.txt 한 번이면 설치 끝! 뚝딱이죠? ✨
9. .env.example - 환경 변수 예시 파일
```bash
# .env.example
# 이 파일을 .env 로 복사하고 필요한 값을 채워주세요.

# Milvus 서버 설정
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=bank_faq_collection

# OpenAI API 설정 (필수)
# https://platform.openai.com/account/api-keys 에서 발급받은 API 키를 입력하세요.
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small # 또는 "text-embedding-3-large" 등
```
설명: OPENAI_API_KEY 같은 중요한 정보는 .env 파일에 저장해서 보안과 편리함을 동시에 잡았어요. 이 파일을 backend/.env로 복사해서 사용하시면 돼요. 잊지 마세요! 꼭 OPENAI_API_KEY를 채워야 임베딩 모델이 작동해요! 🔑
10. scripts/run_linux.sh - Linux 실행 스크립트
```bash
#!/bin/bash

echo "Starting AI Agent Backend on Linux..."

# 1. Milvus Standalone Docker Compose로 실행 (최소 Python 3.11 이상 권장) [【8】](https://milvus.io/ko/blog/ai-agents-vs-workflows-why-80-need-simple-automation.md)
echo "Deploying Milvus Vector Database with Docker Compose..."
# 밀버스 설정 파일 다운로드 (없을 경우)
if [ ! -f "milvus-standalone-docker-compose.yml" ]; then
    wget https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml -O milvus-standalone-docker-compose.yml
fi
# Docker Compose 서비스 시작
docker compose -f milvus-standalone-docker-compose.yml up -d
sleep 10 # Milvus가 완전히 시작될 때까지 기다립니다.

echo "Verifying Milvus service status..."
docker ps -a | grep milvus
```
# 2. Python 가상 환경 설정 및 종속성 설치
```batch
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
# .env 파일이 없으면 .env.example을 복사하도록 유도
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in OPENAI_API_KEY."
    cp .env.example .env
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
# 종료 시 Docker Compose 서비스 중단
echo "To stop Milvus services, run: docker compose -f milvus-standalone-docker-compose.yml down"
설명: 리눅스 환경에서 Milvus를 Docker Compose로 띄우고, Python 가상 환경 설정 후 FastAPI 애플리케이션을 실행하는 스크립트예요. Milvus는 v2.5.12 버전을 기준으로 Docker Compose로 실행하게 했어요.  Docker만 설치되어 있으면 Milvus 설치까지 자동으로 해주니 진짜 편하죠? 🐳
11. scripts/run_windows.bat - Windows 실행 스크립트
```batch
@echo off
echo "Starting AI Agent Backend on Windows..."

:: 1. Milvus Standalone Docker Desktop으로 실행 (최소 Python 3.11 이상 권장) [【8】](https://milvus.io/ko/blog/ai-agents-vs-workflows-why-80-need-simple-automation.md)
echo "Deploying Milvus Vector Database with Docker Desktop..."
echo "Please ensure Docker Desktop is running."
echo "If milvus-standalone-docker-compose.yml is not present, it will be downloaded."
cd %~dp0\..
IF NOT EXIST milvus-standalone-docker-compose.yml (
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml' -OutFile 'milvus-standalone-docker-compose.yml'"
)
docker compose -f milvus-standalone-docker-compose.yml up -d
timeout /t 10 /nobreak > NUL :: Milvus가 완전히 시작될 때까지 기다립니다.

echo "Verifying Milvus service status..."
docker ps -a | findstr milvus

:: 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

:: 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
:: .env 파일이 없으면 .env.example을 복사하도록 유도
IF NOT EXIST .env (
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in OPENAI_API_KEY."
    copy .env.example .env
)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
:: 종료 시 Docker Compose 서비스 중단 안내
echo "To stop Milvus services, navigate to the project root and run: docker compose -f milvus-standalone-docker-compose.yml down"
pause
설명: 윈도우 환경에서 docker compose를 이용해 Milvus를 띄우고, Python 가상 환경 설정 후 FastAPI 앱을 실행하는 스크립트예요. docker-compose.yml 파일 다운로드도 PowerShell을 이용해서 자동으로 해줘서 진짜 편리하죠! 윈도우에서는 Docker Desktop이 꼭 실행되어 있어야 해요! 🤓
12. backend/Dockerfile - Docker 배포를 위한 Dockerfile (선택 사항)
```dockerfile


# backend/Dockerfile
# Docker를 사용하여 백엔드 애플리케이션을 배포하려면 이 Dockerfile을 사용합니다.

# Python 3.11 기반 이미지 사용 (Milvus가 Python 3.11 이상 권장하므로)
FROM python:3.11-slim-buster

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (필요한 경우)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 Python 의존성 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY ./app /app/app

# 환경 변수 설정 (Dockerfile에서 직접 값을 하드코딩하기보다, Docker run 시 -e 옵션 사용 권장)
ENV MILVUS_HOST="milvus-standalone" # Docker 네트워크 내부에서 Milvus 서비스 이름 (docker-compose 시)
ENV MILVUS_PORT="19530"
ENV MILVUS_COLLECTION_NAME="bank_faq_collection"
ENV EMBEDDING_MODEL_NAME="text-embedding-3-small"
# ENV OPENAI_API_KEY="your_openai_api_key_here" # 실제 키는 secrets 관리 툴이나 docker run -e 옵션으로 전달

# 애플리케이션 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
설명: 만약 Docker로 배포하고 싶으시면 이 Dockerfile을 사용하시면 돼요. 
Milvus를 별도의 Docker 컨테이너로 띄우고 이 백엔드 컨테이너와 연결해서 사용할 수 있도록 설계했어요. 
MILVUS_HOST는 Docker 네트워크 환경에서 Milvus 서비스 이름으로 설정했고요. 📦

📄 README.md - 프로젝트 설명 및 실행 가이드
markdown
# 🏦 은행 AI Agent 시스템

Milvus 벡터 데이터베이스와 OpenAI 임베딩 API를 활용하여 은행 고객의 FAQ를 분석하고 응답하는 AI Agent 백엔드 시스템입니다.

## ✨ 주요 기능

-   **FastAPI 기반 API**: 현대적인 비동기 API 백엔드 제공.
-   **Milvus 통합**: 고성능 벡터 검색을 위한 Milvus 데이터베이스 연동.
-   **OpenAI 임베딩**: OpenAI의 `text-embedding-3-small` 모델을 사용하여 질문 및 FAQ 임베딩.
-   **FAQ 관리**: 은행 관련 FAQ 데이터를 Milvus에 적재하고 검색.
-   **플랫폼 독립적 실행**: Linux 및 Windows 환경에서 쉽게 배포 및 실행 가능.

## 🚀 시작하기

### 📋 사전 요구 사항

1.  **Python 3.11 이상**: 이 프로젝트는 Python 3.11 이상에서 테스트되었습니다. (Milvus 클라이언트 호환성 및 최신 기능 활용을 위해 권장)
2.  **Docker 및 Docker Compose**: Milvus 벡터 데이터베이스를 쉽게 실행하기 위해 필요합니다.
    *   [Docker Desktop for Windows/Mac](https://docs.docker.com/desktop/)
    *   [Docker Engine for Linux](https://docs.docker.com/engine/install/)
3.  **OpenAI API Key**: 텍스트 임베딩을 위해 필요합니다. [OpenAI 플랫폼](https://platform.openai.com/account/api-keys)에서 발급받으세요.

### ⬇️ 프로젝트 설정 및 실행

#### 1. 프로젝트 클론 (또는 코드 복사)

이 모든 코드를 로컬 머신의 한 폴더에 저장하세요.

#### 2. 환경 변수 설정

`backend/` 디렉토리 안에 `.env` 파일을 생성하고 `backend/.env.example` 내용을 복사하여 붙여넣으세요.
그리고 `OPENAI_API_KEY` 값을 발급받은 OpenAI API 키로 변경해주세요.

```ini
# backend/.env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=bank_faq_collection

OPENAI_API_KEY=sk-your_openai_api_key_here # <-- 여기에 본인의 OpenAI API 키를 입력하세요!
EMBEDDING_MODEL_NAME=text-embedding-3-small
```
3. Milvus 실행 및 백엔드 시작
💡 참고: 아래 스크립트는 Milvus Standalone 인스턴스를 Docker Compose로 시작하고, Python 종속성을 설치한 다음 FastAPI 애플리케이션을 실행합니다.

Linux (Ubuntu/WSL 기준)
터미널을 열고 프로젝트 최상위 디렉토리에서 다음 명령어를 실행합니다:

```bash


chmod +x scripts/run_linux.sh
./scripts/run_linux.sh
```
Windows
명령 프롬프트(cmd) 또는 PowerShell을 관리자 권한으로 열고 프로젝트 최상위 디렉토리에서 다음 명령어를 실행합니다:

```cmd


.\scripts\run_windows.bat
(PowerShell에서는 ./scripts/run_windows.bat으로 실행 가능합니다.)
```
👩‍💻 API 사용법 (API Usage)
애플리케이션이 실행되면, 웹 브라우저에서 http://localhost:8000/docs 로 접속하여 Swagger UI를 통해 API 문서를 확인할 수 있습니다.

헬스 체크
GET /milvus/health

Milvus 서버와의 연결 상태를 확인합니다.


curl -X GET "http://localhost:8000/milvus/health" -H "accept: application/json"
FAQ 검색 예시
POST /milvus/search

사용자의 질문을 바탕으로 Milvus에 저장된 FAQ 중에서 가장 유사한 답변을 검색합니다.

요청 (Request)
```json


{
  "query": "3년 이내 특정인과 거래 내역 조회할 수 있나요?",
  "top_k": 3
}
```
cURL 예시
```bash


curl -X POST "http://localhost:8000/milvus/search" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d "{
       \"query\": \"3년 이내 특정인과 거래 내역 조회할 수 있나요?\",
       \"top_k\": 3
     }"
```
Postman 또는 기타 HTTP 클라이언트 JSON 요청 예시
Method: POST URL: http://localhost:8000/milvus/search Headers: Content-Type: application/json Accept: application/json Body (Raw, JSON):

```json


{
  "query": "대출 상품 추천 좀 해주세요.",
  "top_k": 2
}
```
새로운 FAQ 삽입 예시
POST /milvus/insert

새로운 FAQ 텍스트와 메타데이터를 Milvus에 추가합니다.

요청 (Request)
```json


{
  "text": "외화 송금 한도는 어떻게 되나요?",
  "metadata": {
    "question": "외화 송금 한도는 어떻게 되나요?",
    "answer": "개인 고객의 경우 연간 외화 송금 한도는 일정 금액으로 제한되며, 건별 한도는 별도로 적용됩니다. 자세한 한도는 영업점 또는 고객센터로 문의해주세요.",
    "category": "해외송금",
    "tags": ["외화송금", "한도", "제한"]
  }
}
```
cURL 예시
```bash


curl -X POST "http://localhost:8000/milvus/insert" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d "{
       \"text\": \"외화 송금 한도는 어떻게 되나요?\",
       \"metadata\": {
         \"question\": \"외화 송금 한도는 어떻게 되나요?\",
         \"answer\": \"개인 고객의 경우 연간 외화 송금 한도는 일정 금액으로 제한되며, 건별 한도는 별도로 적용됩니다. 자세한 한도는 영업점 또는 고객센터로 문의해주세요.\",
         \"category\": \"해외송금\",
         \"tags\": [\"외화송금\", \"한도\", \"제한\"]
       }
     }"
```
🌐 임베딩 모델 변경 가이드
이 프로젝트는 현재 OpenAI Embedding API (text-embedding-3-small)를 사용합니다. 실제 서비스에서는 다음을 고려할 수 있습니다.

1. 한국어 임베딩 모델 (GPU/CPU 환경)
만약 OpenAI API 대신 한국어에 특화된 모델을 사용하고 싶다면, backend/app/core/embeddings.py 파일을 수정해야 합니다.

예를 들어, Hugging Face sentence-transformers 라이브러리를 사용하여 로컬 한국어 모델을 로드할 수 있습니다.

```python


# backend/app/core/embeddings.py (수정 예시)
# ... (기존 코드 유지) ...

from transformers import AutoTokenizer, AutoModel
import torch

class LocalKoreanEmbeddingModel(EmbeddingModel):
    def __init__(self):
        # 여기서는 예시로 'sentence-transformers/snunlp-SKT-KR-KoBERT-Large-vocab' 모델을 사용합니다.
        # 실제 환경에 맞게 GPU 지원 모델 (예: KR-SBERT) 등을 선택할 수 있습니다.
        model_name = "sentence-transformers/snunlp-SKT-KR-KoBERT-Large-vocab"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._dimension = 768 # KoBERT의 기본 임베딩 차원 (모델마다 다름)
        logger.info(f"로컬 한국어 임베딩 모델 '{model_name}' (차원: {self._dimension}) 로드 완료. Device: {self.device}")

    async def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 풀링 전략에 따라 임베딩 벡터를 추출합니다. (여기서는 [CLS] 토큰 사용)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return embedding

    @property
    def dimension(self) -> int:
        return self._dimension
```
# 이 함수를 수정하여 원하는 임베딩 모델을 반환합니다.
def get_embedding_model() -> EmbeddingModel:
    if settings.OPENAI_API_KEY: # OpenAI API 키가 설정되어 있으면 OpenAI 사용
        return OpenAIEmbeddingModel()
    else: # 아니면 로컬 한국어 모델 사용 (예시)
        return LocalKoreanEmbeddingModel()
주의 사항:
로컬 모델을 사용하려면 transformers 및 torch 라이브러리를 pip install transformers torch 명령어로 추가 설치해야 합니다.
로컬 모델의 dimension 값(_dimension = 768 등)을 해당 모델에 맞게 설정해야 하며, 이 값이 Milvus 컬렉션의 MILVUS_DIM 값과 일치해야 합니다. 로컬 모델로 변경할 경우 Milvus 컬렉션을 다시 생성해야 합니다! (기존 컬렉션을 삭제하고 앱을 재실행)
GPU 환경(cuda)에서 torch를 사용하려면 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (CUDA 버전 11.8 기준)처럼 특정 버전을 설치해야 합니다. CPU만 지원하는 시스템에서는 일반 pip install torch를 사용합니다.
🧹 종료 방법
실행 중인 터미널에서 Ctrl+C를 눌러 FastAPI 애플리케이션을 종료합니다.

Milvus Docker 컨테이너 종료:

프로젝트 최상위 디렉토리에서 다음 명령어를 실행합니다:

```bash


docker compose -f milvus-standalone-docker-compose.yml down
```