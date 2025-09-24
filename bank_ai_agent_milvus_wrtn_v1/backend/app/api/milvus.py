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
