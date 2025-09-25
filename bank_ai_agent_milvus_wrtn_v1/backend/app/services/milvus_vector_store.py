# backend/app/services/milvus_vector_store.py
from pymilvus import MilvusClient, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.common_vector_store import AbstractVectorStore, VectorSearchResult
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusVectorStore(AbstractVectorStore):
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        logger.info(f"Milvus 클라이언트 초기화: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

    async def check_connection(self):
        try:
            version = self._client.get_version()
            logger.info(f"Milvus 서버 연결 성공. 버전: {version}")
            return True
        except Exception as e:
            logger.error(f"Milvus 연결 확인 실패: {e}", exc_info=True)
            raise ConnectionError(f"Milvus 서버에 연결할 수 없습니다: {e}")

    async def check_collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self._client.list_collections()
            return collection_name in collections
        except Exception as e:
            logger.error(f"Milvus 컬렉션 목록 조회 중 오류 발생: {e}", exc_info=True)
            raise

    async def create_collection(self, collection_name: str, dim: int):
        if await self.check_collection_exists(collection_name):
            logger.info(f"Milvus 컬렉션 '{collection_name}'이 이미 존재합니다.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),  # 텍스트 길이 증가
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=512)
        ]
        schema = CollectionSchema(fields, description="FAQ 데이터 저장 컬렉션")

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": settings.MILVUS_NLIST}
        )

        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Milvus 컬렉션 '{collection_name}'이 성공적으로 생성되었습니다. DIM: {dim}")
        await self._client.load_collection(collection_name=collection_name)

    async def insert(self, collection_name: str, texts: List[str], embeddings: List[List[float]],
                     metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("텍스트, 임베딩, 메타데이터 리스트 길이가 일치해야 합니다.")

        data_to_insert = []
        for i in range(len(texts)):
            # 메타데이터에서 question, answer 등을 직접 필드로 매핑
            data_to_insert.append({
                "text": texts[i],
                "vector": embeddings[i],
                "question": metadatas[i].get("question", ""),
                "answer": metadatas[i].get("answer", ""),
                "category": metadatas[i].get("category", "일반"),
                "tags": metadatas[i].get("tags", [])
            })

        self._client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        logger.info(f"{len(texts)}개의 데이터가 Milvus 컬렉션 '{collection_name}'에 성공적으로 삽입되었습니다.")
        await self._client.load_collection(collection_name=collection_name)

    async def search(self, collection_name: str, query_embedding: List[float], top_k: int) -> List[VectorSearchResult]:
        res = self._client.search(
            collection_name=collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "question", "answer", "category", "tags"],
            search_params={"nprobe": settings.MILVUS_NPROBE},  # nprobe 설정 추가
            # metric_type="COSINE"는 create_collection에서 정의되므로, 검색 시에는 재정의 불필요.
        )

        results = []
        if res and res[0] and res[0].ids:
            for hit in res[0]:
                metadata = {
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "category": hit.entity.get("category"),
                    "tags": hit.entity.get("tags")
                }
                results.append(VectorSearchResult(
                    text=hit.entity.get("text", "내용 없음"),
                    metadata=metadata,
                    distance=hit.distance,
                    id=hit.id  # Milvus의 ID 포함
                ))
        return results

    async def delete_collection(self, collection_name: str):
        if await self.check_collection_exists(collection_name):
            self._client.drop_collection(collection_name=collection_name)
            logger.info(f"Milvus 컬렉션 '{collection_name}'이 성공적으로 삭제되었습니다.")
        else:
            logger.info(f"Milvus 컬렉션 '{collection_name}'이 존재하지 않습니다.")
