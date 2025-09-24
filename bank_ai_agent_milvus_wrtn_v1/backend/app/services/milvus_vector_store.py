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
            index_type="IVF_FLAT",  # IVF_FLAT, HNSW 등
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
        await self.client.load_collection(
            collection_name=self.collection_name)  # 검색을 위해 로드 [【1】](https://milvus.io/ko/blog/how-to-get-started-with-milvus.md)

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
        await self.client.load_collection(
            collection_name=self.collection_name)  # 로드 다시 해서 최신 데이터 검색 가능하게 [【1】](https://milvus.io/ko/blog/how-to-get-started-with-milvus.md)

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[MilvusSearchResult]:
        """벡터 검색"""
        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "category", "tags"],  # 검색 결과로 가져올 필드
            search_params={"nprobe": 10},  # 검색 파라미터 (IVF_FLAT 인덱스에 사용)
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
