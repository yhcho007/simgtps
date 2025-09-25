# backend/app/services/chroma_vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.common_vector_store import AbstractVectorStore, VectorSearchResult
import logging

logger = logging.getLogger(__name__)


class ChromaVectorStore(AbstractVectorStore):
    def __init__(self):
        # 영구 저장소 사용 (폐쇄망 노트북 환경에 적합)
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        logger.info(f"ChromaDB 클라이언트 초기화: {settings.CHROMA_DB_PATH}")

    async def check_connection(self):
        try:
            # ChromaDB는 로컬 파일 기반이라 단순한 기능 호출로 확인
            version = self._client.get_version()
            logger.info(f"ChromaDB 서버 연결 성공. 버전: {version}")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 연결 확인 실패: {e}", exc_info=True)
            raise ConnectionError(f"ChromaDB에 연결할 수 없습니다: {e}")

    async def check_collection_exists(self, collection_name: str) -> bool:
        try:
            self._client.get_collection(name=collection_name)
            return True
        except Exception:  # Collection does not exist
            return False

    async def create_collection(self, collection_name: str, dim: int):
        if await self.check_collection_exists(collection_name):
            logger.info(f"ChromaDB 컬렉션 '{collection_name}'이 이미 존재합니다.")
            return

        # ChromaDB는 컬렉션 생성 시 임베딩 함수를 지정하지만, 여기서는 외부에서 임베딩을 받으므로 None
        # 차원(dim)은 삽입 시 자동으로 처리되거나, 임베딩 함수에 의해 결정
        collection = self._client.create_collection(name=collection_name)
        logger.info(f"ChromaDB 컬렉션 '{collection_name}'이 성공적으로 생성되었습니다.")
        # ChromaDB는 Milvus처럼 명시적인 load_collection 과정이 필요 없음

    async def insert(self, collection_name: str, texts: List[str], embeddings: List[List[float]],
                     metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        if len(texts) != len(embeddings) or len(texts) != len(metadatas):
            raise ValueError("텍스트, 임베딩, 메타데이터 리스트 길이가 일치해야 합니다.")

        collection = self._client.get_or_create_collection(name=collection_name)

        # ChromaDB는 고유 ID를 요구. 없으면 자동으로 생성
        if ids is None:
            ids = [f"id-{i}" for i in range(len(texts))]  # 임의의 ID 생성 또는 UUID 사용
            # 실제 사용 시에는 FAQ ID 등 고유한 값을 사용하는 것이 좋습니다.

        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"{len(texts)}개의 데이터가 ChromaDB 컬렉션 '{collection_name}'에 성공적으로 삽입되었습니다.")

    async def search(self, collection_name: str, query_embedding: List[float], top_k: int) -> List[VectorSearchResult]:
        collection = self._client.get_or_create_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'distances', 'metadatas']
        )

        search_results = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                search_results.append(VectorSearchResult(
                    text=document,
                    metadata=metadata,
                    distance=distance,
                    id=doc_id
                ))
        return search_results

    async def delete_collection(self, collection_name: str):
        if await self.check_collection_exists(collection_name):
            self._client.delete_collection(name=collection_name)
            logger.info(f"ChromaDB 컬렉션 '{collection_name}'이 성공적으로 삭제되었습니다.")
        else:
            logger.info(f"ChromaDB 컬렉션 '{collection_name}'이 존재하지 않습니다.")
