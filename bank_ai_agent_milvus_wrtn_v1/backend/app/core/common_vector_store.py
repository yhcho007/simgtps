# backend/app/core/common_vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorSearchResult:
    def __init__(self, text: str, metadata: Dict[str, Any], distance: float, id: Optional[Any] = None):
        self.text = text
        self.metadata = metadata
        self.distance = distance
        self.id = id  # 각 벡터 DB의 고유 ID를 저장할 수 있도록 추가


class AbstractVectorStore(ABC):
    @abstractmethod
    async def check_connection(self):
        """Vector DB 연결 상태를 확인합니다."""
        pass

    @abstractmethod
    async def check_collection_exists(self, collection_name: str) -> bool:
        """지정된 컬렉션/코덱스가 존재하는지 확인합니다."""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, dim: int):
        """지정된 이름과 차원으로 컬렉션/코덱스를 생성합니다."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str):
        """컬렉션/코덱스를 삭제합니다."""
        pass

    @abstractmethod
    async def insert(self, collection_name: str, texts: List[str], embeddings: List[List[float]],
                     metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """데이터를 삽입합니다."""
        pass

    @abstractmethod
    async def search(self, collection_name: str, query_embedding: List[float], top_k: int) -> List[VectorSearchResult]:
        """벡터 검색을 수행합니다."""
        pass

# get_vector_store 함수는 아래에서 구현합니다.