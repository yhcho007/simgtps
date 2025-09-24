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
            from openai import AsyncOpenAI  # 비동기 클라이언트 사용
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

