# backend/app/core/embeddings.py
from abc import ABC, abstractmethod
from typing import List
from app.core.config import settings
import logging
import os

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
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self._dimension = settings.MILVUS_DIM  # config에서 정의된 차원 사용
            logger.info("OpenAI Embedding 모델 초기화.")
        except ImportError:
            logger.error("OpenAI 라이브러리가 설치되지 않았습니다. `pip install openai`를 실행해주세요.")
            raise
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. OpenAI 임베딩은 작동하지 않습니다.")

    async def embed_query(self, text: str) -> List[float]:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않아 임베딩을 생성할 수 없습니다. 로컬 모델을 사용하거나 키를 설정하세요.")
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


# --- 폐쇄망 환경을 위한 로컬 임베딩 모델 (KoBERT 예시) ---
# KoBERT는 sentence-transformers/snunlp-SKT-KR-KoBERT-Large-vocab를 사용하면 768 차원.
# 모델 파일은 미리 다운로드하여 접근 가능한 로컬 경로에 저장해야 합니다.
try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    logger.info("Hugging Face Transformers 및 PyTorch가 로드되었습니다.")
except ImportError:
    logger.warning(
        "Transformers 또는 PyTorch가 설치되지 않았습니다. 로컬 임베딩 모델은 사용 불가합니다. `pip install transformers torch`를 실행해주세요.")


class LocalKoreanEmbeddingModel(EmbeddingModel):
    def __init__(self):
        # 로컬 경로에서 모델 로드 (미리 다운로드 필요)
        # 예: "./models/snunlp-SKT-KR-KoBERT-Large-vocab"
        # 또는 인터넷이 잠깐 되는 환경에서 한 번 다운로드 후 경로 지정:
        # model_name_or_path = "sentence-transformers/snunlp-SKT-KR-KoBERT-Large-vocab"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.model = AutoModel.from_pretrained(model_name_or_path)

        # 폐쇄망을 위해 미리 다운로드된 모델 경로를 지정한다고 가정
        local_model_path = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "./models/snunlp-SKT-KR-KoBERT-Large-vocab")
        logger.info(f"로컬 임베딩 모델 '{local_model_path}' 로드 시도...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModel.from_pretrained(local_model_path)
        except Exception as e:
            logger.error(f"로컬 임베딩 모델 로드 실패: {e}", exc_info=True)
            logger.warning(f"'{local_model_path}' 경로에 모델 파일이 없거나 잘못되었습니다. 인터넷 연결이 가능할 때 미리 다운로드 받아 해당 경로에 두어야 합니다.")
            raise RuntimeError(f"로컬 임베딩 모델 로드 실패: {e}")

        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self.model.to(self.device)
        self.model.eval()  # 평가 모드 설정
        self._dimension = settings.MILVUS_DIM  # config에서 정의된 차원 사용 (KoBERT Large는 768)

        # 실제 모델의 차원을 확인하고 싶다면:
        # dummy_input = self.tokenizer("test", return_tensors='pt').to(self.device)
        # with torch.no_grad():
        #     dummy_output = self.model(**dummy_input).last_hidden_state
        # self._dimension = dummy_output.shape[-1]

        logger.info(f"로컬 한국어 임베딩 모델 초기화 완료: '{local_model_path}' (차원: {self._dimension}). Device: {self.device}")

    async def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # [CLS] 토큰 임베딩을 사용
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
        return embedding

    @property
    def dimension(self) -> int:
        return self._dimension


# 이 함수를 통해 임베딩 모델을 가져옵니다.
def get_embedding_model() -> EmbeddingModel:
    if settings.OPENAI_API_KEY and not settings.FORCE_LOCAL_EMBEDDING:
        logger.info("OPENAI_API_KEY가 설정되어 있고, 로컬 임베딩 강제 사용 설정이 아님: OpenAI 모델 사용")
        return OpenAIEmbeddingModel()
    else:
        logger.info("OPENAI_API_KEY가 없거나, 로컬 임베딩 강제 사용 설정: 로컬 임베딩 모델 사용 시도")
        return LocalKoreanEmbeddingModel()
