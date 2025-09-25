# backend/app/services/faq_loader.py
import json
from typing import List, Dict, Any
from app.core.config import settings
from app.core.common_vector_store import AbstractVectorStore
from app.core.embeddings import EmbeddingModel
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# load_faqs_to_milvus 함수명을 범용적으로 변경하는 것이 좋지만, 기존 호출부와의 호환성을 위해 유지.
# 내부 로직은 추상화된 AbstractVectorStore를 사용.
async def load_faqs_to_milvus(vector_store: AbstractVectorStore, embedding_model: EmbeddingModel):
    """
    FAQ JSON 파일을 읽어 임베딩 후 벡터 DB에 적재합니다.
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

    texts: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    logger.info(f"{len(faqs_data)}개의 FAQ 데이터를 임베딩하여 벡터 DB에 적재합니다...")
    for faq in faqs_data:
        combined_text = f"질문: {faq.get('question', '')}\n답변: {faq.get('answer', '')}"
        texts.append(combined_text)

        metadatas.append({
            "text": combined_text,  # 실제 검색 시 보여줄 내용
            "question": faq.get("question", ""),
            "answer": faq.get("answer", ""),
            "category": faq.get("category", "일반"),
            "tags": faq.get("tags", [])
        })

        try:
            embedding = await embedding_model.embed_query(combined_text)
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"텍스트 임베딩 중 오류 발생 ('{faq.get('question', '')}'): {e}", exc_info=True)
            texts.pop()
            metadatas.pop()
            continue

    if texts:
        collection_name = settings.MILVUS_COLLECTION_NAME if settings.SELECTED_VECTOR_DB == "milvus" else settings.CHROMA_COLLECTION_NAME
        await vector_store.insert(collection_name, texts, embeddings, metadatas)
        logger.info(f"총 {len(texts)}개의 FAQ가 {settings.SELECTED_VECTOR_DB}에 성공적으로 적재되었습니다.")
    else:
        logger.warning("적재할 FAQ 데이터가 없습니다.")