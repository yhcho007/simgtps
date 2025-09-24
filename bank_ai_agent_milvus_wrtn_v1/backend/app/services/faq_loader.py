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
            "text": combined_text,  # 실제 검색 시 보여줄 내용
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
