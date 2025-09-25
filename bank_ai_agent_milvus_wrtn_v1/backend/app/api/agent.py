# backend/app/api/agent.py
import uuid  # session_id 자동 생성용 (프론트엔드에서 넘겨주는 경우 대비)
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from app.core.config import settings
from app.core.embeddings import get_embedding_model, EmbeddingModel
from app.core.tools import get_all_tools, Tool
from app.services.response_generator import generate_agent_response, MultimodalAgentResponse
from app.core.common_vector_store import AbstractVectorStore
from app.main import get_vector_store, get_session_manager  # main.py에서 DI용 함수 임포트
from app.core.security import get_current_user
from app.services.session_manager import SessionManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
embedding_model: EmbeddingModel = get_embedding_model()  # 전역으로 로드된 임베딩 모델 사용
available_tools = {tool.name: tool for tool in get_all_tools()}


class AgentRequest(BaseModel):
    query: str = Field(..., example="휴면 계좌 잔고를 조회하고, 저에게 맞는 대출 상품도 추천해주세요.")
    session_id: Optional[str] = Field(None, description="현재 채팅 세션 ID (없으면 백엔드에서 생성)")
    customer_id: str = Field("user123", example="user123", description="고객 식별 ID (내부 API 연동용)")
    top_k_faq: int = Field(3, description="FAQ 검색 시 가져올 상위 결과 개수")


@router.post("/ask", response_model=MultimodalAgentResponse)
async def ask_agent(
        request: AgentRequest,
        vector_store: AbstractVectorStore = Depends(get_vector_store),
        session_manager: SessionManager = Depends(get_session_manager),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    사용자의 질문을 분석하여 VectorDB RAG, 내부 API 연동, 다른 Agent 연동,
    멀티모달 응답 등을 활용하여 최적의 답변을 생성합니다.
    """
    user_email = current_user.get("email", "unknown_user")
    current_session_id = request.session_id if request.session_id else str(uuid.uuid4())

    try:
        # 0. 세션 시작/로드 및 히스토리 추가
        session = await session_manager.start_or_load_session_async(current_session_id, user_email)
        session.add_message("user", request.query)  # 인메모리 세션에 메시지 추가
        logger.info(f"세션 {session.session_id}에서 사용자 '{user_email}'의 요청: {request.query}")

        # 1. 쿼리 임베딩
        query_embedding = await embedding_model.embed_query(request.query)

        # 2. Vector DB (FAQ RAG) 검색
        collection_name = settings.MILVUS_COLLECTION_NAME if settings.SELECTED_VECTOR_DB == "milvus" else settings.CHROMA_COLLECTION_NAME
        try:
            faq_results = await vector_store.search(collection_name, query_embedding, request.top_k_faq)
            faq_context = "\n".join([f"FAQ: {r.text} (거리: {r.distance:.4f})" for r in faq_results])
        except Exception as e:
            logger.error(f"Vector DB 검색 중 오류 발생: {e}", exc_info=True)
            faq_context = "FAQ 검색에 실패했습니다."

        # 3. 툴 사용 결정 및 실행 (간단한 키워드 기반 로직으로 대체)
        tool_outputs = []
        if "휴면 계좌" in request.query or "거래 내역" in request.query:
            internal_api_tool = available_tools.get("InternalAccountAPI")
            if internal_api_tool:
                try:
                    tool_output = await internal_api_tool.run(customer_id=request.customer_id, query=request.query)
                    tool_outputs.append(f"InternalAccountAPI 응답: {tool_output}")
                except Exception as e:
                    logger.warning(f"InternalAccountAPI 호출 실패: {e}", exc_info=True)
                    tool_outputs.append(f"InternalAccountAPI 호출 실패: {e}")

        # ... (나머지 툴 호출 로직은 동일하게 유지하되, 각 호출마다 try-except 추가) ...

        # 4. 멀티모달 리소스 검색
        multimodal_resource = None
        if "주택담보대출 가이드" in request.query or "은행 카드 혜택" in request.query:
            multimodal_tool = available_tools.get("MultimodalResourceLookup")
            if multimodal_tool:
                try:
                    resource_output = await multimodal_tool.run(query=request.query)
                    if resource_output.get("status") == "success":
                        multimodal_resource = {
                            "type": resource_output["resource_type"],
                            "url": f"/static/documents/{resource_output['path']}",
                            "summary": resource_output["text_summary"]
                        }
                        tool_outputs.append(f"멀티모달 리소스 검색: {multimodal_resource['summary']}")
                except Exception as e:
                    logger.warning(f"MultimodalResourceLookup 호출 실패: {e}", exc_info=True)
                    tool_outputs.append(f"MultimodalResourceLookup 호출 실패: {e}")

        # 5. 최종 응답 생성 (폐쇄망 LLM 호출 포함)
        final_response_obj = await generate_agent_response(
            user_query=request.query,
            faq_context=faq_context,
            tool_outputs="\n".join(tool_outputs),
            multimodal_resource=multimodal_resource,
            chat_history=session.get_history()  # 세션 히스토리 전달 (LLM이 대화 맥락 이해하도록)
        )

        # 6. 세션 히스토리에 Agent 응답 추가
        session.add_message("agent", final_response_obj.text_response)

        return final_response_obj

    except Exception as e:
        logger.error(f"Agent 요청 처리 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"에이전트 요청 처리 실패: {e}")