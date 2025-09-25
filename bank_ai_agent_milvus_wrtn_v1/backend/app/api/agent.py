# backend/app/api/agent.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union
from app.services.milvus_vector_store import MilvusVectorStore
from app.core.embeddings import get_embedding_model
from app.core.tools import get_all_tools, Tool
from app.services.response_generator import generate_agent_response, MultimodalAgentResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
milvus_store = MilvusVectorStore()
embedding_model = get_embedding_model()  # 임베딩 모델 로드 (로컬 or OpenAI)
available_tools = {tool.name: tool for tool in get_all_tools()}  # 에이전트가 사용할 수 있는 툴


class AgentRequest(BaseModel):
    query: str = Field(..., example="휴면 계좌 잔고를 조회하고, 저에게 맞는 대출 상품도 추천해주세요.")
    customer_id: str = Field("user123", example="user123", description="고객 식별 ID (내부 API 연동용)")
    top_k_faq: int = Field(3, description="FAQ 검색 시 가져올 상위 결과 개수")


@router.post("/ask", response_model=MultimodalAgentResponse)
async def ask_agent(request: AgentRequest):
    """
    사용자의 질문을 분석하여 Milvus RAG, 내부 API 연동, 다른 Agent 연동,
    멀티모달 응답 등을 활용하여 최적의 답변을 생성합니다.
    """
    try:
        # LLM 기반 에이전트가 여기서 툴 사용 여부를 결정합니다.
        # 폐쇄망 환경에서는 오픈소스 LLM(예: Llama 2, Mistral 등)을 로컬에 배포하거나,
        # 정교하게 설계된 프롬프트를 통해 룰-기반/템플릿-기반으로 툴을 선택하도록 구현해야 합니다.
        # 여기서는 복잡한 LLM 연동 대신, 질의에 따라 가상의 툴 사용 로직을 구현합니다.

        # 1. 쿼리 임베딩
        query_embedding = await embedding_model.embed_query(request.query)

        # 2. Milvus (FAQ RAG) 검색
        faq_results = await milvus_store.search(query_embedding, request.top_k_faq)
        faq_context = "\n".join([f"FAQ: {r.text} (거리: {r.distance:.4f})" for r in faq_results])

        # 3. 툴 사용 결정 및 실행 (간단한 키워드 기반 로직으로 대체)
        # 실제로는 여기서 LLM(local LLM)이 query와 faq_context를 보고 어떤 툴을 쓸지,
        # 어떤 인자로 툴을 쓸지 결정하는 "Function Calling" 혹은 "Tool Use" 로직이 들어갑니다.
        tool_outputs = []
        if "휴면 계좌" in request.query or "거래 내역" in request.query:
            internal_api_tool = available_tools.get("InternalAccountAPI")
            if internal_api_tool:
                tool_output = await internal_api_tool.run(customer_id=request.customer_id, query=request.query)
                tool_outputs.append(f"InternalAccountAPI 응답: {tool_output}")

        if "대출 상품" in request.query or "금융 상품" in request.query:
            financial_tool = available_tools.get("FinancialProductRecommendationAPI")
            if financial_tool:
                tool_output = await financial_tool.run(customer_id=request.customer_id, criteria=request.query,
                                                       product_type="대출")
                tool_outputs.append(f"FinancialProductRecommendationAPI 응답 (대출): {tool_output}")

        if "신용 상태" in request.query:
            credit_agent_tool = available_tools.get("ExternalCreditScoreAgent")
            if credit_agent_tool:
                tool_output = await credit_agent_tool.run(customer_id=request.customer_id)
                tool_outputs.append(f"ExternalCreditScoreAgent 응답: {tool_output}")

        if "카드 추천" in request.query:
            financial_tool = available_tools.get("FinancialProductRecommendationAPI")
            if financial_tool:
                tool_output = await financial_tool.run(customer_id=request.customer_id, criteria=request.query,
                                                       product_type="카드")
                tool_outputs.append(f"FinancialProductRecommendationAPI 응답 (카드): {tool_output}")

        # 4. 멀티모달 리소스 검색
        multimodal_resource = None
        if "주택담보대출 가이드" in request.query or "은행 카드 혜택" in request.query:
            multimodal_tool = available_tools.get("MultimodalResourceLookup")
            if multimodal_tool:
                resource_output = await multimodal_tool.run(query=request.query)
                if resource_output.get("status") == "success":
                    multimodal_resource = {
                        "type": resource_output["resource_type"],
                        "url": f"/static/documents/{resource_output['path']}",  # FastAPI static files 경로 가정
                        "summary": resource_output["text_summary"]
                    }
                    tool_outputs.append(f"멀티모달 리소스 검색: {multimodal_resource['summary']}")

        # 5. 최종 응답 생성 (가상 LLM)
        # 실제로는 FAQ context와 tool_outputs를 입력으로 LLM(local LLM)이 최종 답변을 생성합니다.
        final_response = await generate_agent_response(
            user_query=request.query,
            faq_context=faq_context,
            tool_outputs="\n".join(tool_outputs),
            multimodal_resource=multimodal_resource
        )
        return final_response

    except Exception as e:
        logger.error(f"Agent 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"에이전트 요청 처리 실패: {e}")
