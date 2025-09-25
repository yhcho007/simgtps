# backend/app/services/response_generator.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MultimodalContent(BaseModel):
    type: str = Field(..., example="pdf")  # "text", "pdf", "image", "video"
    url: str = Field(..., example="/static/documents/bank_loan_guide.pdf")  # 파일 접근 URL
    summary: str = Field(..., example="주택담보대출 가이드 문서입니다. 자세한 내용은 PDF를 참고하세요.")


class MultimodalAgentResponse(BaseModel):
    text_response: str = Field(..., example="안녕하세요, 고객님! 무엇을 도와드릴까요?")
    multimodal_content: Optional[MultimodalContent] = Field(None, description="PDF 문서, 이미지 등 추가 멀티모달 응답")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="디버깅을 위한 추가 정보 (배포 시 제거 권장)")


async def generate_agent_response(
        user_query: str,
        faq_context: str,
        tool_outputs: str,
        multimodal_resource: Optional[Dict[str, Any]] = None
) -> MultimodalAgentResponse:
    """
    에이전트의 최종 응답을 생성합니다.
    폐쇄망 LLM(또는 정교한 룰셋)이 FAQ 검색 결과와 툴 실행 결과를 바탕으로
    사용자에게 자연스러운 답변을 만들고, 필요시 멀티모달 리소스를 포함합니다.
    """
    final_text_response = "고객님의 질문을 받아 처리했습니다.\n"

    # FAQ 컨텍스트 반영
    if faq_context:
        final_text_response += "\n[FAQ 참고 내용]\n" + faq_context.split('\n')[0]  # 첫 줄만 예시로
        if len(faq_context.split('\n')) > 1:
            final_text_response += " (더 많은 FAQ가 검색되었습니다.)"

    # 툴 실행 결과 반영
    if tool_outputs:
        final_text_response += "\n[추가 정보]\n" + tool_outputs

    # 멀티모달 리소스 반영
    if multimodal_resource:
        final_text_response += f"\n\n관련 자료를 찾았습니다: {multimodal_resource['summary']}"
        multimodal_content_obj = MultimodalContent(
            type=multimodal_resource['type'],
            url=multimodal_resource['url'],
            summary=multimodal_resource['summary']
        )
    else:
        multimodal_content_obj = None

    final_text_response += "\n\n더 궁금한 점이 있으시면 언제든지 문의해주세요! 😊"

    # 실제 폐쇄망 LLM을 여기에 통합해야 합니다.
    # 예: local_llm_model.generate(prompt=f"{user_query}\n\n{faq_context}\n\n{tool_outputs}")
    # 현재는 placeholder 로직으로 동작합니다.

    logger.info(f"생성된 최종 텍스트 응답: {final_text_response}")

    return MultimodalAgentResponse(
        text_response=final_text_response,
        multimodal_content=multimodal_content_obj,
        debug_info={
            "user_query": user_query,
            "faq_context": faq_context,
            "tool_outputs": tool_outputs
        }
    )
