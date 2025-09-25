# backend/app/services/response_generator.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from app.core.config import settings
import logging
import httpx  # 로컬 LLM API 호출을 위해
import asyncio_throttle  # LLM 호출 제한을 위해 (옵션)

logger = logging.getLogger(__name__)

# LLM API 호출을 위한 Rate Limiter (선택 사항)
# 초당 5회 호출로 제한 (로컬 LLM 서비스의 부하를 고려하여 조절)
llm_call_throttle = asyncio_throttle.Throttle(rate_limit=5)


class MultimodalContent(BaseModel):
    type: str = Field(..., example="pdf")
    url: str = Field(..., example="/static/documents/bank_loan_guide.pdf")
    summary: str = Field(..., example="주택담보대출 가이드 PDF를 찾았습니다. 자세한 내용은 PDF를 참고하세요.")


class MultimodalAgentResponse(BaseModel):
    text_response: str = Field(..., example="안녕하세요, 고객님! 무엇을 도와드릴까요?")
    multimodal_content: Optional[MultimodalContent] = Field(None, description="PDF 문서, 이미지 등 추가 멀티모달 응답")
    session_id: str = Field(..., example="unique_session_id", description="현재 채팅 세션 ID")  # 프론트엔드 연동용
    message_id: str = Field(..., example="agent_response_001", description="생성된 응답의 고유 ID (피드백용)")  # 프론트엔드 연동용
    debug_info: Optional[Dict[str, Any]] = Field(None, description="디버깅을 위한 추가 정보 (배포 시 제거 권장)")


async def call_local_llm(prompt: str, chat_history: List[Dict[str, Any]]) -> str:
    """
    폐쇄망 로컬 LLM을 호출하여 응답을 생성합니다.
    settings.LOCAL_LLM_API_URL로 HTTP 요청을 보냅니다.
    """
    try:
        # LLM 모델에게 전달할 메시지 형식 (OpenAI API 형식과 유사)
        messages = [{"role": m["role"], "content": m["content"]} for m in chat_history[-5:]]  # 최근 5개 메시지만 전달
        messages.append({"role": "user", "content": prompt})  # 현재 프롬프트 추가

        async with llm_call_throttle:  # Rate Limiting 적용 (선택 사항)
            async with httpx.AsyncClient(timeout=30.0) as client:  # LLM 응답 시간 고려
                response = await client.post(
                    settings.LOCAL_LLM_API_URL,
                    json={
                        "model": settings.LOCAL_LLM_MODEL_NAME,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 500,
                        # 기타 LLM 서비스에 필요한 파라미터들
                    },
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()  # HTTP 오류가 발생하면 예외 발생
                response_data = response.json()

                # 로컬 LLM 서비스의 응답 구조에 따라 파싱
                # 예: Ollama, vLLM 등의 Chat Completion API 응답
                if response_data.get("choices") and response_data["choices"][0].get("message"):
                    return response_data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"로컬 LLM 응답 형식이 예상과 다릅니다: {response_data}")
                    return "죄송합니다. LLM이 올바른 형식으로 응답하지 않았습니다."

    except httpx.RequestError as e:
        logger.error(f"로컬 LLM API 연결 실패: {e}", exc_info=True)
        return "죄송합니다. LLM 서비스에 연결할 수 없습니다. 관리자에게 문의해주세요."
    except httpx.HTTPStatusError as e:
        logger.error(f"로컬 LLM API HTTP 오류 발생: {e.response.status_code} - {e.response.text}", exc_info=True)
        return f"죄송합니다. LLM 서비스에서 오류가 발생했습니다. ({e.response.status_code})"
    except Exception as e:
        logger.error(f"로컬 LLM 응답 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
        return "죄송합니다. 답변 생성 중 알 수 없는 오류가 발생했습니다."


async def generate_agent_response(
        user_query: str,
        faq_context: str,
        tool_outputs: str,
        multimodal_resource: Optional[Dict[str, Any]] = None,
        chat_history: List[Dict[str, Any]] = None  # 대화 히스토리 추가
) -> MultimodalAgentResponse:
    """
    에이전트의 최종 응답을 생성합니다. 폐쇄망 LLM이 FAQ 검색 결과와 툴 실행 결과를 바탕으로
    사용자에게 자연스러운 답변을 만들고, 필요시 멀티모달 리소스를 포함합니다.
    """
    if chat_history is None:
        chat_history = []

    # LLM에게 전달할 프롬프트 구성
    llm_prompt = f"""
    당신은 친절한 은행 챗봇 Agent 입니다. 고객의 질문에 대해 아래 제공된 정보들을 활용하여 답변해주세요.
    제공된 정보만으로 답변하기 어렵거나 추가적인 조치가 필요하면 그렇게 안내해주세요.

    [고객의 질문]
    {user_query}

    [검색된 FAQ 정보]
    {faq_context if faq_context else "관련 FAQ를 찾지 못했습니다."}

    [툴 실행 결과]
    {tool_outputs if tool_outputs else "실행된 툴이 없습니다."}

    {multimodal_resource['summary'] if multimodal_resource else ""}

    친절하고 명확하게 답변해주세요.
    """

    final_text_response = "답변 생성 중 오류가 발생했습니다."  # 기본 오류 메시지
    try:
        final_text_response = await call_local_llm(llm_prompt, chat_history)
    except Exception as e:
        logger.error(f"로컬 LLM 호출 중 오류 발생: {e}", exc_info=True)
        final_text_response = "죄송합니다. 답변 생성 서비스에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    # 멀티모달 콘텐츠 객체 생성
    multimodal_content_obj = None
    if multimodal_resource:
        multimodal_content_obj = MultimodalContent(
            type=multimodal_resource['type'],
            url=multimodal_resource['url'],
            summary=multimodal_resource['summary']
        )

    # 프론트엔드에서 사용할 message_id 생성 (각 응답마다 고유)
    message_id = f"agent_response_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    return MultimodalAgentResponse(
        text_response=final_text_response,
        multimodal_content=multimodal_content_obj,
        session_id=chat_history[-1].get("session_id", "unknown_session") if chat_history else "unknown_session",
        # 세션 ID도 응답에 포함
        message_id=message_id,
        debug_info={
            "user_query": user_query,
            "faq_context": faq_context,
            "tool_outputs": tool_outputs,
            "local_llm_prompt": llm_prompt  # 디버깅용
        }
    )
