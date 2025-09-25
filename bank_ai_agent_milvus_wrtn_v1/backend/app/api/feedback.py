# backend/app/api/feedback.py
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, Any
from app.core.security import get_current_user
from app.main import get_stats_manager, get_session_manager  # DI 함수 임포트
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., example="unique_session_id", description="피드백을 제공하는 세션 ID")
    message_id: str = Field(..., example="agent_response_001", description="피드백 대상 응답의 ID (클라이언트에서 생성하여 넘겨야 함)")
    feedback_type: str = Field(..., example="like", description="피드백 타입: 'like' 또는 'dislike'")


@router.post("/submit")
async def submit_feedback(
        request: FeedbackRequest,
        stats_manager: StatsManager = Depends(get_stats_manager),
        session_manager: SessionManager = Depends(get_session_manager),
        current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    사용자의 응답 피드백 (좋아요/싫어요)을 수집합니다.
    """
    user_email = current_user.get("email", "unknown_user")

    if request.feedback_type not in ["like", "dislike"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="유효하지 않은 피드백 타입입니다. 'like' 또는 'dislike'를 사용해주세요.")

    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"세션 ID '{request.session_id}'를 찾을 수 없습니다.")

    # 특정 message_id에 대한 쿼리와 응답 내용을 찾아서 저장 (프론트엔드에서 넘어오는 message_id 활용)
    # 실제 구현에서는 message_id를 정확히 매핑하는 로직이 필요. 여기서는 간단히 가장 최근 쿼리-응답 쌍을 찾음
    user_query = "찾을 수 없음"
    agent_response_text = "찾을 수 없음"

    for i in range(len(session.history) - 1, 0, -1):
        if session.history[i].get("role") == "agent":
            agent_response_text = session.history[i].get("content", "")
            if i > 0 and session.history[i - 1].get("role") == "user":
                user_query = session.history[i - 1].get("content", "")
            break

    try:
        await stats_manager.add_feedback(
            user_email=user_email,
            session_id=request.session_id,
            message_id=request.message_id,
            feedback_type=request.feedback_type,
            user_query=user_query,
            agent_response=agent_response_text
        )
        return {"message": "피드백이 성공적으로 기록되었습니다."}
    except RuntimeError as e:
        logger.error(f"피드백 저장 실패: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="피드백 저장 중 오류가 발생했습니다.")
