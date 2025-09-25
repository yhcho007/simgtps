# backend/app/api/dashboard.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from app.core.security import get_current_user
from app.main import get_stats_manager, get_session_manager # DI 함수 임포트
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ⚠️ 주의: 실제 서비스에서는 관리자만 접근 가능하도록 추가 권한 확인 로직 필요
# 예를 들어, current_user의 role이 'admin'인지 확인
def is_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    # 이메일 등으로 관리자 여부 판단 (임시)
    if not current_user.get("email") in ["admin@example.com", "your_admin_email@domain.com"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="관리자만 접근 가능합니다.")
    return current_user

@router.get("/metrics")
async def get_chatbot_metrics(
    stats_manager: StatsManager = Depends(get_stats_manager),
    admin_user: Dict[str, Any] = Depends(is_admin_user)
):
    """
    챗봇의 전반적인 성능 지표를 반환합니다. (관리자 전용)
    """
    logger.info(f"대시보드 성능 지표 요청: user={admin_user['email']}")
    metrics = stats_manager.get_performance_metrics()
    return metrics

@router.get("/active_sessions")
async def get_active_sessions(
    session_manager: SessionManager = Depends(get_session_manager),
    admin_user: Dict[str, Any] = Depends(is_admin_user)
) -> List[Dict[str, Any]]:
    """
    현재 활성화된 사용자 세션 목록을 반환합니다. (관리자 전용)
    """
    logger.info(f"활성 세션 목록 요청: user={admin_user['email']}")
    sessions = [s.to_dict() for s in session_manager.get_all_active_sessions()]
    return sessions
