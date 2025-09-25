# backend/app/services/stats_manager.py
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.core.database import execute_query
import logging

logger = logging.getLogger(__name__)


class StatsManager:
    _instance = None
    _feedback_data: List[Dict[str, Any]] = []  # In-memory cache for feedback
    _performance_metrics: Dict[str, Any] = {}
    _session_ref: Optional[Any] = None  # SessionManager 참조

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StatsManager, cls).__new__(cls)
            cls._feedback_data = []
            cls._performance_metrics = {
                "total_queries": 0,
                "avg_response_time_ms": 0.0,  # 아직 구현 안됨
                "like_count": 0,
                "dislike_count": 0,
                "overall_satisfaction": 0.0,
                "last_updated": datetime.now().isoformat()
            }
        return cls._instance

    def set_session_manager(self, session_manager_instance):
        """SessionManager 인스턴스를 주입받음 (DI)."""
        self._session_ref = session_manager_instance

    async def load_all_feedback_from_db(self):
        """DB에서 모든 피드백을 로드하여 인메모리에 적재합니다."""
        logger.info("DB에서 피드백 데이터 로드 중...")
        try:
            records = await execute_query("SELECT * FROM feedback", fetch_all=True)
            if records:
                self._feedback_data = [dict(record._asdict()) for record in records]  # NamedTuple을 딕셔너리로 변환
                logger.info(f"DB에서 {len(records)}개의 피드백 데이터를 인메모리로 로드했습니다.")
                self._calculate_performance_metrics()
        except Exception as e:
            logger.error(f"DB에서 피드백 로드 실패: {e}", exc_info=True)

    async def add_feedback(self, user_email: str, session_id: str, message_id: str, feedback_type: str, user_query: str,
                           agent_response: str):
        """피드백을 DB에 저장하고 인메모리에 추가합니다."""
        query = """
            INSERT INTO feedback (user_email, session_id, message_id, feedback_type, user_query, agent_response, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        try:
            await execute_query(
                query,
                (user_email, session_id, message_id, feedback_type, user_query, agent_response, datetime.now())
            )
            feedback_entry = {
                "user_email": user_email,
                "session_id": session_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "user_query": user_query,
                "agent_response": agent_response,
                "timestamp": datetime.now().isoformat()
            }
            self._feedback_data.append(feedback_entry)
            self._calculate_performance_metrics()
            logger.info(f"피드백 추가: user={user_email}, session={session_id}, type={feedback_type}, DB 저장 완료.")
        except Exception as e:
            logger.error(f"피드백 DB 저장 실패: {e}", exc_info=True)
            raise RuntimeError(f"피드백 저장 오류: {e}")

    def _calculate_performance_metrics(self):
        like_count = sum(1 for f in self._feedback_data if f["feedback_type"] == "like")
        dislike_count = sum(1 for f in self._feedback_data if f["feedback_type"] == "dislike")
        total_feedback = like_count + dislike_count

        self._performance_metrics["like_count"] = like_count
        self._performance_metrics["dislike_count"] = dislike_count
        if total_feedback > 0:
            self._performance_metrics["overall_satisfaction"] = (like_count - dislike_count) / total_feedback
        else:
            self._performance_metrics["overall_satisfaction"] = 0.0

        if self._session_ref:
            # 모든 세션의 총 메시지 수 합산 (실시간 데이터)
            self._performance_metrics["total_queries"] = sum(
                session.message_count for session in self._session_ref.get_all_active_sessions()
            )
        else:
            self._performance_metrics["total_queries"] = 0

        self._performance_metrics["last_updated"] = datetime.now().isoformat()

    def get_performance_metrics(self) -> Dict[str, Any]:
        self._calculate_performance_metrics()
        return self._performance_metrics

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        return self._feedback_data
