# backend/app/services/background_tasks.py
import asyncio
from datetime import datetime, timedelta
from app.services.stats_manager import StatsManager
from app.services.session_manager import SessionManager, ChatSession
from app.core.embeddings import EmbeddingModel
from app.core.database import execute_query  # DB 연동
import logging

logger = logging.getLogger(__name__)


# 임의의 백그라운드 모델 (예: 파인튜닝 대상 모델)
class DummyFineTuneModel:
    def __init__(self):
        self.accuracy = 0.70
        self.last_trained = None
        logger.info("더미 파인튜닝 모델 초기화")

    async def fine_tune(self, feedback_data: List[Dict[str, Any]]):
        logger.info(f"백그라운드에서 모델 파인튜닝 시작... (피드백 데이터 {len(feedback_data)}개 활용)")
        # --- 실제 모델 파인튜닝 로직 (수분~수시간 소요 가능) ---
        # 1. feedback_data를 이용하여 학습 데이터셋 생성
        #    예: feedback_data 중 'dislike' 피드백이 있는 경우, 해당 쿼리와 응답 쌍을
        #    잘못된 답변으로 분류하고, 'like' 피드백은 좋은 답변으로 분류하여
        #    새로운 학습 데이터셋을 만듦.
        # 2. EmbeddingModel (또는 LLM)의 특정 레이어를 fine-tuning
        #    (폐쇄망 LLM의 경우: 로컬 LLM API에 파인튜닝 요청을 보내거나,
        #     로컬 서버에서 직접 Fine-tuning 모델을 로드하여 재학습)
        # 3. 새로운 모델 가중치 저장 및 로드 (또는 Fine-tuned 모델 API 업데이트)

        await asyncio.sleep(10)  # 파인튜닝 작업 시뮬레이션 (10초 소요)

        # 모델 개선 시뮬레이션
        if feedback_data:  # 피드백이 있는 경우만
            like_count = sum(1 for f in feedback_data if f["feedback_type"] == "like")
            dislike_count = sum(1 for f in feedback_data if f["feedback_type"] == "dislike")
            total_feedback = like_count + dislike_count
            if total_feedback > 0:
                # 긍정 피드백이 많을수록 정확도 상승
                self.accuracy += (0.01 * (like_count / total_feedback))
                # 부정 피드백이 많으면 정확도 하락 또는 재검토 필요 시뮬레이션
                if dislike_count > like_count:
                    self.accuracy -= 0.005  # 부정 피드백이 많으면 소폭 하락
            if self.accuracy > 0.95: self.accuracy = 0.95  # 상한선
            if self.accuracy < 0.60: self.accuracy = 0.60  # 하한선

        self.last_trained = datetime.now()
        logger.info(f"모델 파인튜닝 완료! 새로운 정확도: {self.accuracy:.2f}")


dummy_fine_tune_model = DummyFineTuneModel()


async def start_background_fine_tuning(stats_manager: StatsManager):
    """주기적으로 피드백을 모아 모델을 파인튜닝하는 스케줄러."""
    while True:
        await asyncio.sleep(60 * 60 * 12)  # 12시간마다 파인튜닝 시도 (배포 시 설정 조절)
        logger.info("백그라운드 파인튜닝 작업 스케줄러 실행.")
        feedback_data = stats_manager.get_all_feedback()
        if feedback_data:
            await dummy_fine_tune_model.fine_tune(feedback_data)
        else:
            logger.info("파인튜닝할 피드백 데이터가 없습니다.")


async def start_session_db_saver(session_manager: SessionManager):
    """주기적으로 변경된 세션 데이터를 DB에 저장하는 스케줄러."""
    while True:
        await asyncio.sleep(60)  # 60초마다 세션 데이터를 DB에 저장
        logger.debug("백그라운드 세션 DB 저장 스케줄러 실행.")
        active_sessions = session_manager.get_all_active_sessions()
        for session in active_sessions:
            try:
                # 일정 시간 이상 활동이 없으면 비활성화 처리
                if (datetime.now() - session.last_activity) > timedelta(minutes=60):  # 60분
                    session.is_active = False
                    logger.info(f"세션 {session.session_id}가 비활성 상태로 전환됩니다.")
                await session_manager.save_session_to_db(session)
                if not session.is_active:
                    session_manager._sessions.pop(session.session_id, None)  # 인메모리에서 제거
            except Exception as e:
                logger.error(f"백그라운드에서 세션 {session.session_id} 저장 실패: {e}", exc_info=True)
