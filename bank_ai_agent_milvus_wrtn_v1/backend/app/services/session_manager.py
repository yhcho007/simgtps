# backend/app/services/session_manager.py
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.core.database import execute_query  # DB 쿼리 실행 함수 임포트
import logging

logger = logging.getLogger(__name__)


class ChatSession:
    def __init__(self, session_id: str, user_email: str, start_time: datetime, last_activity: datetime,
                 messages: List[Dict[str, Any]], is_active: bool = True):
        self.session_id = session_id
        self.user_email = user_email
        self.start_time = start_time
        self.last_activity = last_activity
        self.history: List[Dict[str, Any]] = messages  # [{"role": "user", "content": "...", "timestamp": "..."}]
        self.is_active = is_active

    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.history.append(message)
        self.last_activity = datetime.now()

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_email": self.user_email,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.history),
            "is_active": self.is_active
        }

    @classmethod
    def from_record(cls, record: Any):  # NamedTupleCursor의 결과를 받기 위함
        return cls(
            session_id=record.session_id,
            user_email=record.user_email,
            start_time=record.start_time,
            last_activity=record.last_activity,
            messages=json.loads(record.messages) if isinstance(record.messages, str) else record.messages,
            # JSONB는 파이썬에서 딕셔너리로 자동변환될 수도 있음
            is_active=record.is_active
        )


class SessionManager:
    _instance = None
    _sessions: Dict[str, ChatSession] = {}  # In-memory storage for active sessions

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
            cls._sessions = {}  # 초기화
        return cls._instance

    async def load_all_sessions_from_db(self):
        """DB에서 모든 활성 세션을 로드하여 인메모리에 적재합니다."""
        logger.info("DB에서 활성 세션 로드 중...")
        try:
            records = await execute_query("SELECT * FROM chat_sessions WHERE is_active = TRUE", fetch_all=True)
            if records:
                for record in records:
                    session = ChatSession.from_record(record)
                    self._sessions[session.session_id] = session
                logger.info(f"DB에서 {len(records)}개의 활성 세션을 인메모리로 로드했습니다.")
        except Exception as e:
            logger.error(f"DB에서 세션 로드 실패: {e}", exc_info=True)

    async def save_session_to_db(self, session: ChatSession):
        """단일 세션을 DB에 저장하거나 업데이트합니다."""
        query = """
            INSERT INTO chat_sessions (session_id, user_email, start_time, last_activity, messages, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE
            SET user_email = EXCLUDED.user_email,
                last_activity = EXCLUDED.last_activity,
                messages = EXCLUDED.messages,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP;
        """
        try:
            await execute_query(
                query,
                (
                    session.session_id,
                    session.user_email,
                    session.start_time,
                    session.last_activity,
                    json.dumps(session.history),  # JSONB로 저장
                    session.is_active,
                )
            )
            logger.debug(f"세션 {session.session_id} DB에 저장/업데이트 완료.")
        except Exception as e:
            logger.error(f"세션 {session.session_id} DB 저장/업데이트 실패: {e}", exc_info=True)
            raise RuntimeError(f"세션 저장 오류: {e}")

    async def start_or_load_session_async(self, session_id: str, user_email: str) -> ChatSession:
        """
        인메모리에 세션이 없으면 DB에서 로드하거나 새로 생성합니다.
        """
        if session_id not in self._sessions:
            # DB에서 해당 세션 찾기
            record = await execute_query("SELECT * FROM chat_sessions WHERE session_id = %s", (session_id,),
                                         fetch_one=True)
            if record:
                session = ChatSession.from_record(record)
                self._sessions[session_id] = session
                logger.info(f"DB에서 기존 세션 로드: {session_id} by {user_email}")
            else:
                # 새로운 세션 생성
                session = ChatSession(session_id, user_email, datetime.now(), datetime.now(), [])
                self._sessions[session_id] = session
                # 새로운 세션은 즉시 DB에 저장 (초기 레코드 생성)
                await self.save_session_to_db(session)
                logger.info(f"새로운 세션 시작: {session_id} by {user_email}")

        session = self._sessions[session_id]
        session.user_email = user_email  # 세션 재활용 시 사용자 업데이트
        session.last_activity = datetime.now()
        return session

    def add_message_to_history(self, session_id: str, role: str, content: str):
        session = self._sessions.get(session_id)
        if session:
            session.add_message(role, content)
            # 여기서는 DB에 즉시 저장하지 않고, 백그라운드 스케줄러가 저장하도록 위임
        else:
            logger.warning(f"세션 {session_id}를 찾을 수 없어 메시지를 추가할 수 없습니다.")

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_all_active_sessions(self) -> List[ChatSession]:
        # 비활성 세션 정리 로직 (예: 특정 시간 이상 활동 없는 세션)은 백그라운드 태스크나 주기적인 스캔으로 처리
        return list(self._sessions.values())

    async def deactivate_session(self, session_id: str):
        session = self._sessions.get(session_id)
        if session:
            session.is_active = False
            await self.save_session_to_db(session)
            del self._sessions[session_id]
            logger.info(f"세션 {session_id} 비활성화 및 인메모리에서 제거.")
        else:
            logger.warning(f"비활성화하려는 세션 {session_id}가 인메모리에 없습니다.")