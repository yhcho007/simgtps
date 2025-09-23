"""AgentService: thin wrapper that delegates to orchestrator and records interactions.
- Records each interaction into SQLite via app.db
- Exposes record_feedback helper used by feedback endpoint
"""
from agent.orchestrator import BankAgentOrchestrator
from app.db import SessionLocal, Interaction
import time

class AgentService:
    def __init__(self):
        self.orch = BankAgentOrchestrator()

    def handle_user_message(self, session_id: str, text: str, user_info: dict):
        start = time.time()
        # call orchestrator to get reply
        result = self.orch.run(session_id=session_id, user=user_info, query=text)
        elapsed = (time.time() - start) * 1000.0
        # save to DB
        db = SessionLocal()
        try:
            it = Interaction(session_id=session_id, user=user_info.get('username', user_info.get('id','anon')),
                             query=text, reply=result.get('reply'), response_time_ms=elapsed, success=1)
            db.add(it)
            db.commit()
            db.refresh(it)
            # include message id so frontend can reference it for feedback
            result['msg_id'] = it.id
        finally:
            db.close()
        return result

    def record_feedback(self, session_id: str, msg_id: int, user: dict, rating: int, reason: str):
        db = SessionLocal()
        try:
            it = db.query(Interaction).filter(Interaction.id == msg_id).first()
            if it:
                it.satisfaction = rating
                it.feedback = reason
                db.commit()
        finally:
            db.close()
