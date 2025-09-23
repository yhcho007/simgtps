# Bank AI Agent — Full Production-ready Demo (v2)

이 저장소는 교육용 '프로덕션 수준' 데모를 제공합니다.
주요 기능:
- FastAPI backend: JWT 인증, mTLS/Vault 가이드, Prometheus metrics, interaction logging (SQLite)
- RAG: Sentence-Transformers + FAISS 인덱스 생성 스크립트 포함
- Model proxy: 로컬 모델서버 또는 외부 모델(예: OpenAI) 선택 가능
- User feedback 수집: 1-10 만족도 점수(1-3이면 이유 필수) 및 로그 저장
- Daily report: 일일 성능 리포트를 이메일로 전송하는 스크립트 포함
- Streamlit dashboard: 실시간/과거 통계 시각화
- Frontend: React chat UI (로그인, WebSocket, 만족도 평가 UI, 피드백 유도)
- Infra: k8s/helm 예제, mTLS/Cert-manager 가이드, Vault example 스크립트

---

빠른 실행 (개발 환경)
1) Backend 준비
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   # 개발용 DB 초기화
   python -m app.db_init

2) 모델서버(옵션)
   cd model_server
   python model_server.py --model_name gpt2 --port 9000

3) 백엔드 실행
   export SECRET_KEY='dev-secret-key'
   export MODEL_PROVIDER='local'  # or 'openai'
   export MODEL_SERVER_URL='http://localhost:9000/generate'
   uvicorn app.main:app --reload --port 8000

4) FAISS 인덱스 생성 (RAG)
   cd backend
   python scripts/build_faiss.py

5) Frontend 실행
   cd frontend
   npm install
   npm start
   브라우저: http://localhost:3000

6) Streamlit Dashboard (옵션)
   cd streamlit_app
   pip install -r requirements.txt
   streamlit run streamlit_app.py --server.port 8501

7) Daily report: 수동 실행 또는 cron에 등록
   cd backend
   python scripts/send_daily_report.py

---
모든 주요 파일에 주석을 크게 달아두었습니다. 프로덕션 적용 전 필수 보안 조치 (mTLS, Vault, TLS 전체 적용, 모니터링 강화 등)를 반드시 구현하세요.
