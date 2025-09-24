@echo off
echo "Starting AI Agent Backend on Windows..."

:: 1. Milvus Standalone Docker Desktop으로 실행 (최소 Python 3.11 이상 권장) [【8】](https://milvus.io/ko/blog/ai-agents-vs-workflows-why-80-need-simple-automation.md)
echo "Deploying Milvus Vector Database with Docker Desktop..."
echo "Please ensure Docker Desktop is running."
echo "If milvus-standalone-docker-compose.yml is not present, it will be downloaded."
cd %~dp0\..
IF NOT EXIST milvus-standalone-docker-compose.yml (
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml' -OutFile 'milvus-standalone-docker-compose.yml'"
)
docker compose -f milvus-standalone-docker-compose.yml up -d
timeout /t 10 /nobreak > NUL :: Milvus가 완전히 시작될 때까지 기다립니다.

echo "Verifying Milvus service status..."
docker ps -a | findstr milvus

:: 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

:: 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
:: .env 파일이 없으면 .env.example을 복사하도록 유도
IF NOT EXIST .env (
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in OPENAI_API_KEY."
    copy .env.example .env
)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
