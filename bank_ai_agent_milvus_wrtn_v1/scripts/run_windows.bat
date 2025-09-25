@echo off
echo "Starting 폐쇄망 멀티모달 AI Agent Backend on Windows..."

REM  0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
IF NOT EXIST "..\backend\models\snunlp-SKT-KR-KoBERT-Large-vocab" (
    echo "Warning: Local embedding model directory '..\backend\models\snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  md ..\backend\models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ..\backend\models\snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
)

REM  1. Milvus Standalone Docker Desktop으로 실행
echo "Deploying Milvus Vector Database with Docker Desktop..."
echo "Please ensure Docker Desktop is running."
echo "If milvus-standalone-docker-compose.yml is not present, it will be downloaded."
cd %~dp0\..
IF NOT EXIST milvus-standalone-docker-compose.yml (
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml' -OutFile 'milvus-standalone-docker-compose.yml'"
)
docker compose -f milvus-standalone-docker-compose.yml up -d
timeout /t 15 /nobreak > NUL :: Milvus가 완전히 시작될 때까지 충분히 기다립니다. (로컬 모델 로딩 시간 고려)

echo "Verifying Milvus service status..."
docker ps -a | findstr milvus

REM  2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

REM  3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
REM  .env 파일이 없으면 .env.example을 복사하도록 유도
IF NOT EXIST .env (
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    copy .env.example .env
)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

REM  종료 시 Docker Compose 서비스 중단 안내
echo "To stop Milvus services, navigate to the project root and run: docker compose -f milvus-standalone-docker-compose.yml down"
pause