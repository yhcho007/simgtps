@echo off
echo "Starting 폐쇄망 멀티모달 & 지능형 AI Agent Backend on Windows..."

:: 0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
IF NOT EXIST "..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab" (
    echo "Warning: Local embedding model directory '..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  md ..\backend\app\data\models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ..\backend\app\data\models\snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
)

:: 0. ChromaDB 데이터 경로 생성 (Windows 기본값)
mkdir ..\backend\chroma_data

:: 1. PostgreSQL DB 실행 (Docker Desktop)
echo "Deploying PostgreSQL with Docker Desktop..."
echo "Please ensure Docker Desktop is running."
cd %~dp0\..

:: (PostgreSQL docker-compose.yml이 준비되었다고 가정)
:: 예시: 다음 docker-compose-db.yml 내용 참고
:: version: '3.8'
:: services:
::   pgdb: # PostgreSQL 서비스
::     image: postgres:13
::     restart: always
::     environment:
::       POSTGRES_USER: ${POSTGRES_USER}
::       POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
::       POSTGRES_DB: ${POSTGRES_DB}
::     ports:
::       - "5432:5432"
::     volumes:
::       - postgres_data:/var/lib/postgresql/data # 데이터 영구 저장
:: volumes:
::   postgres_data: {}
:: 이 내용을 docker-compose-db.yml 에 저장

IF NOT EXIST docker-compose-db.yml (
    echo "Warning: docker-compose-db.yml not found. Please create one for PostgreSQL."
)
docker compose -f docker-compose-db.yml up -d
timeout /t 30 /nobreak > NUL :: DB가 완전히 시작될 때까지 충분히 기다립니다.

echo "Verifying PostgreSQL service status..."
docker ps -a | findstr pgdb

:: PostgreSQL DB 스키마 적용 (Windows CMD에서 psql 설치 후 실행하거나 Docker exec)
echo "Applying PostgreSQL database schema..."
docker exec -i $(docker ps -aqf "name=pgdb") psql -U %POSTGRES_USER% -d %POSTGRES_DB% < db_schema.sql
echo "PostgreSQL schema applied."

:: Milvus는 Windows 환경에서는 ChromaDB가 기본이므로 이 부분은 생략
:: 만약 WINDOWS에서 Milvus를 사용하고 싶다면, .env에서 VECTOR_DB=milvus 로 명시하고 별도로 Milvus docker-compose 실행

:: 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt

:: 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
IF NOT EXIST .env (
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    copy .env.example .env
)

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo "To stop PostgreSQL (and Milvus if running), navigate to the project root and run: docker compose -f docker-compose-db.yml down"
pause