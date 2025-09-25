#!/bin/bash

echo "Starting 폐쇄망 멀티모달 AI Agent Backend on Linux..."

# 0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
if [ ! -d "../backend/models/snunlp-SKT-KR-KoBERT-Large-vocab" ]; then
    echo "Warning: Local embedding model directory '../backend/models/snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  mkdir -p ../backend/models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ../backend/models/snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
fi

# 1. Milvus Standalone Docker Compose로 실행
echo "Deploying Milvus Vector Database with Docker Compose..."
# 밀버스 설정 파일 다운로드 (없을 경우)
if [ ! -f "../milvus-standalone-docker-compose.yml" ]; then
    echo "Downloading Milvus Docker Compose configuration..."
    wget https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml -O ../milvus-standalone-docker-compose.yml
fi
cd .. # 프로젝트 루트로 이동
docker compose -f milvus-standalone-docker-compose.yml up -d
sleep 15 # Milvus가 완전히 시작될 때까지 충분히 기다립니다. (로컬 모델 로딩 시간 고려)

echo "Verifying Milvus service status..."
docker ps -a | grep milvus

# 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
# .env 파일이 없으면 .env.example을 복사하도록 유도
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    cp .env.example .env
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 종료 시 Docker Compose 서비스 중단
echo "To stop Milvus services, go to the project root directory and run: docker compose -f milvus-standalone-docker-compose.yml down"
