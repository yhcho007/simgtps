#!/bin/bash

echo "Starting 폐쇄망 멀티모달 & 지능형 AI Agent Backend on Linux..."

# 0. 로컬 임베딩 모델을 위한 디렉토리 생성 및 모델 다운로드 안내
echo "Checking local embedding model directory..."
if [ ! -d "../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab" ]; then
    echo "Warning: Local embedding model directory '../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab' not found."
    echo "폐쇄망 환경을 위해 모델 파일을 미리 다운로드하여 해당 경로에 저장해야 합니다."
    echo "인터넷이 되는 환경에서 다음 명령어를 사용하여 모델을 다운로드하세요:"
    echo "  mkdir -p ../backend/app/data/models"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download snunlp/KR-SBERT-V40K --local-dir ../backend/app/data/models/snunlp-SKT-KR-KoBERT-Large-vocab --repo-type model"
    echo "참고: LOCAL_EMBEDDING_MODEL_PATH와 MILVUS_DIM이 모델과 일치하는지 확인하세요."
fi

# 0. ChromaDB 데이터 경로 생성 (리눅스에서 Chroma를 사용하고 싶을 경우)
mkdir -p ../backend/chroma_data

# 1. PostgreSQL DB 및 Milvus Vector DB 실행 (Docker Compose)
echo "Deploying PostgreSQL and Milvus with Docker Compose..."
cd .. # 프로젝트 루트로 이동
# docker-compose.yml 파일에 PostgreSQL 서비스 추가 필요!
# 여기서는 편의상 Milvus만 돌리고, PostgreSQL은 수동으로 가정하거나 별도 docker-compose 파일을 써야 함.
# 여기서는 DB 연결 정보를 .env에서 가져오므로, Postgres 컨테이너가 5432 포트로 떠 있어야 함.
# 예시: 다음 docker-compose.yml 내용 참고
# version: '3.8'
# services:
#   milvus:
#     image: milvusdb/milvus:v2.3.0
#     container_name: milvus-standalone
#     environment:
#       ETCD_ENDPOINTS: etcd:2379
#       MINIO_ADDRESS: minio:9000
#       MILVUS_PORT: 19530
#     ports:
#       - "19530:19530"
#       - "9091:9091"
#     depends_on:
#       - etcd
#       - minio
#     command: milvus-standalone
#   etcd:
#     image: quay.io/coreos/etcd:v3.5.0
#     environment:
#       ETCD_AUTO_COMPACTION_MODE: revision
#       ETCD_AUTO_COMPACTION_RETENTION: 1000
#       ETCD_QUOTA_BACKEND_BYTES: 4294967296
#       ETCD_SNAPSHOT_INTERVAL: 100000
#       ETCD_ELECTION_TIMEOUT: 10000
#       ETCD_LISTEN_CLIENT_URLS: http://0.0.0.0:2379
#       ETCD_ADVERTISE_CLIENT_URLS: http://etcd:2379
#   minio:
#     image: minio/minio:RELEASE.2023-03-20T20-16-04Z
#     environment:
#       MINIO_ACCESS_KEY: minioadmin
#       MINIO_SECRET_KEY: minioadmin
#     ports:
#       - "9000:9000"
#     command: minio --address 0.0.0.0:9000 --console-address 0.0.0.0:9001 S3
#   pgdb: # PostgreSQL 서비스 추가
#     image: postgres:13
#     restart: always
#     environment:
#       POSTGRES_USER: ${POSTGRES_USER}
#       POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
#       POSTGRES_DB: ${POSTGRES_DB}
#     ports:
#       - "5432:5432"
#     volumes:
#       - postgres_data:/var/lib/postgresql/data # 데이터 영구 저장
# volumes:
#   postgres_data: {}
# # 이 내용을 milvus-standalone-docker-compose.yml 에 추가하거나 별도 docker-compose-db.yml 생성.

# (PostgreSQL 및 Milvus를 포함하는) docker compose 파일이 준비되었다고 가정
if [ ! -f "./docker-compose.yml" ]; then # 이전에 milvus-standalone-docker-compose.yml이었던 것을 가정.
    echo "Warning: docker-compose.yml not found. Please create one for Milvus and PostgreSQL."
    # wget 등 다운로드 로직은 제거하고, 사용자가 직접 구성하도록 유도.
fi
docker compose -f ./docker-compose.yml up -d # 통합된 docker compose 파일 실행
sleep 30 # DB 및 Milvus가 완전히 시작될 때까지 충분히 기다립니다.

echo "Verifying service status..."
docker ps -a | grep milvus
docker ps -a | grep pgdb

# PostgreSQL DB 스키마 적용 (컨테이너 내부에서 실행하거나 외부에서 psql로 연결)
echo "Applying PostgreSQL database schema..."
docker exec -i $(docker ps -aqf "name=pgdb") psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} < db_schema.sql
echo "PostgreSQL schema applied."


# 2. Python 가상 환경 설정 및 종속성 설치
echo "Setting up Python virtual environment and installing dependencies..."
cd backend # 다시 backend 디렉토리로 이동
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. FastAPI 애플리케이션 실행
echo "Running FastAPI application..."
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please create one by copying .env.example and fill in necessary environment variables."
    cp .env.example .env
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo "To stop services, go to the project root directory and run: docker compose -f ./docker-compose.yml down"
