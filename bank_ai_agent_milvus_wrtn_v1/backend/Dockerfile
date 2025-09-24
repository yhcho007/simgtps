# backend/Dockerfile
# Docker를 사용하여 백엔드 애플리케이션을 배포하려면 이 Dockerfile을 사용합니다.

# Python 3.11 기반 이미지 사용 (Milvus가 Python 3.11 이상 권장하므로)
FROM python:3.11-slim-buster

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (필요한 경우)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 Python 의존성 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 애플리케이션 코드 복사
COPY ./app /app/app

# 환경 변수 설정 (Dockerfile에서 직접 값을 하드코딩하기보다, Docker run 시 -e 옵션 사용 권장)
ENV MILVUS_HOST="milvus-standalone" # Docker 네트워크 내부에서 Milvus 서비스 이름 (docker-compose 시)
ENV MILVUS_PORT="19530"
ENV MILVUS_COLLECTION_NAME="bank_faq_collection"
ENV EMBEDDING_MODEL_NAME="text-embedding-3-small"
# ENV OPENAI_API_KEY="your_openai_api_key_here" # 실제 키는 secrets 관리 툴이나 docker run -e 옵션으로 전달

# 애플리케이션 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]