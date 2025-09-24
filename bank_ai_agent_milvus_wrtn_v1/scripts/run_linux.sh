#!/bin/bash

echo "Starting AI Agent Backend on Linux..."

# 1. Milvus Standalone Docker Compose로 실행 (최소 Python 3.11 이상 권장) [【8】](https://milvus.io/ko/blog/ai-agents-vs-workflows-why-80-need-simple-automation.md)
echo "Deploying Milvus Vector Database with Docker Compose..."
# 밀버스 설정 파일 다운로드 (없을 경우)
if [ ! -f "milvus-standalone-docker-compose.yml" ]; then
    wget https://github.com/milvus-io/milvus/releases/download/v2.5.12/milvus-standalone-docker-compose.yml -O milvus-standalone-docker-compose.yml
fi
# Docker Compose 서비스 시작
docker compose -f milvus-standalone-docker-compose.yml up -d
sleep 10 # Milvus가 완전히 시작될 때까지 기다립니다.

echo "Verifying Milvus service status..."
docker ps -a | grep milvus

