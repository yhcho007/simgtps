FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir transformers torch uvicorn fastapi
EXPOSE 9000
CMD ["python", "model_server.py"]
