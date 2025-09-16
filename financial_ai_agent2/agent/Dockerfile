FROM python:3.10-slim
WORKDIR /app
COPY ./agent /app
COPY ./agent/requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
