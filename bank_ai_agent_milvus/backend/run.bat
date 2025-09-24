setx /m SECRET_KEY "dev-secret-key"
REM setx /m MODEL_PROVIDER "local"  # or "openai"
setx /m MODEL_PROVIDER "local"
setx /m MODEL_SERVER_URL "http://localhost:9000/generate"
..\.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
deactivate