'''
financial_ai_agent 에서
  uvicorn app:app --reload --port 8000

'''
from fastapi import FastAPI, Request
from main_agent import handle_user_query

app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_input = body.get("input", "")
    result_text = await handle_user_query(user_input)
    return {"output": result_text}
