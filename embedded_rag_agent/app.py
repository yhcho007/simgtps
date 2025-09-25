from fastapi import FastAPI, UploadFile
from agent import ChatAgent

app = FastAPI()

agent = ChatAgent()

@app.post('/chat')
async def chat(message: str):
    return {"response": agent.respond(message)}
