'''
main.py 기동
uvicorn main:app --reload

조언 요청 테스트:
curl -X POST http://127.0.0.1:8123/chat -H "Content-Type: application/json" -d '{"message": "이번 주말에 뭐 하면 좋을까?"}'

스케줄 추가 요청 테스트:
curl -X POST http://127.0.0.1:8123/chat -H "Content-Type: application/json" -d '{"message": "다음 주 월요일에 점심 약속을 스케줄에 추가해줘."}'

'''
import os
from typing import Dict, Any

from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from cryptography.fernet import Fernet
import yaml
import base64

# --- API Key Decryption ---
# We use the same password as in the encryption script.
# Again, for production, store this password securely (e.g., in a secret vault or environment variable).
PASSWORD = "123456"
encryption_key = Fernet(base64.urlsafe_b64encode(PASSWORD.encode().ljust(32)[:32]))

def get_gemini_api_key():
    """Reads, decrypts, and returns the Gemini API key from my.yml."""
    try:
        with open('my.yml', 'r') as file:
            data = yaml.safe_load(file)

        encrypted_key = data['api_keys']['gemini']
        decrypted_key = encryption_key.decrypt(encrypted_key.encode()).decode()
        return decrypted_key
    except FileNotFoundError:
        raise Exception("my.yml not found. Please run encrypt_key.py first.")
    except Exception as e:
        raise Exception(f"Failed to decrypt API key: {e}")

# Get the decrypted API key
gemini_api_key = get_gemini_api_key()
os.environ['GEMINI_API_KEY'] = gemini_api_key # Set the environment variable for LangChain to use

# --- 1. AI 비서가 사용할 도구(Tool) 정의 ---
@tool
def create_schedule(date: str, task: str) -> str:
    """
    사용자의 요청에 따라 특정 날짜에 스케줄을 추가합니다.
    Args:
        date: 스케줄을 추가할 날짜 (예: '내일', '2025-01-01').
        task: 스케줄 내용.
    Returns:
        스케줄이 성공적으로 추가되었음을 알리는 메시지.
    """
    print(f"[알림]: '{date}'에 '{task}' 스케줄이 추가되었습니다.")
    return f"'{date}'에 '{task}' 스케줄을 추가했습니다."

tools = [create_schedule]

# --- 2. 언어 모델(LLM)과 프롬프트 설정 ---
# Google Gemini 모델을 초기화합니다.
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

system_prompt = (
    "당신은 사용자에게 조언을 해주거나 스케줄을 관리해주는 친절한 AI 비서입니다."
    "사용자의 요청에 따라 적절한 답변을 하거나 도구를 사용하세요."
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- 3. 에이전트 생성 및 실행기 설정 ---
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. FastAPI 서버 설정 ---
app = FastAPI(title="내 개인 AI 비서")

@app.post("/chat")
async def chat_with_assistant(user_input: Dict[str, str]) -> Dict[str, Any]:
    query = user_input.get("message", "")
    if not query:
        return {"response": "메시지를 입력해주세요."}

    try:
        response = await agent_executor.ainvoke({"input": query})
        return {"response": response.get("output", "응답을 생성하지 못했습니다.")}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"response": "요청 처리 중 오류가 발생했습니다. 다시 시도해주세요."}

# --- 5. 서버 실행 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8123, reload=True)