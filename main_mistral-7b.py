'''

실행:
uvicorn main_mistral-7b:app --reload
uvicorn main_mistral-7b:app --host 0.0.0.0 --port 8125 --reload
'''
from typing import Dict, Any

from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_react_agent #create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.llms import LlamaCpp # GGUF 모델을 위한 라이브러리
from dotenv import load_dotenv
from langchain import hub


# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()

# --- 1. AI 비서가 사용할 도구(Tool) 정의 ---
# 이 부분은 이전 코드와 동일합니다.
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

# --- 2. 언어 모델(LLM)과 프롬프트 설정 (Mistral-7B 사용) ---
# Hugging Face Hub에서 "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"와 같은
# GGUF 형식 파일을 다운로드하여 경로를 지정합니다.
# Model file is in the same directory as main.py
model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# LlamaCpp 모델을 초기화합니다.
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=0,  # GPU를 사용하지 않음
    n_ctx=4096,      # 컨텍스트 윈도우 크기 (메모리 사용량에 따라 조정 가능)
    verbose=True,
    temperature=0.7
)


# 에이전트의 역할을 정의하는 시스템 프롬프트를 작성합니다.
system_prompt = (
    "당신은 사용자에게 코딩 조언, 스케줄 관리 등 다양한 도움을 주는 친절한 AI 비서입니다."
    "사용자의 요청에 따라 적절한 답변을 하거나 도구를 사용하세요."
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- 3. 에이전트 생성 및 실행기 설정 (React Agent 사용) ---
# Hub에서 ReAct 프롬프트를 가져옵니다.
# ReAct 프롬프트는 LLM이 행동(Action)과 관찰(Observation)을 통해 추론하도록 도와줍니다.
prompt = hub.pull("hwchase17/react")

# 이제 create_tool_calling_agent 대신 create_react_agent를 사용합니다.
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. FastAPI 서버 설정 ---
# 이 부분은 이전 코드와 동일합니다.
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
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)