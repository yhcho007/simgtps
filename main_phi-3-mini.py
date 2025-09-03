import os
from typing import Dict, Any

from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv

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

# --- 2. 언어 모델(LLM)과 프롬프트 설정 (Phi-3-mini 사용) ---
# Phi-3-mini 모델과 토크나이저를 로드합니다. (최초 실행 시 다운로드)
model_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 텍스트 생성 파이프라인을 설정합니다.
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device="cpu"  # GPU가 아닌 CPU를 명시적으로 사용
)

# Hugging Face 파이프라인을 LangChain LLM으로 변환
llm = HuggingFacePipeline(pipeline=pipe)

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

# --- 3. 에이전트 생성 및 실행기 설정 ---
# 이 부분은 이전 코드와 동일합니다.
agent = create_tool_calling_agent(llm, tools, prompt)
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
    uvicorn.run(app, host="127.0.0.1", port=8124, reload=True)