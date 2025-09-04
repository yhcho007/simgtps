'''
실행 :
 uvicorn main_gemini:app --host 0.0.0.0 --port 8125 --reload
'''
import os, time, threading
import re
from typing import Dict, Any
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from cryptography.fernet import Fernet
import yaml
import base64

# --- API 사용 한도 설정 ---
# 🚩 Google Gemini 1.5 Flash의 무료 한도에 맞춰 설정
REQUESTS_PER_MINUTE_LIMIT = 15
REQUESTS_PER_DAY_LIMIT = 1500

# --- API 사용량 추적을 위한 전역 변수 ---
request_count_minute = 0
request_count_day = 0
last_minute_reset = time.time()
last_day_reset = time.time()

PASSWORD = "123456"
encryption_key = Fernet(base64.urlsafe_b64encode(PASSWORD.encode().ljust(32)[:32]))

# --- Uvicorn 서버 종료를 위한 전역 변수 ---
# 🚩 reload 기능이 활성화된 경우에만 작동합니다.
server_to_shutdown = None


# --- API 사용량 리셋 함수 ---
def reset_counters():
    """매분, 매일 카운터를 리셋합니다."""
    global request_count_minute, request_count_day, last_minute_reset, last_day_reset

    current_time = time.time()

    # 분당 카운터 리셋
    if current_time - last_minute_reset >= 60:
        print(f"[API_LIMIT]: Minute counter reset. Total requests this minute: {request_count_minute}")
        request_count_minute = 0
        last_minute_reset = current_time

    # 일일 카운터 리셋 (24시간 = 86400초)
    if current_time - last_day_reset >= 86400:
        print(f"[API_LIMIT]: Daily counter reset. Total requests this day: {request_count_day}")
        request_count_day = 0
        last_day_reset = current_time

    # 1초마다 리셋 함수를 다시 호출하여 카운터 관리
    threading.Timer(1, reset_counters).start()


# --- Uvicorn 서버 시작 시 리셋 타이머를 시작합니다. ---
# 🚩 서버가 로드될 때 한 번만 실행되도록 합니다.
if not os.environ.get('RELOADER_RUNNING'):
    os.environ['RELOADER_RUNNING'] = 'true'
    reset_counters()

def get_api_key():
    """Reads, decrypts, and returns the Gemini API key from my.yml."""
    try:
        with open('my.yml', 'r') as file:
            data = yaml.safe_load(file)

        encrypted_tavily_key = data['api_keys']['tavily']
        decrypted_tavily_key = encryption_key.decrypt(encrypted_tavily_key.encode()).decode()

        encrypted_gemini_key = data['api_keys']['gemini']
        decrypted_gemini_key = encryption_key.decrypt(encrypted_gemini_key.encode()).decode()
        return decrypted_tavily_key, decrypted_gemini_key
    except FileNotFoundError:
        raise Exception("my.yml not found. Please run encrypt_key.py first.")
    except Exception as e:
        raise Exception(f"Failed to decrypt API key: {e}")

# --- Simple Cache for Fast Responses ---
SIMPLE_CACHE = {
    "1+1": "1 + 1은 2입니다.",
    "2+2": "2 + 2는 4입니다.",
    "안녕": "안녕하세요! 파이썬 개발을 어떻게 도와드릴까요?",
    "너는 누구야": "저는 전문 파이썬 개발을 돕는 AI 비서입니다.",
    "hello": "Hello! How can I help you with Python development today?",
    "who are you": "I am a professional Python developer assistant."
}


# --- 1. Agent's Tools ---
@tool
def calculator(expression: str) -> str:
    """수학적 표현을 계산합니다 (예: '100 + 2')."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"

decrypted_tavily_key, decrypted_gemini_key = get_api_key()

os.environ["TAVILY_API_KEY"] = decrypted_tavily_key

tavily_search = TavilySearchResults(max_results=3)
tavily_search.name = "web_search"
tavily_search.description = "웹에서 정보를 검색합니다."

tools = [tavily_search]

# --- 2. Language Model (LLM) Setup ---
# 🚩 모델 이름을 gemini-1.5-flash로 변경
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    google_api_key=decrypted_gemini_key
)

# --- 3. Agent Creation ---
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. FastAPI Server ---
app = FastAPI(title="전문 파이썬 AI 에이전트")

MATH_PATTERN = re.compile(r"^\s*(\d+(\s*[-+*/]\s*\d+)+\s*)$")


@app.post("/chat")
async def chat_with_agent(user_input: Dict[str, str]) -> Dict[str, Any]:
    global request_count_minute, request_count_day, server_to_shutdown

    # 🚩 API 한도 확인
    if request_count_minute >= REQUESTS_PER_MINUTE_LIMIT:
        print("[API_LIMIT]: Minute limit exceeded. Shutting down server.")
        if server_to_shutdown:
            server_to_shutdown.should_exit = True
        return {"response": "API 분당 요청 한도를 초과하여 서버가 종료됩니다. 잠시 후 다시 시도해주세요."}

    if request_count_day >= REQUESTS_PER_DAY_LIMIT:
        print("[API_LIMIT]: Daily limit exceeded. Shutting down server.")
        if server_to_shutdown:
            server_to_shutdown.should_exit = True
        return {"response": "API 일일 요청 한도를 초과하여 서버가 종료됩니다. 내일 다시 시도해주세요."}

    # 🚩 API 호출 직전에 카운터 증가
    request_count_minute += 1
    request_count_day += 1

    query = user_input.get("message", "")
    if not query:
        return {"response": "메시지를 입력해주세요."}

    # 1. 캐시 확인
    normalized_query = query.lower().replace(' ', '')
    if normalized_query in SIMPLE_CACHE:
        return {"response": SIMPLE_CACHE[normalized_query]}

    # 2. 수학 연산 사전 처리
    if MATH_PATTERN.match(normalized_query):
        try:
            result = eval(normalized_query)
            return {"response": f"계산 결과는 {result}입니다."}
        except:
            pass

    try:
        # 3. 에이전트 실행
        response = await agent_executor.ainvoke({"input": query})
        return {"response": response.get("output", "응답을 생성하지 못했습니다.")}
    except Exception as e:
        print(f"오류: {e}")
        return {"response": "요청 처리 중 오류가 발생했습니다. 다시 시도해주세요."}


# --- Uvicorn 서버 시작 시 서버 객체를 전역 변수에 저장 ---
if __name__ == "__main__":
    import uvicorn

    # 서버 객체를 직접 생성하여 should_exit 플래그를 제어할 수 있게 합니다.
    config = uvicorn.Config("main_gemini:app", host="127.0.0.1", port=8125, reload=True)
    server_to_shutdown = uvicorn.Server(config=config)
    server_to_shutdown.run()

