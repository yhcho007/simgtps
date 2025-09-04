import os
import json
from typing import Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import LlamaCpp
from langchain_core.tools import tool
from langchain import hub
from cryptography.fernet import Fernet
import yaml
import base64

PASSWORD = "123456"
encryption_key = Fernet(base64.urlsafe_b64encode(PASSWORD.encode().ljust(32)[:32]))

def get_tavily_api_key():
    """Reads, decrypts, and returns the Gemini API key from my.yml."""
    try:
        with open('my.yml', 'r') as file:
            data = yaml.safe_load(file)

        encrypted_key = data['api_keys']['tavily']
        decrypted_key = encryption_key.decrypt(encrypted_key.encode()).decode()
        return decrypted_key
    except FileNotFoundError:
        raise Exception("my.yml not found. Please run encrypt_key.py first.")
    except Exception as e:
        raise Exception(f"Failed to decrypt API key: {e}")


tavily_api_key = get_tavily_api_key()
os.environ["TAVILY_API_KEY"] = tavily_api_key
# Tavily API를 사용하여 실제 웹 검색 도구를 만듭니다.
# max_results는 검색 결과의 개수를 제한합니다.
tavily_search = TavilySearchResults(max_results=3)
tavily_search.name = "web_search"
tavily_search.description = "Searches the web for information."

# --- Simple Cache for Fast Responses ---
# Add your simple questions and answers here.
SIMPLE_CACHE = {
    "1+1": "1 + 1 is 2.",
    "2+2": "2 + 2 is 4.",
    "hello": "Hello! How can I help you with Python development today?",
    "who are you": "I am a professional Python developer assistant.",
    "안녕": "안녕하세요. 무엇을 도와드릴까요?",
}

# --- 1. Agent's Tools ---
@tool
def web_search(query: str) -> str:
    """Searches the web for information about a given query.
    This is a placeholder for a real web search tool (e.g., TavilySearch, SerpAPI).

    Args:
        query: The search query string.
    Returns:
        A string containing a summary of the search results.
    """
    print(f"\n[Tool]: Performing web search for: '{query}'")
    # Simulate a web search result based on a simple query
    if "python" in query.lower() and "file transfer" in query.lower():
        return "Python file transfer can be done using the 'socket' library for a simple client-server model or using high-level libraries like 'paramiko' for SFTP."
    return "I couldn't find a direct answer. Please provide more context."

# 에이전트가 사용할 도구 목록
tools = [tavily_search]

# --- 2. Language Model (LLM) Setup ---
# Set the path to your downloaded CodeGemma GGUF file.
model_path = "./gemma-2-2b-it.F16.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=0,
    n_ctx=4096,
    verbose=True,
    temperature=0.7
)

# --- 3. Agent Creation ---
# Pull the ReAct prompt template from LangChain Hub.
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. FastAPI Server ---
app = FastAPI(title="Professional Python Agent")

@app.post("/chat")
async def chat_with_agent(user_input: Dict[str, str]) -> Dict[str, Any]:
    query = user_input.get("message", "")
    if not query:
        return {"response": "Please provide a message."}

    # --- Check Cache First ---
    # Convert query to lowercase to handle case-insensitivity
    normalized_query = query.lower().strip()
    if normalized_query in SIMPLE_CACHE:
        return {"response": SIMPLE_CACHE[normalized_query]}

    try:
        # If not in cache, pass to the AI agent
        response = await agent_executor.ainvoke({"input": query})
        return {"response": response.get("output", "Could not generate a response.")}
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "An error occurred while processing your request. Please try again."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8125, reload=True)