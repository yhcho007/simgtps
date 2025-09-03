'''
usage :
uvicorn main_codegemma:app --host 0.0.0.0 --port 8125 --reload
'''
from typing import Dict, Any

from fastapi import FastAPI
from langchain.agents import AgentExecutor
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
#from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent
from langchain import hub


# --- 1. Agent's Tools ---
@tool
def get_python_documentation(query: str) -> str:
    """Fetches key information about a Python topic or function from documentation.
    Args:
        query: The Python topic or function to look up.
    Returns:
        Relevant information from Python's documentation.
    """
    # This is a placeholder. A real tool would use an API or web scraper.
    return f"Searching Python documentation for: {query}"


tools = [get_python_documentation]

# --- 2. Language Model (LLM) Setup ---
# Set the path to your downloaded CodeGemma GGUF file.
model_path = "./gemma-2-2b-it.F16.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=0,  # Use CPU
    n_ctx=4096,
    verbose=True,
    temperature=0.7
)

# --- 3. Agent Creation ---
# Pull the ReAct prompt template from LangChain Hub.
# ReAct is a proven method for making agents that can reason and use tools.
prompt = hub.pull("hwchase17/react")

# Create a React agent for tool use and reasoning.
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 4. FastAPI Server ---
app = FastAPI(title="Professional Python Agent")


@app.post("/chat")
async def chat_with_agent(user_input: Dict[str, str]) -> Dict[str, Any]:
    query = user_input.get("message", "")
    if not query:
        return {"response": "Please provide a message."}

    try:
        response = await agent_executor.ainvoke({"input": query})
        return {"response": response.get("output", "Could not generate a response.")}
    except Exception as e:
        print(f"Error: {e}")
        return {"response": "An error occurred while processing your request. Please try again."}


if __name__ == "__main__":
    import uvicorn

    # You may need to change the port if 8125 is in use.
    uvicorn.run(app, host="127.0.0.1", port=8125, reload=True)