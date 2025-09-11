# main_agent.py
# pip install -U langchain-huggingface

from config import load_config
# main_agent.py

import os
from langchain_huggingface import HuggingFaceEndpoint
from tools.open_banking_tools import get_accounts, get_loans
from tools.financial_tools import analyze_debt_tool, recommend_products_tool

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("⚠️ HUGGINGFACEHUB_API_TOKEN 환경변수를 설정해주세요.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN

# HuggingFace 모델 설정
llm = HuggingFaceEndpoint(
    repo_id="beomi/KcELECTRA-base",
    task="text-generation",
    huggingfacehub_api_token=HF_API_TOKEN,
    temperature=0.3,
    max_new_tokens=256
)

# 사용자 쿼리 처리
async def handle_user_query(user_input: str) -> str:
    """
    HuggingFace 모델을 사용하여 간단히 응답 생성.
    데모에서는 일부 도구(get_accounts 등)도 호출 가능.
    """
    try:
        # 1. Tool 실행 (invoke 사용)
        accounts = get_accounts.invoke({})
        loans = get_loans.invoke({})
        debt = analyze_debt_tool.invoke({"loans": loans})
        recs = recommend_products_tool.invoke({"accounts": accounts})

        # 2. 모델에게 자연어 설명 요청
        context = f"""
        사용자 입력: {user_input}

        📌 보유 계좌: {accounts}
        📌 대출 내역: {loans}
        📌 부채 분석 결과: {debt}
        📌 추천 상품: {recs}
        """

        response = llm.invoke(context)
        return response
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다."

