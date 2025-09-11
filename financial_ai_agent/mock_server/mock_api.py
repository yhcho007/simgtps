'''
financial_ai_agent 에서
 uvicorn mock_server.mock_api:app --reload --port 8001
'''
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 간단한 Mock 응답
@app.get("/accounts")
def get_accounts():
    # 사용자의 계좌 목록과 잔액
    return {
        "accounts": [
            {"bank": "국민은행", "account_no": "111-222", "balance": 3500000},
            {"bank": "카카오뱅크", "account_no": "222-333", "balance": 1200000}
        ]
    }

@app.get("/savings")
def get_savings():
    return {
        "savings": [
            {"name": "스마트예금A", "type": "deposit", "balance": 2000000},
            {"name": "저축플랜B", "type": "installment", "monthly": 500000, "term_months": 12}
        ]
    }

@app.get("/loans")
def get_loans():
    return {"loans": [{"name": "학자금대출", "balance": 8000000, "rate": 0.055}]}

@app.get("/subscription_score")
def get_subscription_score():
    # 청약통장 점수(모의값)
    return {"score": 62, "eligibility": True}