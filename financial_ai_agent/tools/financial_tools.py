from api_clients.open_banking import get_accounts, get_savings, get_loans, get_subscription_score
from modules.debt_analyzer import plan_home_saving
from langchain.agents import tool

# 부채 분석 도구
@tool
def analyze_debt_tool(loans: list) -> str:
    """
    사용자의 대출 내역을 받아 부채 상환 전략을 제안합니다.
    """
    if not loans:
        return "현재 등록된 대출이 없습니다."

    total_debt = sum(loan.get("balance", 0) for loan in loans)
    return f"총 대출 잔액은 {total_debt:,}원입니다. 우선 고금리 대출부터 상환하는 전략을 추천합니다."

# 금융상품 추천 도구
@tool
def recommend_products_tool(user_profile: dict) -> str:
    """
    사용자의 수입, 저축 가능 금액 등을 입력받아 금융상품을 추천합니다.
    """
    monthly_capacity = user_profile.get("monthly_capacity", 0)
    if monthly_capacity >= 500000:
        return "매월 50만원 이상 저축 가능하므로, '저축플랜B' 상품을 추천합니다."
    else:
        return "소액 저축이 적합하므로, '스마트예금A' 상품을 추천합니다."


def fetch_user_financials():
    accounts = get_accounts().get("accounts", [])
    savings = get_savings().get("savings", [])
    loans = get_loans().get("loans", [])
    subscription = get_subscription_score()

    # 간단한 합치기
    financials = {
        "accounts": accounts,
        "savings": savings,
        "loans": loans,
        "subscription": subscription,
        # 데모용 가정: 사용자의 월 저축 여력 (간단히 계산하거나 하드코딩)
        "assumed_monthly_capacity": 500000
    }
    return financials


def build_home_plan(target_price: int = 300000000):
    financials = fetch_user_financials()
    plan = plan_home_saving(financials, target_price=target_price)
    return {"financials": financials, "plan": plan}


