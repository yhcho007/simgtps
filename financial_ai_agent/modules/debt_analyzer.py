from typing import Dict, Any


def plan_home_saving(financials: Dict[str, Any], target_price: int = 300000000):
    """
    매우 단순화된 내집마련 플랜 생성 로직
    - target_price: 목표 집값 (기본 3억)
    - financials: accounts, savings, loans, income 등
    반환: 연간/월간 납입 계획과 예상 기간
    """
    # 현재 보유 현금 + 저축(원금)
    cash = sum(a.get("balance", 0) for a in financials.get("accounts", []))
    savings = 0
    monthly_savings = 0
    for s in financials.get("savings", []):
        if s.get("type") == "deposit":
            savings += s.get("balance", 0)
        if s.get("type") == "installment":
            monthly_savings += s.get("monthly", 0)
            savings += s.get("monthly", 0) * (s.get("term_months", 0) // 1)

    loans_balance = sum(l.get("balance", 0) for l in financials.get("loans", []))

    # 가용 자금
    current_total = cash + savings

    remaining = max(0, target_price - current_total)

    # 사용자의 월별 저축 여력 (간단히 기존 저축 + 가정 수입 일부)
    # 데모에서는 월 여력 50만 원 가정
    assumed_monthly_capacity = financials.get("assumed_monthly_capacity", 500000)

    # 목표 달성개월수(단순): remaining / monthly_capacity
    if assumed_monthly_capacity <= 0:
        months_needed = None
    else:
        months_needed = (remaining // assumed_monthly_capacity) + (1 if remaining % assumed_monthly_capacity else 0)

    years = None if months_needed is None else months_needed // 12

    plan = {
        "current_total": current_total,
        "loans_balance": loans_balance,
        "remaining_to_target": remaining,
        "monthly_plan": assumed_monthly_capacity,
        "months_needed": months_needed,
        "years_needed": years,
        "recommendation": []
    }

    # 간단한 금융상품 추천 (데모)
    if assumed_monthly_capacity >= 500000:
        plan["recommendation"].append({"product": "저축플랜B", "monthly": 500000, "term_years": 3})
    else:
        plan["recommendation"].append({"product": "스마트예금A", "monthly": assumed_monthly_capacity, "term_years": 2})

    return plan