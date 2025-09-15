from typing import List, Dict
import math
import numpy as np

tools_py = ''
# 부채 분석 도구

def analyze_debts(debts: List[Dict], income_monthly: float = None, expenses_monthly: float = None):
    # 단순한 스노우볼, 눈덩이(최고이율우선) 비교
    total_balance = sum(d.get('balance', 0) for d in debts)
    total_monthly_payment = sum(d.get('monthly_payment', 0) or 0 for d in debts)
    avg_rate = np.mean([d.get('interest_rate_annual', 0) for d in debts]) if debts else 0

    # 우선순위 - 이자율 기반
    sorted_by_rate = sorted(debts, key=lambda x: (-(x.get('interest_rate_annual') or 0)))
    plan = []
    for d in sorted_by_rate:
        plan.append({
            'creditor': d.get('creditor'),
            'balance': d.get('balance'),
            'interest_rate_annual': d.get('interest_rate_annual'),
            'recommended_extra_payment': 0  # 추후 계산
        })

    # 제안: 여유자금이 있으면 최고이율 부채에 추가 납부 권장
    recommended = {
        'total_balance': total_balance,
        'total_monthly_payment': total_monthly_payment,
        'average_interest_rate': float(avg_rate),
        'priority_list': plan,
        'advice': '기본 제안: 여유자금이 있을 경우 높은 이자율 부채부터 추가 상환하세요.'
    }
    return recommended

# 증식 계획 도구

def propose_growth(balance: float, risk_profile: str = 'moderate', horizon_years: int = 3):
    # 매우 단순화된 자산 배분 예시
    if risk_profile == 'conservative':
        allocation = {'cash': 0.5, 'bonds': 0.4, 'equity': 0.1}
        expected_annual_return = 0.03
    elif risk_profile == 'aggressive':
        allocation = {'cash': 0.05, 'bonds': 0.15, 'equity': 0.8}
        expected_annual_return = 0.08
    else:
        allocation = {'cash': 0.2, 'bonds': 0.4, 'equity': 0.4}
        expected_annual_return = 0.05

    projected = balance * ((1 + expected_annual_return) ** horizon_years)
    return {
        'balance': balance,
        'risk_profile': risk_profile,
        'horizon_years': horizon_years,
        'allocation': allocation,
        'expected_annual_return': expected_annual_return,
        'projected_balance': round(projected, 2),
        'advice': '세부 상품 추천은 고객의 투자성향 및 규제/세금 고려 필요'
    }