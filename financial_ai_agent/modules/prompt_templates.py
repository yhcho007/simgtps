def debt_prompt(user_question, analysis, product_suggest):
    return f"""
사용자 질문: {user_question}

[분석 요약]
총 부채: {analysis['total_debt']}
상환 플랜:
{analysis['plan']}

[추천 금융상품]
{product_suggest}

친절하고 정중한 한국어로 상환 계획과 상품을 제안해주세요.
"""