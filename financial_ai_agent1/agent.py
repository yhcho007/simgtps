from models import LLM, Embedder, VectorStore
from tools import analyze_debts, propose_growth
import uuid

agent_py = ''
class FinancialAgent:
    def __init__(self):
        # 경량화 설정: OpenAI 사용 권장
        self.llm = LLM(model_name='gpt-4o-mini')
        self.embedder = Embedder()
        self.vs = VectorStore()

    def _save_customer_memory(self, customer_id: str, text: str):
        emb = self.embedder.encode([text])[0]
        doc_id = f"{customer_id}-{uuid.uuid4()}"
        self.vs.add(ids=[doc_id], metadatas=[{"customer_id": customer_id}], embeddings=[emb], documents=[text])

    def handle_debt_analysis(self, payload: dict):
        # 입력 데이터 수집
        customer_id = payload.get('customer_id')
        debts = payload.get('debts', [])
        income = payload.get('income_monthly')
        expenses = payload.get('expenses_monthly')

        # 도메인 계산기 호출
        calc = analyze_debts(debts, income, expenses)

        # LLM에게 자연어 설명 생성 요청 (한국어)
        prompt = f"고객ID:{customer_id}\n요약: 총부채 {calc['total_balance']}원, 월납 {calc['total_monthly_payment']}원, 평균이율 {calc['average_interest_rate']}. 우선순위: {[(p['creditor'], p['balance']) for p in calc['priority_list']]}\n\n위 정보를 바탕으로 한국어로 친절하고 실무적인 부채 상환 계획과 실행 가능한 단계(월별 체크리스트 포함)를 작성해줘."
        text = self.llm.generate(prompt)

        # 메모리 저장
        self._save_customer_memory(customer_id, prompt + "\n" + text)

        return {'analysis': calc, 'plan_narrative': text}

    def handle_growth_proposal(self, payload: dict):
        customer_id = payload.get('customer_id')
        balance = payload.get('balance', 0)
        risk = payload.get('risk_profile', 'moderate')
        horizon = payload.get('horizon_years', 3)

        calc = propose_growth(balance, risk, horizon)
        prompt = f"고객ID:{customer_id}\n요약: 잔액 {balance}원, 위험성향 {risk}, 기간 {horizon}년.\n\n한국어로 친절하게 자산배분과 단계별 행동계획(월 단위)을 작성하고, 세부상품군(예: 예적금, 국채, 국내 ETF, 해외 ETF, P2P 등)별 장단점을 간단히 설명해줘."
        text = self.llm.generate(prompt)
        self._save_customer_memory(customer_id, prompt + "\n" + text)
        return {'proposal': calc, 'proposal_narrative': text}
