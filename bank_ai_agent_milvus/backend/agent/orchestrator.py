"""BankAgentOrchestrator implements:
1) RAG: searches FAISS vectorstore for relevant docs
2) Constructs a safe prompt and sends to model proxy (local server or vendor)
3) Post-processes reply via output_filter and PII masking
4) Detects actions (simple rules) and calls BankAPIAdapter when needed

This is a simplified flow intended to be extended for production.
"""
import os
import httpx
# from app.vectorstore.faiss_store import FaissVectorStore
from app.vectorstore.milvus_store import MilvusClient
from app.services.bank_api_adapter import BankAPIAdapter
from app.models.output_filter import output_filter, mask_pii

class BankAgentOrchestrator:
    def __init__(self, model_server_url=None):
        # Load vectorstore (FAISS) and bank adapter
        #self.store = FaissVectorStore(persist_path=os.path.join(os.getcwd(), 'data', 'faiss_store'))
        self.store = MilvusClient()
        self.bank_adapter = BankAPIAdapter()
        # Model server URL or vendor selection via environment
        self.model_server_url = model_server_url or os.getenv('MODEL_SERVER_URL', 'http://localhost:9000/generate')
        self.model_provider = os.getenv('MODEL_PROVIDER', 'local')  # 'local' or 'openai'

    def run(self, session_id, user, query):
        # 1) RAG search for context
        docs = self.store.search(query, k=5)
        kb_text = '\n'.join([d.get('text', '') for d in docs])

        # 2) Build prompt template with clear system instructions for safety
        prompt = f"""
System: 당신은 은행 고객 상담 AI입니다. 아래 규칙을 반드시 따르세요:
- 고객의 PII(계좌번호, 주민번호 등)를 외부로 노출하지 마세요.
- 내부절차와 규정을 벗어나는 권한 행사는 할 수 없습니다.
- 필요한 경우 내부 API를 호출하여 정보를 확인하세요.

User Question: {query}

Context documents:\n{kb_text}

Please answer concisely in Korean.
"""

        # 3) Send prompt to model proxy (HTTP)
        payload = {'prompt': prompt, 'max_tokens': 256}
        try:
            r = httpx.post(self.model_server_url, json=payload, timeout=15.0)
            r.raise_for_status()
            generated = r.json().get('text', '').strip()
        except Exception:
            generated = '모델 응답을 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요.'

        # 4) Post-process output
        filtered = output_filter(generated)
        masked = mask_pii(filtered)

        # 5) Simple action detection: if user asks for balance
        if '잔액' in query or '얼마' in query:
            account_id = user.get('account_id', '111-222-333')
            api_res = self.bank_adapter.call('get_balance', {'account_id': account_id}, user)
            reply = f"계좌 {account_id}의 잔액은 {api_res['balance']}원입니다. 최근 거래: {api_res['recent']}"
            reply = mask_pii(reply)
            return {'reply': reply}

        return {'reply': masked}
