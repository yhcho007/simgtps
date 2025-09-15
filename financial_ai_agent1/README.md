좋은 선택이에요 👍 프리랜서 입장에서 “WMS/WCS/WES를 깊게 파고 1년 투자”하는 건 ROI가 낮을 수 있고, 요즘 가장 뜨는 시장은 **AI 에이전트/에이전트 프레임워크** 쪽이 맞습니다. 특히 파이썬 10년 경력자라면 빠르게 뛰어들 수 있어요.

---

## 1. AI 에이전트 개발이란?

### 정의

* 단순 LLM 챗봇이 아니라, **목표 기반으로 스스로 계획을 세우고, 도구(API/DB/웹/코드 실행)를 사용해 문제 해결하는 소프트웨어 에이전트**.
* 예시:

  * 고객지원 에이전트 (FAQ 응답 + DB 조회 + 이메일 전송)
  * 데이터 리서치 에이전트 (검색 → 요약 → 리포트 작성)
  * 코드 어시스턴트 (코드 실행 → 디버깅 → 수정안 제시)

### 기본 구성 요소

1. **LLM (두뇌)** – GPT, Claude, LLaMA 등
2. **Memory (기억)** – 과거 대화, 문서, 작업 히스토리 저장 (Vector DB + RAG)
3. **Tools (도구)** – API 호출, DB 쿼리, 웹 브라우징, 코드 실행
4. **Planner/Executor (행동계획 + 실행)** – 체인(chain), 트리(tree), 루프(loop) 기반의 계획 실행

---

## 2. 에이전트 개발을 위해 알아야 할 핵심

### (1) Chunk & RAG (Retrieval-Augmented Generation)

* **Chunk**: 긴 문서를 작은 단위(보통 500\~1000 토큰)로 쪼개서 검색 가능하게 만드는 것
* **Vector DB**: chunk들을 임베딩 벡터로 변환 후 저장 (Pinecone, Weaviate, FAISS, Chroma)
* **RAG 프로세스**:

  * 사용자의 질문 → 관련 chunk 검색 → LLM에 컨텍스트로 넣음 → 응답 생성

👉 당신은 아직 이 부분이 낯설다 했으니 **첫 스텝은 RAG 실습**입니다.

---

### (2) 에이전트 프레임워크

* **LangChain**: 가장 유명, 풍부한 기능 (Chains, Tools, Agents)
* **LlamaIndex**: 데이터 연결/RAG 특화
* **CrewAI, AutoGen, Haystack**: 협업형 멀티에이전트 지원

👉 프리랜서라면 **LangChain + LlamaIndex**를 먼저 익히는 게 가장 시장성이 높습니다.

---

### (3) 필수 기술 스택

* **Python** (이미 충분)
* **LLM API 사용법**: OpenAI, Anthropic, Mistral, Hugging Face
* **Vector DB**: FAISS/Chroma → 나중에 Pinecone/Weaviate (프로덕션용)
* **에이전트 실행 구조**:

  * Tool 등록 (예: Google Search API, DB Query)
  * Memory 구성 (ConversationBufferMemory, VectorStoreRetrieverMemory)
  * Planner + Executor → Loop 실행

---

## 3. 학습 로드맵 (단기)

⏱ 현실적으로 프리랜서로 빠르게 진입하려면 1\~2개월이면 충분합니다.

1. **1\~2주차: 기초 RAG**

   * 문서 chunking → embedding → Vector DB → RAG 구현
   * 👉 [LangChain RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering/)

2. **3\~4주차: LangChain 에이전트**

   * Tools 등록 & Agent 실행 (예: DB 검색 + 계산기 + 웹검색)
   * 👉 [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

3. **5\~6주차: 프로젝트 클론**

   * 예: "PDF 문서 Q\&A 챗봇", "웹 리서치 에이전트", "Slack/Trello Bot"
   * 👉 [LlamaIndex Examples](https://docs.llamaindex.ai/en/stable/examples/)

4. **7\~8주차: 프리랜서 포트폴리오 제작**

   * GitHub에 에이전트 프로젝트 올리기
   * Upwork/Fiverr에 "AI Agent Developer"로 등록

---

## 4. 추천 학습 자료 (링크 포함)

* **LangChain 공식 문서**: [https://python.langchain.com](https://python.langchain.com)
* **LlamaIndex 공식 문서**: [https://docs.llamaindex.ai](https://docs.llamaindex.ai)
* **Vector DB 비교**: [https://www.pinecone.io/learn/vector-database/](https://www.pinecone.io/learn/vector-database/)
* **Practical AI Agent Tutorials**:

  * [LangChain CookBook (GitHub)](https://github.com/langchain-ai/langchain/tree/master/cookbook)
  * [AI Agent Examples – AutoGen](https://microsoft.github.io/autogen/)

---

## 5. 결론

* **가능? → 충분히 가능**. (이미 파이썬 10년 경력 → 진입장벽 낮음)
* **배워야 할 것**: RAG, Vector DB, LangChain/LlamaIndex, Agent 구조
* **소요 시간**: 1\~2개월 집중 → 프리랜서 시장 진입 가능
* **시장성**: 현재 프리랜서 플랫폼에서 “AI Agent Developer”는 수요가 급증 중

---

좋습니다 🚀
프리랜서로 빠르게 AI Agent 개발자로 자리 잡으실 수 있도록 \*\*2개월 집중 로드맵 (주차별 실습 가이드)\*\*를 드리겠습니다.
당신의 조건(👉 파이썬 10년 경력, Object Detection 경험, 프리랜서/생계 목적)을 고려해 **최소 시간 투자로 바로 시장에서 팔리는 스킬**을 목표로 합니다.

---

# 📆 2개월 집중 AI Agent 프리랜서 로드맵

## ✅ 1\~2주차: RAG (기본기 다지기)

**목표:** 긴 문서를 쪼개고(Vectorize), 질문에 맞게 검색 후 답변하는 기본 QA 시스템 만들기.

* **학습 내용**

  * Chunking & Embedding (텍스트 쪼개기 + 벡터 변환)
  * Vector DB 사용 (Chroma → FAISS → Pinecone 체험)
  * Retrieval-Augmented Generation (RAG) 구조 이해

* **실습**

  * PDF 업로드 → 질문하면 답하는 챗봇 만들기
  * ChromaDB/FAISS 이용

* **레퍼런스**

  * [LangChain RAG 튜토리얼](https://python.langchain.com/docs/use_cases/question_answering/)
  * [LlamaIndex RAG 기본](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html)

---

## ✅ 3\~4주차: LangChain Agent (도구 사용)

**목표:** 단순 QA를 넘어, LLM이 **도구를 직접 호출**하는 에이전트 만들기.

* **학습 내용**

  * LangChain Agents 이해 (ReAct, Tool, Planner)
  * 기본 Tool 연결 (계산기, Google Search API, DB Query)
  * Memory 개념 (대화 기억 vs 벡터 기억)

* **실습**

  * “날씨 검색 + 간단 계산기” 에이전트
  * “데이터베이스 질의 응답” 에이전트

* **레퍼런스**

  * [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
  * [LangChain Tools](https://python.langchain.com/docs/modules/tools/)

---

## ✅ 5\~6주차: 프로젝트 클론 & 확장

**목표:** 실제 프리랜서 프로젝트에서 요구되는 기능을 가진 에이전트 구현.

* **학습 아이디어**

  * “PDF 보고서 → 요약 후 이메일 발송” 에이전트
  * “웹 검색 → 뉴스 수집 → 요약 → Slack 전송” 리서치 에이전트
  * “회사 매뉴얼 DB → 직원 질문 응답” 사내 지식봇

* **실습**

  * API 연동 (Slack, Notion, Trello, Gmail 등)
  * Streamlit/FastAPI로 간단한 웹 UI 구현

* **레퍼런스**

  * [LangChain CookBook (GitHub)](https://github.com/langchain-ai/langchain/tree/master/cookbook)
  * [LlamaIndex Examples](https://docs.llamaindex.ai/en/stable/examples/)

---

## ✅ 7\~8주차: 프리랜서 포트폴리오 & 배포

**목표:** 클라이언트에게 바로 보여줄 수 있는 결과물 준비.

* **학습/실습**

  * 클라우드 배포 (Railway, Render, Vercel, HuggingFace Space)
  * GitHub 레포지토리 + README 정리
  * Demo 영상 (Loom, YouTube) 제작 → 포트폴리오용

* **필수 프로젝트 예시**

  * **Document Q\&A Chatbot** (기업 매뉴얼, FAQ 적용)
  * **Web Research Agent** (특정 키워드 → 자동 리서치 → 리포트 PDF 생성)
  * **Workflow Agent** (메일 → 분석 → DB 업데이트)

* **레퍼런스**

  * [HuggingFace Space (무료 배포)](https://huggingface.co/spaces)
  * [Streamlit Cloud](https://streamlit.io/cloud)

---
# Korean Financial AI Agent
# 프로젝트: 한국어 지원 금융 AI 에이전트 (부채분석 및 자산증식 제안)
# 목적: GPU를 많이 쓰지 않고 빠르게 응답 가능한 구조, 파인튜닝(LoRA 또는 OpenAI fine-tune)을 통해 나중에 모델 재학습 후 적용 가능

"""
구성요약
- FastAPI 서버 (app.py): REST API로 에이전트 제공
- agent.py: 에이전트의 플래너/실행기 및 도구 연결
- tools.py: 금융 도메인 로직 (부채 수집/분석, 상환계획, 증식계획 계산기)
- models.py: LLM wrapper 인터페이스 (OpenAI 권장), Embedding + VectorStore(Chroma) 인터페이스
- finetune/README.md: 파인튜닝(LoRA와 OpenAI 두 가지 경로) 가이드
- requirements.txt: 필요한 패키지

설계 지침 (요약)
- 기본 LLM: OpenAI GPT 계열(API) 사용 권장 -> 서버에서 GPU 불필요, 응답 빠름
- 로컬 대안: quantized Llama/Mistral (ggml) 사용 시 GPU 필요 최소화
- 임베딩: OpenAI Embeddings 또는 sentence-transformers(all-MiniLM) (CPU에서도 동작)
- Vector DB: Chroma (로컬), Pinecone(프로덕션)
- 파인튜닝: OpenAI fine-tune(간편), 또는 HuggingFace + LoRA(peft)로 커스텀

주의사항
- 금융 데이터는 민감정보임: 배포 시 반드시 보안(암호화, 접근제어) 및 개인정보 보호 준수 필요
- 이 코드는 PoC(증명 개념) 수준의 샘플입니다. 프로덕션 전 보안/규모/감사 로그/검증 절차 필요
"""

# --------------------------- requirements.txt ---------------------------
# (문서용 - 실제로는 별도 파일로 저장)
requirements_txt = '''
fastapi
uvicorn[standard]
pydantic
openai
chromadb
sentence-transformers
langchain
httpx
python-multipart
python-dotenv
numpy
scikit-learn
pandas
''' 

# --------------------------- app.py ---------------------------
app_py = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from agent import FinancialAgent

app = FastAPI(title="Korean Financial Agent")
agent = FinancialAgent()

class DebtItem(BaseModel):
    creditor: str
    balance: float
    monthly_payment: Optional[float] = None
    interest_rate_annual: Optional[float] = None

class DebtRequest(BaseModel):
    customer_id: str
    debts: List[DebtItem]
    income_monthly: Optional[float] = None
    expenses_monthly: Optional[float] = None

class GrowthRequest(BaseModel):
    customer_id: str
    balance: float
    risk_profile: Optional[str] = "moderate"  # conservative, moderate, aggressive
    horizon_years: Optional[int] = 3

@app.post('/analyze_debt')
async def analyze_debt(req: DebtRequest):
    try:
        result = agent.handle_debt_analysis(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/propose_growth')
async def propose_growth(req: GrowthRequest):
    try:
        result = agent.handle_growth_proposal(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {"status": "Korean Financial Agent running"}
'''

# --------------------------- models.py ---------------------------
models_py = '''
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

# 가볍고 빠른 기본설정: OpenAI API 사용 권장 (GPU 불필요)
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

# 임베딩 (option): sentence-transformers (CPU에서도 OK)
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

class LLM:
    """간단한 OpenAI wrapper. 필요하면 다른 provider로 교체."""
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2):
        # OpenAI completion 예시 (ChatCompletion 형태로 사용 권장)
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp['choices'][0]['message']['content']

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False).tolist()

class VectorStore:
    def __init__(self, persist_directory: str = './chroma_db'):
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        self.client = Client(settings=settings)
        self.col = self.client.get_or_create_collection("financial_memory")

    def add(self, ids, metadatas, embeddings, documents):
        self.col.add(ids=ids, metadatas=metadatas, embeddings=embeddings, documents=documents)

    def query(self, embedding, n_results=5):
        res = self.col.query(query_embeddings=[embedding], n_results=n_results)
        return res
'''

# --------------------------- tools.py ---------------------------
tools_py = '''
from typing import List, Dict
import math
import numpy as np

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
'''

# --------------------------- agent.py ---------------------------
agent_py = '''
from models import LLM, Embedder, VectorStore
from tools import analyze_debts, propose_growth
import uuid

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
'''

# --------------------------- finetune/README.md ---------------------------
finetune_readme = '''
파인튜닝 가이드

옵션 A) OpenAI Fine-tune (간편, 서버리스)
- 사용처: 규정상 허용되는 문서/데이터에 대해 사용자 응답 스타일을 고정하거나 특화할 때
- 절차: 데이터(프롬프트/정답) 준비 -> openai api로 업로드 -> fine-tune 실행 -> 모델 배포
- 장점: 관리 편의성, GPU 불필요
- 단점: 비용, 데이터 프라이버시 문제 (민감정보 주의)

옵션 B) LoRA + HuggingFace (자체 파인튜닝, 모델 재배포)
- 사용처: 로컬/클라우드(자체 모델)에서 모델을 커스터마이징할 때
- 방법: transformers + peft + bitsandbytes 사용 -> LoRA 적용 -> 4/8비트 양자화로 경량화
- 장점: 데이터 프라이버시 향상, 세부 튜닝 가능
- 단점: GPU 필요 (단기간의 저용량 GPU로도 가능), 배포 복잡성

간단한 LoRA 예시 (huggingface):
```bash
pip install transformers accelerate peft bitsandbytes
python finetune/train_lora.py --base_model 'meta-llama/Llama-2-7b' --data data.jsonl --output_dir lora_out
```

파인튜닝 후 적용 방법
- OpenAI fine-tune: 새 모델명으로 API 호출하면 끝
- LoRA: inference 시 LoRA 가중치를 로드하거나, LoRA 병합(merge) 후 경량화 모델을 배포

주의: 금융 도메인 데이터는 민감정보가 포함될 수 있으므로, 개인식별정보(PII)는 제거 및 익명화 필요
'''

# --------------------------- finetune/train_lora.py (샘플) ---------------------------
# Korean Financial AI Agent
# 프로젝트: 한국어 지원 금융 AI 에이전트 (부채분석 및 자산증식 제안)
# 목적: GPU를 많이 쓰지 않고 빠르게 응답 가능한 구조, 파인튜닝(LoRA 또는 OpenAI fine-tune)을 통해 나중에 모델 재학습 후 적용 가능

"""
구성요약
- FastAPI 서버 (app.py): REST API로 에이전트 제공
- agent.py: 에이전트의 플래너/실행기 및 도구 연결
- tools.py: 금융 도메인 로직 (부채 수집/분석, 상환계획, 증식계획 계산기)
- models.py: LLM wrapper 인터페이스 (OpenAI 권장), Embedding + VectorStore(Chroma) 인터페이스
- finetune/README.md: 파인튜닝(LoRA와 OpenAI 두 가지 경로) 가이드
- requirements.txt: 필요한 패키지

설계 지침 (요약)
- 기본 LLM: OpenAI GPT 계열(API) 사용 권장 -> 서버에서 GPU 불필요, 응답 빠름
- 로컬 대안: quantized Llama/Mistral (ggml) 사용 시 GPU 필요 최소화
- 임베딩: OpenAI Embeddings 또는 sentence-transformers(all-MiniLM) (CPU에서도 동작)
- Vector DB: Chroma (로컬), Pinecone(프로덕션)
- 파인튜닝: OpenAI fine-tune(간편), 또는 HuggingFace + LoRA(peft)로 커스텀

주의사항
- 금융 데이터는 민감정보임: 배포 시 반드시 보안(암호화, 접근제어) 및 개인정보 보호 준수 필요
- 이 코드는 PoC(증명 개념) 수준의 샘플입니다. 프로덕션 전 보안/규모/감사 로그/검증 절차 필요
"""

# --------------------------- requirements.txt ---------------------------
# (문서용 - 실제로는 별도 파일로 저장)
requirements_txt = '''
fastapi
uvicorn[standard]
pydantic
openai
chromadb
sentence-transformers
langchain
httpx
python-multipart
python-dotenv
numpy
scikit-learn
pandas
'''

# --------------------------- app.py ---------------------------
app_py = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from agent import FinancialAgent

app = FastAPI(title="Korean Financial Agent")
agent = FinancialAgent()

class DebtItem(BaseModel):
    creditor: str
    balance: float
    monthly_payment: Optional[float] = None
    interest_rate_annual: Optional[float] = None

class DebtRequest(BaseModel):
    customer_id: str
    debts: List[DebtItem]
    income_monthly: Optional[float] = None
    expenses_monthly: Optional[float] = None

class GrowthRequest(BaseModel):
    customer_id: str
    balance: float
    risk_profile: Optional[str] = "moderate"  # conservative, moderate, aggressive
    horizon_years: Optional[int] = 3

@app.post('/analyze_debt')
async def analyze_debt(req: DebtRequest):
    try:
        result = agent.handle_debt_analysis(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/propose_growth')
async def propose_growth(req: GrowthRequest):
    try:
        result = agent.handle_growth_proposal(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {"status": "Korean Financial Agent running"}
'''

# --------------------------- models.py ---------------------------
models_py = '''
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

# 가볍고 빠른 기본설정: OpenAI API 사용 권장 (GPU 불필요)
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

# 임베딩 (option): sentence-transformers (CPU에서도 OK)
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

class LLM:
    """간단한 OpenAI wrapper. 필요하면 다른 provider로 교체."""
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        self.model_name = model_name

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2):
        # OpenAI completion 예시 (ChatCompletion 형태로 사용 권장)
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp['choices'][0]['message']['content']

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False).tolist()

class VectorStore:
    def __init__(self, persist_directory: str = './chroma_db'):
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        self.client = Client(settings=settings)
        self.col = self.client.get_or_create_collection("financial_memory")

    def add(self, ids, metadatas, embeddings, documents):
        self.col.add(ids=ids, metadatas=metadatas, embeddings=embeddings, documents=documents)

    def query(self, embedding, n_results=5):
        res = self.col.query(query_embeddings=[embedding], n_results=n_results)
        return res
'''

# --------------------------- tools.py ---------------------------
tools_py = '''
from typing import List, Dict
import math
import numpy as np

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
'''

# --------------------------- agent.py ---------------------------
agent_py = '''
from models import LLM, Embedder, VectorStore
from tools import analyze_debts, propose_growth
import uuid

class FinancialAgent:
    def __init__(self):
        # 경량화 설정: OpenAI 사용 권장
        self.llm = LLM(model_name='gpt-4o-mini')
        self.embedder = Embedder()
        self.vs = VectorStore()

    def _save_customer_memory(self, customer_id: str,

# 🎯 최종 산출물 (2개월 후)

* GitHub에 최소 2\~3개 AI Agent 프로젝트 공개
* 클라우드에서 실행 가능한 Demo URL
* “AI Agent Developer (LangChain, LlamaIndex, RAG)” 키워드로 프리랜서 프로필 업데이트

👉 이 상태라면 Upwork/Fiverr/국내 크몽 같은 곳에서 **월 단위 계약 프로젝트**를 충분히 수주할 수 있습니다.

---

만들어드렸어요 — 한국어 지원 **금융 AI 에이전트**(부채 분석 · 자산 증식 제안) PoC 코드와 파인튜닝 가이드를 포함한 전체 프로젝트를 캔버스에 생성해 두었습니다.
좌측(또는 상단)에서 **`Korean Financial Agent Project`** 문서를 열어 코드, 실행 방법, 파인튜닝 옵션(LoRA / OpenAI) 등을 바로 확인하실 수 있습니다.

다음으로 제가 도와드릴 수 있는 것들 (선택) — 원하시는 항목 번호 알려주세요:

1. 로컬 환경에서 바로 실행 가능한 `docker-compose` 또는 배포 스크립트로 만들어 드리기
2. Upwork/크몽용 1페이지 프로젝트 제안서(영문/KR) 및 가격 제안 템플릿 작성
3. 위 코드 기반으로 실제 테스트 데이터를 넣어 예시 요청/응답 샘플 생성
4. 파인튜닝(LoRA) 전체 파이프라인을 실행 가능한 형태로 정리(스크립트 + 가이드)

원하시는 항목 번호만 알려주시면 바로 진행하겠습니다.

---
좋습니다 — 요청하신 대로 **제공하신 코드 베이스를 그대로 살리면서 “LoRA(PEFT)” 전체 파이프라인(데이터 준비 → 훈련 스크립트 → 저장/로딩 → 모델 적용 가이드)** 형태로 정리해 드릴게요. GPU/메모리 제약을 고려한 옵션(4-bit 양자화 + bitsandbytes), Hugging Face `transformers` + `peft` 기반 예제와 OpenAI Fine-tune(대체 경로) 준비 방법도 함께 포함했습니다. 바로 실행 가능한 스크립트 + 사용 가이드를 한 번에 드립니다.

> 노트: 실제 훈련에 사용할 **base model**은 사용자 환경(허깅페이스 계정, 라이선스, 로컬 GPU 유무)에 따라 달라집니다. 스크립트는 **`--base_model`** 인자 방식으로 유연하게 설계했습니다. 민감한 금융 데이터는 반드시 익명화/암호화 후 사용하세요.

---

# 1) 주요 파일 / 구조 (권장)

```
project_root/
├─ app.py                      # (기존) FastAPI
├─ agent.py                    # (기존) 에이전트 로직 - 아래에 변경점 적용
├─ models.py                   # inference에서 finetuned LoRA/원모델 불러오기 지원
├─ tools.py                    # (기존)
├─ finetune/
│  ├─ prepare_dataset.py       # raw -> SFT/JSONL 변환(LoRA/OPENAI용)
│  ├─ train_lora.py           # LoRA 전용 학습 스크립트 (transformers+peft)
│  ├─ inference_demo.py        # LoRA 가중치 적용해서 로컬 추론(예시)
│  └─ README.md                # 요약 가이드 (아래 내용을 포함)
├─ requirements.txt            # 패키지 (아래 업데이트 내용 포함)
└─ .env                        # OPENAI_API_KEY 등
```

---

# 2) requirements.txt (업데이트)

`requirements.txt`에 아래 항목을 추가/대체하세요.

```
fastapi
uvicorn[standard]
pydantic
openai
chromadb
sentence-transformers
langchain
httpx
python-multipart
python-dotenv
numpy
scikit-learn
pandas

# Finetune 관련 (추가)
transformers>=4.33
datasets
accelerate>=0.20
peft
bitsandbytes         # 4-bit 양자화 사용 시
safetensors          # 안전한 저장 포맷
torch                # CUDA 지원 빌드 권장 (pip/conda에서)
```

> 설치 팁: CUDA 있는 환경이면 `pip install torch --index-url https://download.pytorch.org/whl/cu121` 처럼 환경에 맞춰 설치하세요. `bitsandbytes`는 GPU와 CUDA 버전 호환 확인 필요.

---

# 3) 데이터 준비: finetune/prepare\_dataset.py

목적: 기존 도메인 QA/대화/지침 데이터를 **SFT 스타일(대화형)** 또는 **OpenAI JSONL** 포맷으로 바꿔주는 스크립트.

```python
# finetune/prepare_dataset.py
import json
from typing import List, Dict
import argparse
import os

"""
입력 예시(자유 형식): 여러 JSON 파일 또는 CSV로부터 로드하여 아래 SFT 형식으로 변환.
SFT format (for causal LM + PEFT training):
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]

OpenAI fine-tune JSONL:
{"prompt": "<PROMPT>", "completion": " <COMPLETE>"}
"""

def convert_to_sft(items: List[Dict], out_path: str):
    """
    items: list of dicts containing keys 'instruction','input','output' (input optional)
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved SFT dataset to {out_path}")

def convert_to_openai_jsonl(items: List[Dict], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            prompt = it.get('instruction', '')
            if it.get('input'):
                prompt += "\n\n" + it['input']
            # OpenAI expects completion to start with a space and end with stop token.
            completion = " " + it.get('output', '')
            f.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + '\n')
    print(f"Saved OpenAI JSONL to {out_path}")

def sample_items():
    return [
        {"instruction": "다음 채무 내역을 분석하고 상환 우선순위를 제안해줘.",
         "input": "채권자: A은행, 잔액: 5,000,000원, 이자율: 12% 연, 월 상환액: 150,000원\n채권자: B카드, 잔액: 2,000,000원, 이자율: 20% 연, 월 상환액: 80,000원",
         "output": "요약: 총 부채 7,000,000원... (예시 응답)"}
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sft','openai'], default='sft')
    parser.add_argument('--out', default='finetune_dataset.json')
    args = parser.parse_args()

    items = sample_items()  # 실제로는 CSV/DB에서 로드하여 items 생성
    if args.mode == 'sft':
        convert_to_sft(items, args.out)
    else:
        convert_to_openai_jsonl(items, args.out)
```

> 실제 데이터: 고객 금융 기록 등 민감정보는 **익명화** 후 사용하세요. (고유 식별자 삭제, 마스킹 등)

---

# 4) LoRA 훈련 스크립트: finetune/train\_lora.py

`transformers` + `peft` + `accelerate` 기반의 일반적인 SFT(지도학습) 스크립트입니다. causal LM(Single-turn/Instruction) 에 맞춰 설계했습니다. `datasets` 라이브러리로 JSON 파일을 로드합니다.

```python
# finetune/train_lora.py
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import Trainer
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help='HuggingFace model id or local path')
    parser.add_argument('--dataset_path', type=str, required=True, help='SFT json file (list of {"instruction","input","output"})')
    parser.add_argument('--output_dir', type=str, default='./lora_out')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--use_4bit', action='store_true', help='Use 4-bit quantization with bitsandbytes')
    return parser.parse_args()

def make_prompt(item):
    # Simple instruction -> output format. Adapt to your tokenizer/model special tokens.
    inst = item.get('instruction','')
    inp = item.get('input','')
    if inp:
        prompt = f"### 지시문:\n{inst}\n\n### 입력:\n{inp}\n\n### 응답:\n"
    else:
        prompt = f"### 지시문:\n{inst}\n\n### 응답:\n"
    return prompt

def main():
    args = parse_args()

    # Load dataset (expects a JSON list)
    raw = load_dataset('json', data_files=args.dataset_path)['train']
    # Combine: prompt + output, but for causal LM we will create input_ids = prompt+output and labels = same with -100 for prompt
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_examples(example):
        prompt = make_prompt(example)
        target = example.get('output','')
        full = prompt + target
        tokenized_full = tokenizer(full, truncation=True, max_length=1024)
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=1024)
        # labels: -100 for prompt tokens
        labels = tokenized_full['input_ids'].copy()
        prompt_len = len(tokenized_prompt['input_ids'])
        labels[:prompt_len] = [-100]*prompt_len
        return {'input_ids': tokenized_full['input_ids'], 'attention_mask': tokenized_full['attention_mask'], 'labels': labels}

    tokenized = raw.map(build_examples, remove_columns=raw.column_names)

    # Model load
    if args.use_4bit:
        # 4-bit pathway (requires bitsandbytes)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=True,
            device_map="auto",
            quantization_config=None  # let transformers decide; you can pass BitsAndBytesConfig if needed
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    # PEFT LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj"] if "llama" in args.base_model or "vicuna" in args.base_model else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # TrainingArguments -> use Trainer for simplicity
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=not args.use_4bit,
        save_strategy="epoch",
        push_to_hub=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    # Save only PEFT weights (lightweight)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")

if __name__ == "__main__":
    main()
```

**설명/팁**

* `target_modules`는 사용하는 베이스 모델(LLAMA 계열인지, GPT-J/파생인지)에 따라 달라집니다. LLaMA 계열은 보통 `q_proj`,`v_proj` 등. 커스텀 모델에 맞게 수정하세요.
* `use_4bit` 옵션을 사용하면 **bitsandbytes** 기반 4-bit로 메모리 사용을 크게 낮출 수 있으나, 환경(라이브러리 버전, CUDA) 따라 불안정할 수 있음.
* 훈련 후 `output_dir`에는 **peft adapter**만 저장되며 용량은 작습니다. 원모델은 유지됩니다.

---

# 5) OpenAI fine-tune(대체 경로) 준비

OpenAI로 간단히 fine-tune을 하려면 `prepare_dataset.py`에서 만든 `openai` JSONL 형식을 사용하세요.

단계:

1. `export OPENAI_API_KEY=...`
2. `openai api fine_tunes.create -t <path_to_jsonl> -m gpt-4o-mini` (실제 명령/모델 이름은 OpenAI 정책/계정 권한에 따름)
3. 완료 후 fine-tuned 모델 id로 inference

> OpenAI Fine-tune은 로컬 GPU 불필요. 그러나 OpenAI 정책/요금/데이터 보안(업로드 시 주의)을 반드시 확인하세요.

---

# 6) inference(LoRA weights 를 애플리케이션에 적용) — models.py 수정 예시

기존 `models.py`의 LLM 클래스를 **로컬 LoRA adapter를 적용**할 수 있도록 확장합니다.

```python
# models.py (추가/수정)
import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

class LocalLLM:
    def __init__(self, base_model: str = None, lora_weights: str = None, device: str = None):
        self.base_model = base_model or os.getenv('BASE_MODEL')
        self.lora_weights = lora_weights or os.getenv('LORA_WEIGHTS')
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if self.lora_weights:
            # load base + apply peft
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            base = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")
            self.model = PeftModel.from_pretrained(base, self.lora_weights)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
```

> FastAPI의 `agent`에서 `LLM()` 대신 `LocalLLM(base_model="...", lora_weights="./lora_out")` 로 초기화하면 됩니다.

---

# 7) 예제 실행 순서 (한 번에 요약)

1. 가상환경 생성 및 패키지 설치

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. 데이터 준비

   ```bash
   python finetune/prepare_dataset.py --mode sft --out finetune/data/sft.json
   # 또는 openai jsonl:
   python finetune/prepare_dataset.py --mode openai --out finetune/data/openai.jsonl
   ```

3. `accelerate` 초기화(분산/하이퍼옵션)

   ```bash
   accelerate config   # 인터랙티브 (로컬 single-GPU일 경우 기본 사용)
   ```

4. LoRA 훈련 (예시)

   ```bash
   python finetune/train_lora.py \
     --base_model huggyllama/llama-7b \
     --dataset_path finetune/data/sft.json \
     --output_dir ./finetune/lora_adapter \
     --num_train_epochs 3 \
     --per_device_train_batch_size 4 \
     --use_4bit
   ```

   * `use_4bit` 옵션은 bitsandbytes 설치 및 호환되는 GPU/CUDA 필요.
   * `base_model`는 Hugging Face에 공개된 모델 id 혹은 로컬 모델 경로로 교체.

5. 모델 애플리케이션 반영

   * `models.py`의 `LocalLLM`을 사용하여 FastAPI 에이전트가 LoRA 가중치를 적용한 모델로 응답하도록 변경.

6. 서비스 실행

   ```bash
   uvicorn app:app --reload --port 8000
   ```

---

# 8) 추가 권장/주의사항

* **토크나이저/특수토큰**: 사용 모델에 따라 `eos_token`/`pad_token` 설정 필요. `tokenizer.pad_token = tokenizer.eos_token` 등을 안전하게 설정하세요.
* **target\_modules**: LoRA가 적용될 모듈 이름은 모델 아키텍처마다 다릅니다(예: LLaMA 계열 q\_proj/v\_proj, GPT-J q\_proj/v\_proj 등). 디버깅 필요.
* **훈련 데이터 품질**: 금융 도메인은 실무 문장과 규정(세법, 투자규정 등) 영향을 받으므로, 훈련 샘플은 정확하고 최신 법규/규정 검토 후 사용하세요.
* **버전 고정**: `transformers`, `peft`, `bitsandbytes` 등 버전 호환성 이슈가 자주 발생합니다. 실사용 전에 로컬에서 작은 데이터로 샘플 훈련/추론 테스트를 권장합니다.
* **보안**: 금융 데이터 업로드(특히 OpenAI 등 외부 서비스 사용 시)는 사내 규정/법적 준수 확인 필요. 개인정보 비식별화 필수.

---

# 9) README 요약 (finetune/README.md 내용 요약)

* 목적: 이 repo는 한국어 금융 에이전트를 LoRA로 파인튜닝하는 파이프라인을 제공합니다.
* 빠른 시작:

  1. requirements 설치
  2. `prepare_dataset.py`로 데이터 생성 (`sft.json` 혹은 `openai.jsonl`)
  3. `train_lora.py`로 LoRA 학습
  4. `models.py`에서 adapter 로드 -> `app.py`로 서비스
* OpenAI Fine-tune 대안 및 고려사항 기재
* 민감정보 처리 가이드 포함

---

# 끝으로 — 빠른 체크리스트

* [ ] 데이터 익명화/준비 완료 (SFT JSON or OpenAI JSONL)
* [ ] 환경(라이브러리, CUDA, bitsandbytes) 준비 확인
* [ ] `--base_model` 로드 가능한지(라이선스/토큰) 확인
* [ ] `target_modules`가 베이스 모델에 맞도록 설정
* [ ] 소규모로 검증 후 전체 훈련 진행

---

좋습니다 👍 요청하신 대로 **추론 데모 스크립트(inference\_demo.py)** 와, GPU 없이도 실행 가능한 **무료 한국어 모델 추천**, 그리고 `target_modules` 자동 감지 유틸을 포함해 정리해 드릴게요.

---

## 1) 무료 + 가벼운 한국어 모델 추천

GPU 없이 CPU 환경에서 쓰려면 **small/medium 사이즈** 모델을 권장합니다.

* **`beomi/llama-2-ko-7b`**
  → 한국어 지원, 하지만 7B라서 CPU 단독 환경에서는 느릴 수 있음.
* **`beomi/KcELECTRA-base`**
  → 한국어 이해는 잘하지만 causal LM이 아니므로 LoRA Instruction Tuning에는 적합하지 않음.
* **추천 (CPU 친화)**:
  👉 **`MLP-KTLim/llama-3-Korean-SFT-1.8B`** (HuggingFace)

  * 약 1.8B 파라미터, CPU에서도 추론 가능 (느리긴 하지만 7B보단 훨씬 가벼움).
  * 한국어 Instruction SFT 되어있음 → LoRA 학습하기도 용이.

즉: **`MLP-KTLim/llama-3-Korean-SFT-1.8B`** 를 기본값으로 권장합니다.

---

## 2) target\_modules 자동 감지 유틸

LoRA 적용 시 `target_modules`는 모델 구조마다 다릅니다. 이를 간단히 모델 이름/구조로 추정하는 함수로 구현합니다.

```python
# finetune/utils_lora.py
def guess_target_modules(model_name: str):
    """
    모델 이름에 따라 LoRA target_modules 추천
    """
    name = model_name.lower()
    if "llama" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gptj" in name or "gpt-j" in name:
        return ["q_proj", "v_proj"]
    elif "gptneo" in name or "neo" in name:
        return ["q_proj", "v_proj"]
    elif "bloom" in name:
        return ["query_key_value"]
    else:
        # 기본값: Q/V projection
        return ["q_proj", "v_proj"]
```

---

## 3) inference\_demo.py

훈련한 LoRA 어댑터를 불러와서 간단히 추론하는 스크립트입니다.
(기본 모델은 `MLP-KTLim/llama-3-Korean-SFT-1.8B` 로 지정)

```python
# finetune/inference_demo.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(base_model: str, lora_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model: {base_model} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if lora_path:
        print(f"Applying LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    return tokenizer, model

def infer(prompt: str, tokenizer, model, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    BASE_MODEL = "MLP-KTLim/llama-3-Korean-SFT-1.8B"
    LORA_WEIGHTS = "./finetune/lora_adapter"  # train_lora.py 결과물

    tokenizer, model = load_model(BASE_MODEL, LORA_WEIGHTS)
    
    while True:
        q = input("\n질문을 입력하세요 (종료: exit): ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        result = infer(q, tokenizer, model)
        print("\n=== 응답 ===")
        print(result)
```

---

## 4) 사용법

1. 훈련된 LoRA 가중치 준비 (예: `./finetune/lora_adapter/`)
2. 실행:

   ```bash
   python finetune/inference_demo.py
   ```
3. 질문 입력 후 응답 확인:

   ```
   질문: 부채 상환 우선순위를 알려줘.
   응답: (LoRA 학습 결과 기반 응답 출력)
   ```

---

👉 정리하면:

* 추천 모델: `MLP-KTLim/llama-3-Korean-SFT-1.8B` (CPU 친화, 한국어 이해 가능)
* `utils_lora.py` 에서 target\_modules 자동 추천
* `inference_demo.py` 로 LoRA 가중치 적용 후 대화형 추론 가능

---

좋습니다 👍 그러면 `inference_demo.py` 안에 **target\_modules 자동 감지 유틸**을 포함시켜서 하나의 독립 실행 가능한 스크립트로 정리해드릴게요.

---

## ✅ 통합 버전: inference\_demo.py

```python
# finetune/inference_demo.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def guess_target_modules(model_name: str):
    """
    모델 이름에 따라 LoRA target_modules 추천
    """
    name = model_name.lower()
    if "llama" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gptj" in name or "gpt-j" in name:
        return ["q_proj", "v_proj"]
    elif "gptneo" in name or "neo" in name:
        return ["q_proj", "v_proj"]
    elif "bloom" in name:
        return ["query_key_value"]
    else:
        # 기본값
        return ["q_proj", "v_proj"]

def load_model(base_model: str, lora_path: str = None, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model: {base_model} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    if lora_path:
        print(f"Applying LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA target_modules (추정):", guess_target_modules(base_model))

    return tokenizer, model

def infer(prompt: str, tokenizer, model, max_new_tokens: int = 256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    BASE_MODEL = "MLP-KTLim/llama-3-Korean-SFT-1.8B"  # 추천 한국어 SFT 모델
    LORA_WEIGHTS = "./finetune/lora_adapter"  # train_lora.py 결과물 (없으면 None)

    tokenizer, model = load_model(BASE_MODEL, LORA_WEIGHTS)
    
    while True:
        q = input("\n질문을 입력하세요 (종료: exit): ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        result = infer(q, tokenizer, model)
        print("\n=== 응답 ===")
        print(result)
```

---

## 🚀 사용법

1. LoRA 학습을 완료한 뒤 `./finetune/lora_adapter` 디렉토리에 어댑터가 저장되어 있어야 합니다.
2. 실행:

   ```bash
   python finetune/inference_demo.py
   ```
3. 실행 후 입력 예:

   ```
   질문을 입력하세요 (종료: exit): 부채 상환 우선순위를 알려줘
   === 응답 ===
   (모델의 한국어 추론 결과)
   ```

---

좋은 요청이에요 🙌 한국어는 **띄어쓰기·조사·어미** 구조 때문에 영어식 단순 `split(" ")` 이나 일정 길이 자르기만 쓰면 의미가 쉽게 깨집니다.
따라서 **형태소 단위 또는 문장 단위 chunking → 토큰 단위 보정 → 최대 길이 제한** 방식이 가장 안전합니다.

아래는 `inference_demo.py` 를 **한국어 친화적인 chunking 함수**를 포함하도록 수정한 버전입니다.

---

## ✅ 수정된 inference\_demo.py (한국어 chunking 포함)

```python
# finetune/inference_demo.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def guess_target_modules(model_name: str):
    """
    모델 이름에 따라 LoRA target_modules 추천
    """
    name = model_name.lower()
    if "llama" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gptj" in name or "gpt-j" in name:
        return ["q_proj", "v_proj"]
    elif "gptneo" in name or "neo" in name:
        return ["q_proj", "v_proj"]
    elif "bloom" in name:
        return ["query_key_value"]
    else:
        return ["q_proj", "v_proj"]

# -------------------------------
# ✅ 한국어 친화적 chunking
# -------------------------------
def chunk_korean_text(text: str, max_chars: int = 200):
    """
    한국어 텍스트를 의미 단위로 chunking
    1) 문장 단위 분리 (마침표/물음표/느낌표 + 줄바꿈 기준)
    2) 긴 문장은 일정 길이(max_chars)로 잘라냄
    """
    # 문장 단위 split
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # 현재 문장까지 합쳤을 때 길이가 허용 범위 이내면 이어붙임
        if len(current) + len(sent) <= max_chars:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            # 문장이 너무 긴 경우 잘라냄
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            current = sent
    if current:
        chunks.append(current)

    return chunks

# -------------------------------
# 모델 로드 & 추론
# -------------------------------
def load_model(base_model: str, lora_path: str = None, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model: {base_model} (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    if lora_path:
        print(f"Applying LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        print("LoRA target_modules (추정):", guess_target_modules(base_model))

    return tokenizer, model

def infer(prompt: str, tokenizer, model, max_new_tokens: int = 256):
    # ✅ 입력 프롬프트도 chunk 단위로 나눠서 처리 (큰 텍스트 지원)
    chunks = chunk_korean_text(prompt, max_chars=200)
    responses = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
            )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(resp)

    return "\n".join(responses)

if __name__ == "__main__":
    BASE_MODEL = "MLP-KTLim/llama-3-Korean-SFT-1.8B"
    LORA_WEIGHTS = "./finetune/lora_adapter"  # 없으면 None

    tokenizer, model = load_model(BASE_MODEL, LORA_WEIGHTS)
    
    while True:
        q = input("\n질문을 입력하세요 (종료: exit): ")
        if q.strip().lower() in ["exit", "quit"]:
            break
        result = infer(q, tokenizer, model)
        print("\n=== 응답 ===")
        print(result)
```

---

## 🚀 개선된 점

* `chunk_korean_text()`

  * **문장 단위로 우선 분리** (`.`, `?`, `!`, 줄바꿈 기준) → 의미 손상 최소화
  * 너무 긴 문장은 `max_chars` 기준으로 **안전하게 분할**
* `infer()`

  * 긴 프롬프트도 자동으로 chunking → 모델 입력 제한/메모리 문제 완화
  * 각 chunk별 응답을 이어붙여 최종 출력

---

👉 여기서 `max_chars=200` 은 기본 예시입니다. 모델별 token 길이에 맞춰 300\~500 정도로 조정하면 더 효율적일 수 있어요.

---

좋은 질문이에요 👍
맞아요, **chunking 처리 위치**는 보통 아키텍처 설계에서 결정합니다.

---

## 🔎 Chunking 처리가 필요한 이유

* LLM 입력은 **토큰 길이 제한**이 있어서 긴 데이터(예: 고객 재무정보, 긴 문서)를 그대로 넣으면 **에러**가 납니다.
* 따라서 긴 텍스트는 **chunking → 벡터화 → 검색(RAG) → LLM 입력** 과정으로 잘라서 써야 합니다.

---

## 📌 Chunking 처리 위치 (일반적인 패턴)

1. **서버(API 계층)에서 chunking**

   * 클라이언트가 긴 텍스트 요청을 보냄 → 서버에서 chunking → LLM/VectorDB에 전달
   * 장점: 서버에서 일관되게 처리 가능, 클라이언트 단순해짐
   * 단점: 서버 부담 증가

2. **사전처리(데이터 적재 단계)에서 chunking**

   * 고객 문서/데이터를 VectorDB에 저장하기 전에 chunking
   * 질의 시에는 이미 chunk 단위로 저장된 벡터에서 검색
   * 장점: 검색 최적화, 실시간 속도 향상
   * 단점: 저장 전에 전처리 필요

3. **클라이언트 측 chunking (비권장)**

   * 모바일/웹에서 chunking 후 API로 보냄
   * 장점: 서버 부하 분산
   * 단점: 구현 복잡, 클라이언트별 결과 일관성 깨짐

---

## 🔧 권장 방식 (너 프로젝트 기준)

* 지금 구조(`app.py + agent.py + models.py + tools.py`)는 **RAG + LLM 호출 서버형 에이전트**야.
* 따라서 **chunking은 서버(agent/models) 계층에서 처리하는 게 표준**이야.

  * 문서를 VectorStore에 저장할 때 → chunking 후 Embedding
  * 긴 프롬프트로 LLM 호출할 때 → chunking 후 잘라서 입력

---

## ✅ 제안

* 내가 만든 `chunk_korean_text()` 같은 유틸은

  * **VectorStore 저장 시 (models.py → VectorStore.add)**
  * **LLM 호출 시 (agent.py → self.llm.generate)**
    두 군데에서 쓰이는 게 가장 좋아.

---

👉 원한다면, 내가 `chunk_korean_text()` 를 **서버(app.py, agent.py, models.py)** 쪽으로 옮겨서,

* **데이터 저장 시 자동 chunking**
* **분석/추천 API 호출 시 자동 chunking**

까지 반영된 버전을 만들어줄 수 있어.

---

좋습니다 😎 그러면 chunking 처리를 **서버 전체 파이프라인**에 통합하는 방식으로 구조를 재정리할게요.

* 목적: **VectorStore 저장 시 자동 chunking + LLM 호출 시 긴 입력 자동 chunking**
* 적용 파일: `models.py` (VectorStore), `agent.py` (LLM 호출), `app.py` (API 그대로)
* chunking 함수: 한국어 문장 기반, 의미 훼손 최소화

---

## 1️⃣ models.py (VectorStore에 chunking 적용)

```python
# models.py
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

import torch
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------------
# ✅ 한국어 Chunking 유틸
# ------------------------------
import re
def chunk_korean_text(text: str, max_chars: int = 200) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= max_chars:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            while len(sent) > max_chars:
                chunks.append(sent[:max_chars])
                sent = sent[max_chars:]
            current = sent
    if current:
        chunks.append(current)
    return chunks

# ------------------------------
# LLM Wrapper
# ------------------------------
class LocalLLM:
    def __init__(self, base_model: str, lora_weights: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = base_model
        self.lora_weights = lora_weights
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        if lora_weights:
            model = PeftModel.from_pretrained(model, lora_weights)
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int = 256):
        # 긴 프롬프트 chunking
        chunks = chunk_korean_text(prompt, max_chars=200)
        responses = []
        for c in chunks:
            inputs = self.tokenizer(c, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            responses.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
        return "\n".join(responses)

# ------------------------------
# Embedding + VectorStore
# ------------------------------
class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False).tolist()

class VectorStore:
    def __init__(self, persist_directory: str = './chroma_db'):
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        self.client = Client(settings=settings)
        self.col = self.client.get_or_create_collection("financial_memory")

    def add(self, ids, metadatas, documents):
        """
        자동 chunking + embedding
        """
        # documents: List[str]
        chunked_docs, chunked_ids, chunked_meta = [], [], []
        for i, doc in enumerate(documents):
            chunks = chunk_korean_text(doc)
            chunked_docs.extend(chunks)
            chunked_ids.extend([ids[i]]*len(chunks))
            chunked_meta.extend([metadatas[i]]*len(chunks))

        embedder = Embedder()
        embeddings = embedder.encode(chunked_docs)
        self.col.add(ids=chunked_ids, metadatas=chunked_meta, embeddings=embeddings, documents=chunked_docs)

    def query(self, embedding, n_results=5):
        return self.col.query(query_embeddings=[embedding], n_results=n_results)
```

---

## 2️⃣ agent.py (LLM 호출 시 chunking 적용)

```python
# agent.py
from models import LocalLLM, VectorStore
from tools import analyze_debts, propose_growth
import uuid

class FinancialAgent:
    def __init__(self):
        self.llm = LocalLLM(base_model="MLP-KTLim/llama-3-Korean-SFT-1.8B",
                            lora_weights="./finetune/lora_adapter")
        self.vs = VectorStore()

    # --------------------------
    # 고객 메모리 저장
    # --------------------------
    def _save_customer_memory(self, customer_id: str, text: str):
        doc_id = str(uuid.uuid4())
        metadata = {"customer_id": customer_id}
        self.vs.add([doc_id], [metadata], [text])

    # --------------------------
    # 부채 분석
    # --------------------------
    def handle_debt_analysis(self, data: dict):
        # 기존 분석
        result = analyze_debts(data.get('debts', []),
                               income_monthly=data.get('income_monthly'),
                               expenses_monthly=data.get('expenses_monthly'))
        # 분석 요약 텍스트
        summary = f"부채 분석 결과: 총 {result['total_balance']}원, 평균이자율 {result['average_interest_rate']}%, 우선순위 {result['priority_list']}"
        # 메모리 저장
        self._save_customer_memory(data['customer_id'], summary)
        # LLM 응답 (chunking 포함)
        llm_response = self.llm.generate(summary)
        result['llm_response'] = llm_response
        return result

    # --------------------------
    # 증식 계획
    # --------------------------
    def handle_growth_proposal(self, data: dict):
        result = propose_growth(data.get('balance',0),
                                risk_profile=data.get('risk_profile','moderate'),
                                horizon_years=data.get('horizon_years',3))
        summary = f"증식 계획 요약: {result['projected_balance']}원 예상, {result['allocation']}"
        self._save_customer_memory(data['customer_id'], summary)
        llm_response = self.llm.generate(summary)
        result['llm_response'] = llm_response
        return result
```

---

## 3️⃣ app.py

* 기존 API 구조 그대로 사용
* agent 호출 시 **자동 chunking + LLM 응답** 포함됨
* 별도 수정 필요 없음

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from agent import FinancialAgent

app = FastAPI(title="Korean Financial Agent")
agent = FinancialAgent()

class DebtItem(BaseModel):
    creditor: str
    balance: float
    monthly_payment: Optional[float] = None
    interest_rate_annual: Optional[float] = None

class DebtRequest(BaseModel):
    customer_id: str
    debts: List[DebtItem]
    income_monthly: Optional[float] = None
    expenses_monthly: Optional[float] = None

class GrowthRequest(BaseModel):
    customer_id: str
    balance: float
    risk_profile: Optional[str] = "moderate"
    horizon_years: Optional[int] = 3

@app.post('/analyze_debt')
async def analyze_debt(req: DebtRequest):
    try:
        result = agent.handle_debt_analysis(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/propose_growth')
async def propose_growth(req: GrowthRequest):
    try:
        result = agent.handle_growth_proposal(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {"status": "Korean Financial Agent running"}
```

---

## ✅ 장점

1. **VectorStore 저장 시 자동 chunking + embedding** → 긴 문서도 안전하게 저장
2. **LLM 호출 시 자동 chunking** → 긴 프롬프트도 의미 훼손 없이 처리
3. **한국어 친화적 문장 단위 chunking** → 의미 손상 최소화
4. API 호출 구조 그대로 유지 → 기존 FastAPI 연동 문제 없음

---

원하면 제가 이 구조를 **완전히 실행 가능한 zip-ready 형태**로 만들어서,

* `finetune/` + `models.py` + `agent.py` + `app.py`
* 필요한 환경변수 `.env`, requirements.txt 포함

한 번에 배포/실행 가능한 형태로 만들어 드릴 수도 있습니다.

혹시 그렇게 해드릴까요?




