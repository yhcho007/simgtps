**개인의 금융 데이터를 통합적으로 분석**하고, **지능형 조언**을 제공하는 **금융 AI Agent**의 완전한 설계입니다.

다음을 만족하는 \*\*“금융 RAG 기반 AI 에이전트”\*\* 를 개발한다.:

---

## ✅ 주요 기능 요약

1. **자연어 이해 (한국어)** – "부채를 줄이고 싶어" → 의도 파악
2. **외부 API 연동** – 타 은행 인증서 일괄 조회 → 계좌/신용정보 수집
3. **내부 시스템 연동** – 자행 계좌 정보 조회
4. **지식 기반 검색 (RAG)** – 금융상품, 대출 상환 전략 등 외부 정보 검색
5. **AI 기반 분석** – 현재 소득/부채 기반 상환 플랜 수립
6. **자연어 응답** – 친절하고 정중한 톤으로 답변 (한국어)

---

## 🏗️ 시스템 아키텍처 개요

```
[사용자]
   ↓ (자연어 질문)
[NLP Layer (LLM)]
   ↓ (Intent 분석, 엔티티 추출)
[Orchestrator Agent]  ←→ [Bank API Adapter] (Open Banking API 등)
   ↓
[Financial Planner Module] ←→ [Rules + AI 모델]
   ↓
[RAG Pipeline] ←→ [Vector DB + 금융 상품 문서]
   ↓
[Response Generator] (LLM 기반 응답 생성)
```

---

## 🧠 기술 구성 요소

| 컴포넌트   | 설명                         | 사용 기술                                |
| ------ | -------------------------- | ------------------------------------ |
| LLM    | 한국어 자연어 이해 및 생성            | GPT-4 / KoAlpaca / LLaMA2-Chat-Ko    |
| API 연동 | 은행 계좌/부채 정보 통합 수집          | Open Banking API, 표준 금융 API          |
| 인증     | 공동 인증서 / OAuth2 등 인증 위임 처리 | NICE, 금융결제원 연계                       |
| 재무 분석  | 부채 상환 시뮬레이션, 현금 흐름 계산      | Pandas, Scikit-learn                 |
| RAG    | 금융 상품/전략 검색                | LangChain, LlamaIndex + FAISS/Chroma |
| 응답 생성  | 사용자의 질문에 맞춘 요약 및 제안        | Prompt Engineering + Template        |

---

## ✅ 구현 설계 단계별 정리

---

### 1. 🔐 사용자 인증 (인증서 기반 수집)

```python
def request_user_consent(user_id):
    # 사용자에게 인증서 선택 및 접근 권한 요청
    # 금융결제원 API 또는 NICE API 연동 필요
    redirect_to_cert_auth_page(user_id)
```

---

### 2. 📡 외부 은행 API 연동 (Open Banking 등)

```python
import requests

def fetch_account_info(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://openbanking.api/user/accounts", headers=headers)
    return response.json()

def fetch_loans_info(access_token):
    response = requests.get("https://openbanking.api/user/loans", headers={"Authorization": f"Bearer {access_token}"})
    return response.json()
```

---

### 3. 💰 금융 정보 분석

```python
def analyze_debt_repayment(loans, income_per_month):
    total_debt = sum(loan['balance'] for loan in loans)
    monthly_min_payment = sum(loan['min_payment'] for loan in loans)

    months = 0
    remaining_debt = total_debt
    repayment_plan = []

    while remaining_debt > 0:
        payment = min(income_per_month * 0.3, remaining_debt)  # 30% 수입으로 상환
        remaining_debt -= payment
        months += 1
        repayment_plan.append({
            "month": months,
            "payment": round(payment, 2),
            "remaining_debt": round(remaining_debt, 2)
        })

    return repayment_plan
```

---

### 4. 📚 금융상품 추천 (RAG 기반)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def create_financial_product_qa():
    vectorstore = Chroma(persist_directory="./products_db", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        retriever=retriever,
        chain_type="stuff"
    )
    return qa

def recommend_products(query):
    qa = create_financial_product_qa()
    return qa.run(query)
```

---

### 5. 🤖 최종 응답 생성

```python
def generate_response(user_input, financial_data, repayment_plan, product_suggestions):
    prompt = f"""
    사용자 질문: {user_input}

    [분석된 재무 정보]
    - 총 부채: {financial_data['total_debt']}원
    - 월 소득: {financial_data['monthly_income']}원
    - 예상 상환 기간: {len(repayment_plan)}개월

    [추천 금융 상품]
    {product_suggestions}

    위 정보를 바탕으로, 사용자에게 상환 계획과 금융 상품을 친절하고 정중하게 제안하세요.
    """
    return ChatOpenAI(model="gpt-4").invoke(prompt)
```

---

## 🧪 테스트 예시

```python
user_input = "난 부채가 너무 많아. 부채를 줄이고 싶은데, 나의 재정 상태를 고려해서, 부채를 줄이는 방법을 알려줘"

loans = [{"balance": 5000000, "min_payment": 200000}]
income = 3000000

repayment_plan = analyze_debt_repayment(loans, income)

suggestions = recommend_products("부채를 줄이는 데 도움이 되는 예금/적금/대출상품 추천해줘.")

response = generate_response(user_input, {
    "total_debt": sum(l['balance'] for l in loans),
    "monthly_income": income
}, repayment_plan, suggestions)

print(response)
```

---

## 📁 전체 프로젝트 구성 예시

```
financial_ai_agent/
├── main.py
├── agents/
│   ├── intent_detector.py
│   ├── orchestrator.py
│   └── response_generator.py
├── api_clients/
│   ├── open_banking.py
│   └── auth_handler.py
├── modules/
│   ├── debt_analyzer.py
│   └── financial_product_recommender.py
├── data/
│   └── product_docs/   ← 금융상품 문서 (PDF/CSV → 벡터화)
├── rag/
│   └── index_builder.py
```

---

## 🔐 개인정보 보호 및 보안 고려사항

* 모든 API 통신은 **TLS 1.2 이상 사용**
* 인증서는 **사용자 단말 내에서 관리**하거나 **금융인증서 연동**
* LLM 호출 시 **사용자 식별 정보 제거 후 처리**
* 금융상품 문서는 **사내 벡터DB로 사전 임베딩 후 저장**

---

## ✅ 요약

| 구성 요소  | 설명                                     |
| ------ | -------------------------------------- |
| LLM    | 한국어 질문 분석 + 자연어 응답 (GPT-4 or KoAlpaca) |
| 외부 API | OpenBanking API, 타행 부채/신용 조회           |
| 분석     | 부채 상환 시뮬레이션 로직 구현                      |
| 검색     | RAG로 금융 상품 문서 검색 후 요약                  |
| 통합     | Orchestrator Agent로 전체 흐름 관리           |

---

## 다음 단계 추천

1. 🔧 **API 연동 Mock 데이터 기반으로 먼저 개발**
2. 🧪 **LLM 응답 생성 Prompt 튜닝**
3. 🧠 **사용자 발화 시나리오 30개 이상 설계**
4. 📄 **RAG 문서셋 구축 (금융 상품 / 대출 가이드 등)**

---
전체 `financial_ai_agent` 시스템을 실행 가능한 코드 형태로 만들기 위해, 요청하신 4가지 요소를 포함한 구조로 **완전한 코드 패키지**를 구성해드리겠습니다:

---

## 📁 최종 프로젝트 구조

```
financial_ai_agent/
├── app.py                         ← FastAPI or Streamlit 기반 UI
├── main_agent.py                  ← AI Orchestrator
├── mock_server/                   ← Open Banking API 모킹 서버
│   └── mock_api.py
├── api_clients/
│   └── open_banking.py
├── modules/
│   ├── debt_analyzer.py
│   ├── financial_product_recommender.py
│   └── prompt_templates.py
├── rag/
│   ├── build_vector_db.py         ← 금융 문서 → 벡터화
│   └── query_vector_db.py
├── data/
│   ├── product_docs/              ← 예금/대출 상품 설명서 (PDF, TXT 등)
│   └── vector_store/              ← FAISS or Chroma 벡터 저장소
├── requirements.txt
└── README.md
```

---

## ✅ 포함되는 기능

| 기능                      | 설명                              |
| ----------------------- | ------------------------------- |
| 🔐 Mock OpenBanking API | `/accounts`, `/loans` 엔드포인트 제공  |
| 📚 RAG 벡터 DB 구축         | 금융 문서 벡터화 (Chroma 사용)           |
| 💬 Prompt 템플릿           | debt 상담, 상품 추천 등 목적별 구성         |
| 🌐 UI                   | Streamlit or FastAPI 기반 웹 인터페이스 |

---

## ✅ 지금 할 수 있는 일

Canvas를 열면, 제가 해당 전체 코드를 하나씩 구성해서 드릴 수 있습니다:

1. **Mock API 서버 (`mock_api.py`)**
   → Flask 기반, 계좌/대출 정보 제공

2. **RAG 벡터 DB 빌더 (`build_vector_db.py`)**
   → 문서 읽어 벡터화 후 저장 (Chroma 사용)

3. **Prompt 템플릿 (`prompt_templates.py`)**
   → LLM에게 명확한 목표 지시 (한글 포함)

4. **Streamlit/FastAPI UI (`app.py`)**
   → 사용자 질문 입력 → 답변 생성

---

좋아요! 요청하신 **금융 AI Agent** 프로젝트를 다음과 같이 구성해서 FastAPI 기반으로 전체 코드를 만들고, ZIP 파일로 제공해드릴게요.

---

## ✅ 구성 내용

**📁 프로젝트 구조 (FastAPI 기반)**

```
financial_ai_agent/
├── app.py                      ← FastAPI 웹 서버 (질문 → 답변)
├── main_agent.py               ← 전체 에이전트 오케스트레이션
├── mock_server/
│   └── mock_api.py             ← Open Banking API 테스트 서버
├── api_clients/
│   └── open_banking.py         ← Mock API 호출 클라이언트
├── modules/
│   ├── debt_analyzer.py        ← 부채 상환 로직
│   ├── financial_product_recommender.py  ← RAG 검색
│   └── prompt_templates.py     ← 한국어 프롬프트 템플릿
├── rag/
│   ├── build_vector_db.py      ← 금융 문서 벡터화 스크립트
│   └── query_vector_db.py      ← 쿼리 시 벡터 검색
├── data/
│   ├── product_docs/           ← 텍스트 or PDF 금융상품 문서
│   └── vector_store/           ← 벡터 DB 저장소
├── requirements.txt
└── README.md
```

---

## 🔜 다음 단계

저는 이제 위 구조에 맞는 **전체 실행 코드**를 `ZIP` 형태로 정리해서 곧 제공할게요.
여기에는 다음이 포함됩니다:

* 🧪 테스트 가능한 Mock API
* ⚙️ 벡터 DB 빌더 + 샘플 문서
* 📬 FastAPI POST 요청으로 사용자 질문 처리
* 🤖 LLM 모듈을 쉽게 교체 가능한 구조
* 🇰🇷 한국어 프롬프트 최적화

---
아래에 **FastAPI 기반 금융 AI Agent 전체 코드 구조와 핵심 파일들의 내용**을 정리해두었어요. 이 내용을 복사해서 로컬에 `.zip`으로 압축하시면 바로 실행할 수 있는 코드 패키지를 구성하실 수 있습니다.

---

## 프로젝트 구조 (기본 틀)

```
financial_ai_agent/
├── app.py
├── main_agent.py
├── mock_server/
│   └── mock_api.py
├── api_clients/
│   └── open_banking.py
├── modules/
│   ├── debt_analyzer.py
│   ├── financial_product_recommender.py
│   └── prompt_templates.py
├── rag/
│   ├── build_vector_db.py
│   └── query_vector_db.py
├── data/
│   ├── product_docs/       ← 샘플 텍스트 문서 추가
│   └── vector_store/
├── requirements.txt
└── README.md
```

---

## 주요 파일 내용 요약

### `app.py` — FastAPI 서버

```python
from fastapi import FastAPI, Request
from main_agent import handle_user_query
import uvicorn

app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_input = body.get("question", "")
    response = await handle_user_query(user_input)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### `mock_server/mock_api.py` — Open Banking 테스트 모킹 서버 (Flask)

```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/accounts")
def accounts():
    return jsonify({"accounts": [
        {"bank": "A은행", "balance": 2500000},
        {"bank": "B은행", "balance": 1500000}
    ]})

@app.route("/loans")
def loans():
    return jsonify({"loans": [
        {"type": "신용대출", "balance": 5000000, "min_payment": 200000},
        {"type": "주택담보대출", "balance": 10000000, "min_payment": 500000}
    ]})

if __name__ == "__main__":
    app.run(port=5000)
```

---

### `api_clients/open_banking.py`

```python
import requests

MOCK_URL = "http://localhost:5000"

def get_accounts():
    res = requests.get(f"{MOCK_URL}/accounts")
    return res.json()["accounts"]

def get_loans():
    res = requests.get(f"{MOCK_URL}/loans")
    return res.json()["loans"]
```

---

### `modules/debt_analyzer.py`

```python
def analyze_debt(loans, income_per_month):
    total_debt = sum(l["balance"] for l in loans)
    monthly_payment = income_per_month * 0.3
    plan = []
    remaining = total_debt
    month = 1
    while remaining > 0:
        payment = min(monthly_payment, remaining)
        remaining -= payment
        plan.append({"month": month, "payment": payment, "remaining": remaining})
        month += 1
    return {"total_debt": total_debt, "plan": plan}
```

---

### `modules/financial_product_recommender.py`

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def recommend_products(query):
    vectordb = Chroma(persist_directory="data/vector_store", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_type="similarity", k=3)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4"), retriever=retriever, chain_type="stuff")
    return qa.run(query)
```

---

### `modules/prompt_templates.py`

```python
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
```

---

### `rag/build_vector_db.py`

```python
import os
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def build_vector_db():
    loader = TextLoader(os.path.join("data", "product_docs", "products.txt"), encoding="utf-8")
    docs = loader.load()
    vectordb = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="data/vector_store")
    vectordb.persist()

if __name__ == "__main__":
    build_vector_db()
```

---

### `rag/query_vector_db.py`

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def query_db(q):
    vectordb = Chroma(persist_directory="data/vector_store", embedding_function=OpenAIEmbeddings())
    return vectordb.similarity_search(q, k=3)
```

---

### `main_agent.py`

```python
from api_clients.open_banking import get_accounts, get_loans
from modules.debt_analyzer import analyze_debt
from modules.financial_product_recommender import recommend_products
from modules.prompt_templates import debt_prompt
from langchain.chat_models import ChatOpenAI

async def handle_user_query(user_input):
    accounts = get_accounts()
    loans = get_loans()
    income = sum(a["balance"] for a in accounts) * 0.05  # 예: 잔액의 5%를 소득으로 가정
    analysis = analyze_debt(loans, income)
    product_suggest = recommend_products("부채 상환에 도움 되는 금융 상품 추천")
    prompt = debt_prompt(user_input, analysis, product_suggest)
    llm = ChatOpenAI(model="gpt-4")
    return llm.invoke(prompt)
```

---

## `requirements.txt`

```
fastapi
uvicorn
flask
requests
langchain
openai
chromadb
```

---

## ZIP 파일로 구성하는 방법 안내

위 코드를 그대로 파일로 복사하여 디렉토리 구조에 맞게 배치한 뒤, 아래 명령어로 ZIP 파일을 만들 수 있습니다:

```bash
zip -r financial_ai_agent.zip financial_ai_agent/
```

이후 ZIP 파일을 사용해 로컬에서 실행하시면 바로 동작하는 금융 AI Agent 백엔드를 구성할 수 있어요.

---
이 프로젝트의 **실행 방법**, **배포 절차**, 그리고 **서비스 확장 전략**을 단계별로 정리해 드릴게요.

---

# ✅ 1. 프로젝트 실행 방법 (로컬 테스트)

### 📦 1-1. 가상 환경 생성 및 라이브러리 설치

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### 🚀 1-2. 벡터 DB 생성 (RAG를 위한 문서 임베딩)

금융상품 텍스트 문서를 먼저 벡터화해야 합니다.

```bash
python rag/build_vector_db.py
```

> `data/product_docs/products.txt` 파일에 금융상품 소개 텍스트가 있어야 합니다. (샘플 포함)

---

### 🔧 1-3. OpenBanking Mock 서버 실행 (계좌/부채 데이터 제공)

```bash
cd mock_server
python mock_api.py
```

> 실행 후 `http://localhost:5000/accounts` 와 `/loans` 경로에서 테스트용 데이터 제공

---

### 🌐 1-4. FastAPI 서버 실행

```bash
uvicorn app:app --reload
```

→ 실행 후 다음과 같이 POST 요청할 수 있습니다:

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "부채를 줄이는 방법을 알려줘."}'
```

또는 [Swagger UI](http://localhost:8000/docs)에서 직접 테스트 가능

---

# 🚀 2. 배포 방법

## 🌍 선택지 1: **클라우드 서버 (Ubuntu + Gunicorn + Nginx)**

### 2-1. 서버 환경 설정

```bash
sudo apt update && sudo apt install python3-pip python3-venv nginx
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2-2. `Gunicorn`으로 FastAPI 실행

```bash
pip install gunicorn
gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

### 2-3. `Nginx` 리버스 프록시 설정

`/etc/nginx/sites-available/aiagent.conf`:

```
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/aiagent.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## ☁️ 선택지 2: **Streamlit Cloud / Hugging Face Spaces / Railway**

* **Streamlit Cloud**: 유저 친화적이고 간단하지만 RAG 시스템은 제약 있음
* **Hugging Face Spaces (Gradio)**: 오픈소스 기반 데모에 적합
* **Railway.app / Render.com**: 서버리스 FastAPI 배포에 적합

---

## 🐳 선택지 3: Docker + AWS ECS / Azure / GCP

**Dockerfile 예시:**

```Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ai-agent .
docker run -p 8000:8000 ai-agent
```

→ AWS ECS, Fargate, Azure Container Apps 등에 배포 가능

---

# 📈 3. 확장 전략 (실제 금융 시스템으로 가기 위한 방향)

| 기능            | 확장 방향                             | 기술 예시                    |
| ------------- | --------------------------------- | ------------------------ |
| ✅ 인증          | 공동인증서 연동 or OAuth2                | 금융결제원, NICE 인증 API       |
| ✅ 데이터 수집      | 실제 오픈뱅킹 API 연동                    | 은행 API 스펙 기반             |
| ✅ 문서 검색       | PDF, Word, CSV 등도 벡터화             | PyMuPDF, Pandas + Chroma |
| ✅ 다국어         | 영어, 일본어 확장                        | LLM 번역 or Ko-EN 파라미터 튜닝  |
| ✅ 보안          | HTTPS, JWT 인증                     | FastAPI Security + TLS   |
| ✅ 사용자별 데이터 저장 | DB 연동 (SQLite → PostgreSQL)       | SQLAlchemy, Supabase 등   |
| ✅ 프론트엔드       | React + Tailwind or Streamlit 프론트 | REST API 연동              |

---

## 🔐 보안 고려사항

* LLM API 호출 시 사용자 식별 정보 제거
* 계좌/부채 정보는 **절대 저장 금지** or 암호화 필요
* 배포 시 HTTPS 필수
* 인증된 사용자만 접근 허용 (JWT 토큰 or OAuth2)

---

# ✅ 마무리 요약

| 항목      | 내용                                    |
| ------- | ------------------------------------- |
| 실행      | `uvicorn app:app` / `mock_api.py` 실행  |
| 벡터DB 구축 | `rag/build_vector_db.py`              |
| 질문 테스트  | `/chat` POST 요청                       |
| 배포      | Gunicorn + Nginx / Docker / Railway 등 |
| 확장성     | 인증, 실제 API, 보안, DB 연동까지 고려 가능         |

---
금융 AI Agent 프로젝트를 **Streamlit** 기반 데모 앱으로 전환한 코드를 제공하겠습니다.

이 데모 앱은 사용자 질문을 입력하면,

1. 계좌/부채 정보를 모킹 API에서 가져오고
2. 부채 분석 + 금융상품 추천을 실행하고
3. LLM을 통해 자연어 답변을 출력하는 구조입니다.

---

## ✅ 1. 프로젝트 구조 (Streamlit용)

```
financial_ai_agent/
├── streamlit_app.py         👈 ★ 메인 데모 앱
├── main_agent.py            ← 기존 LLM 오케스트레이터
├── (기타 모듈 동일)
│   ├── mock_server/
│   ├── modules/
│   ├── api_clients/
│   ├── rag/
│   ├── data/
├── requirements.txt
```

---

## ✅ 2. `streamlit_app.py` (Streamlit 데모 앱)

```python
import streamlit as st
from main_agent import handle_user_query

st.set_page_config(page_title="금융 AI 상담봇 💸", layout="centered")

st.title("📊 개인 금융 상담 AI Agent")
st.markdown("은행 API + AI 분석으로 부채 해결을 도와드립니다.")

# 사용자 입력
user_question = st.text_area("💬 궁금한 점을 입력하세요:", placeholder="예: 부채를 줄이려면 어떻게 해야 하나요?", height=120)

if st.button("AI에게 상담받기"):
    if not user_question.strip():
        st.warning("질문을 먼저 입력해주세요.")
    else:
        with st.spinner("AI가 분석 중입니다... ⏳"):
            try:
                answer = handle_user_query(user_question)
                st.success("✅ AI의 답변:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
```

---

## ✅ 3. `main_agent.py` (Streamlit에 맞게 비동기 제거)

> `FastAPI`에서는 `async def` 였지만, `Streamlit`에서는 `def`로 변경

```python
from api_clients.open_banking import get_accounts, get_loans
from modules.debt_analyzer import analyze_debt
from modules.financial_product_recommender import recommend_products
from modules.prompt_templates import debt_prompt
from langchain.chat_models import ChatOpenAI

def handle_user_query(user_input):
    accounts = get_accounts()
    loans = get_loans()
    income = sum(a["balance"] for a in accounts) * 0.05  # 잔액의 5%를 월 수입으로 가정
    analysis = analyze_debt(loans, income)
    product_suggest = recommend_products("부채 상환에 좋은 금융 상품 추천")
    prompt = debt_prompt(user_input, analysis, product_suggest)
    llm = ChatOpenAI(model="gpt-4")
    return llm.invoke(prompt)
```

---

## ✅ 4. 실행 방법

### 📦 필수 설치

```bash
pip install streamlit
```

또는 `requirements.txt`에 추가:

```
streamlit
```

---

### ▶️ 실행

```bash
streamlit run streamlit_app.py
```

→ 기본 브라우저에서 `http://localhost:8501` 자동 실행됨

---

## 🧪 데모 준비 체크리스트

| 항목                 | 필요 여부 | 설명                                             |
| ------------------ | ----- | ---------------------------------------------- |
| ✅ `mock_server` 실행 | 필요    | Flask로 API (`http://localhost:5000/`) 제공       |
| ✅ 벡터 DB 준비         | 필요    | `python rag/build_vector_db.py` 실행             |
| ✅ `OpenAI API 키`   | 필요    | LangChain에서 GPT-4 호출 시 사용                      |
| ✅ 샘플 문서            | 필요    | `data/product_docs/products.txt` (예금/대출 정보 포함) |

---

## 🎁 샘플 질문 예시

* "부채가 많아. 어떻게 줄일 수 있어?"
* "나에게 맞는 상환 계획을 알려줘"
* "예금이자 높은 상품도 추천해줘"
* "소득에 맞는 대출 상환 방법은?"

---

## 📌 선택 사항: Hugging Face Spaces에 배포

Streamlit 앱은 [Hugging Face Spaces](https://huggingface.co/spaces)에도 무료로 배포 가능합니다:

1. 프로젝트를 GitHub에 올리기
2. `requirements.txt` 포함하기
3. Hugging Face에서 `Create Space` → Streamlit 선택
4. `streamlit_app.py`가 자동으로 실행됩니다

---

## ✅ 정리

| 기능        | 설명                             |
| --------- | ------------------------------ |
| 사용자 인터페이스 | `Streamlit` 기반 대화형 질문창         |
| 백엔드 통합    | `main_agent.py`를 통해 전체 모듈 연결   |
| 질문 처리     | 모킹 API + RAG + GPT-4           |
| 배포        | 로컬, Hugging Face, Docker 모두 가능 |

---
아래에 요청하신 항목들을 모두 준비했습니다. Streamlit 앱에 그래프와 차트를 포함하고, 금융 문서 샘플, 예시 데이터, 배포 템플릿도 포함된 완성된 구조입니다.

---

## ✅ 포함된 항목

### 📄 1. 샘플 금융 문서 파일 (`products.txt`)

**파일 경로:** `data/product_docs/products.txt`

```txt
상품명: 스마트예금A
이율: 연 3.5%
조건: 1년 이상 예치, 월 100만원 이상
설명: 높은 이율을 제공하는 단기 예금 상품입니다.

상품명: 저축플랜B
이율: 연 2.1%
조건: 월 50만원 자동이체
설명: 소액 정기 저축자에게 적합한 상품입니다.

상품명: 부채통합론C
금리: 연 5.5%
설명: 여러 부채를 하나의 대출로 통합하여 상환 부담을 줄여줍니다.
```

---

### 📦 2. 예시 계좌/대출 모킹 데이터

**모킹 API (`mock_api.py`)가 제공하는 JSON 예시**

#### `/accounts` 응답

```json
{
  "accounts": [
    { "bank": "국민은행", "balance": 3000000 },
    { "bank": "우리은행", "balance": 1800000 }
  ]
}
```

#### `/loans` 응답

```json
{
  "loans": [
    { "type": "신용대출", "balance": 6000000, "min_payment": 250000 },
    { "type": "학자금대출", "balance": 2000000, "min_payment": 100000 }
  ]
}
```

---

### 📊 3. Streamlit에 **부채 상환 플랜 시각화 추가**

#### 📈 수정된 `streamlit_app.py` 예시:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main_agent import handle_user_query

st.set_page_config(page_title="금융 AI 상담봇 💸", layout="centered")
st.title("📊 개인 금융 상담 AI Agent with 그래프")
st.markdown("AI가 부채 상환 계획을 수립하고 금융 상품을 추천합니다.")

# 사용자 입력
user_question = st.text_area("💬 상담 질문을 입력하세요:", placeholder="예: 부채를 줄이는 방법이 궁금해요.", height=100)

if st.button("AI 분석 시작"):
    if not user_question.strip():
        st.warning("질문을 먼저 입력하세요.")
    else:
        with st.spinner("AI가 데이터를 분석 중입니다..."):
            try:
                result = handle_user_query(user_question, return_raw=True)
                response, plan_data = result["response"], result["plan"]

                st.success("✅ AI의 답변:")
                st.markdown(response)

                # 📊 부채 상환 플랜 시각화
                st.subheader("📉 부채 상환 플랜 차트")
                df = pd.DataFrame(plan_data)
                st.line_chart(df.set_index("month")[["remaining"]])

                st.bar_chart(df.set_index("month")[["payment"]])

            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
```

---

#### 📌 `main_agent.py`에서 raw 데이터 리턴 추가

```python
def handle_user_query(user_input, return_raw=False):
    accounts = get_accounts()
    loans = get_loans()
    income = sum(a["balance"] for a in accounts) * 0.05
    analysis = analyze_debt(loans, income)
    product_suggest = recommend_products("부채 상환에 적합한 금융 상품 추천")
    prompt = debt_prompt(user_input, analysis, product_suggest)
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(prompt)

    if return_raw:
        return {"response": response, "plan": analysis["plan"]}
    return response
```

---

### 🚀 4. Streamlit 배포용 템플릿 (Hugging Face Spaces, Streamlit Cloud 등)

#### `requirements.txt` (추가 설치 포함)

```
streamlit
openai
langchain
chromadb
requests
pandas
matplotlib
flask
```

#### `README.md` (예시)

````md
# 금융 AI Agent 💸

Streamlit 기반 개인 맞춤형 금융 상담 에이전트입니다.

## 기능
- 사용자의 질문을 받아 부채 현황 분석
- 상환 로드맵 제안 및 금융상품 추천
- 그래프 시각화 제공

## 실행 방법

```bash
streamlit run streamlit_app.py
````

## 데이터

* `mock_server/` → 테스트용 계좌/대출 정보
* `data/product_docs/products.txt` → 상품 설명서

```

---

## ☁️ Hugging Face Spaces 배포 시 주의사항

- Hugging Face에 `streamlit_app.py`와 `requirements.txt`를 함께 업로드
- LLM API 키는 환경 변수로 설정 (예: `OPENAI_API_KEY`)
- `data/` 디렉토리도 포함

---

## ✅ 요약

| 구성 요소 | 포함 여부 |
|-----------|-----------|
| 금융 문서 샘플 (`products.txt`) | ✅ |
| 계좌/부채 예시 JSON 응답 | ✅ |
| 그래프 포함된 Streamlit 데모 | ✅ |
| Hugging Face 배포 템플릿 | ✅ |
| 시각화 (선형, 막대 차트) | ✅ |

