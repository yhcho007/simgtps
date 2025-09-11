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

---
### 금융 AI 에이전트 코드 개선 방향 및 구조 보완

제공해드린 금융 AI 에이전트 코드는 기본적인 틀을 제공하지만, 실제 프로덕션 환경에서 사용하기에는 여러 보완점이 있습니다. 특히, 현재 코드는 사용자의 질문 의도를 파악하는 기능이 없고, 모든 요청에 대해 부채 분석과 상품 추천을 일률적으로 실행합니다. 또한, 각 모듈 간의 결합도가 높아 유연성이 떨어집니다.

아래에 **AI Agent로서의 기능과 확장성**을 높이기 위한 주요 개선 방향과 보강된 코드 구조를 제시합니다.

-----

### 1\. **AI 에이전트 기능 강화 (핵심)**

가장 중요한 개선점은 사용자의 질문 의도를 파악하여 적절한 도구(Tool)를 동적으로 호출하는 것입니다. 이 방식은 \*\*LangChain의 `AgentExecutor`와 `tools`\*\*를 활용하여 구현할 수 있습니다.

  * **`main_agent.py` 재구성**:
      * **도구(Tools) 정의**: `OpenBankingTool`, `DebtAnalyzerTool`, `ProductRecommenderTool` 등 각 기능을 독립적인 도구로 정의합니다.
      * **프롬프트 엔지니어링**: AI Agent가 어떤 질문에 어떤 도구를 사용해야 하는지 알려주는 프롬프트를 구성합니다.
      * **`AgentExecutor` 사용**: `AgentExecutor.from_agent_and_tools()`를 사용하여 사용자의 질문에 따라 도구를 실행할 에이전트를 생성합니다.

-----

### 2\. **모듈 분리 및 의존성 관리**

현재 코드는 `main_agent.py`가 모든 모듈을 직접 호출하는 구조입니다. 이를 **결합도가 낮은 구조**로 변경하여 각 모듈의 독립성을 높여야 합니다.

  * **`tools/` 디렉터리 추가**: API 클라이언트, 분석 모듈 등을 **`LangChain` 도구 형태로** 정의하는 파일들을 이 디렉터리에 모아 관리합니다.
  * **`config.py` 추가**: API 키, URL 등 환경 변수를 한 곳에서 관리하여 코드의 유지보수성을 높입니다.

-----

### 3\. **RAG (검색 증강 생성) 로직 개선**

`RAG`는 `LangChain` 체인(Chain)의 한 부분으로 통합하는 것이 더 효율적입니다.

  * **`modules/financial_product_recommender.py` 재구성**:
      * `recommend_products` 함수 내에서 `RetrievalQA` 체인을 구성하는 대신, \*\*별도의 도구(Tool)\*\*로 만들고 에이전트가 필요할 때 호출하도록 합니다.
      * `query_vector_db.py`의 역할은 `financial_product_recommender` 도구 내부에 통합하여 RAG 파이프라인의 응집도를 높입니다.

-----

### 4\. **보강된 프로젝트 구조 및 주요 파일 내용**

아래는 위에서 제시한 개선 사항들을 반영한 새로운 프로젝트 구조와 주요 파일 내용입니다.

#### **보강된 프로젝트 구조**

```
financial_ai_agent/
├── app.py                      # FastAPI 서버
├── main_agent.py               # AI Agent 실행 로직 (핵심)
├── config.py                   # 환경 변수, API 키 관리
├── tools/                      # LangChain 도구 정의 디렉터리
│   ├── open_banking_tools.py   # 오픈뱅킹 API 도구
│   └── financial_tools.py      # 부채 분석, 상품 추천 도구
├── api_clients/
│   └── open_banking.py
├── modules/
│   ├── debt_analyzer.py        # 핵심 비즈니스 로직
│   └── rag_retriever.py        # RAG 검색기 로직
├── mock_server/
│   └── mock_api.py
├── data/
│   ├── product_docs/
│   └── vector_store/
├── requirements.txt
└── README.md
```

#### **보강된 `main_agent.py`**

```python
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.agents import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from tools.open_banking_tools import get_accounts, get_loans
from tools.financial_tools import analyze_debt_tool, recommend_products_tool
from config import OPENAI_API_KEY

# 1. 도구(Tools) 정의
tools = [
    get_accounts, 
    get_loans, 
    analyze_debt_tool, 
    recommend_products_tool
]

# 2. AI Agent용 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 금융과 관련된 모든 것을 도와주는 친절한 금융 AI Agent입니다. 사용자의 질문에 맞춰 적절한 도구를 사용하거나 직접 답변하세요."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. LLM 설정
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0)

# 4. Agent 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def handle_user_query(user_input):
    """
    사용자 질문을 받아 AgentExecutor를 실행하고 응답을 반환합니다.
    """
    try:
        response = await agent_executor.ainvoke({"input": user_input})
        return response["output"]
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return "죄송합니다. 요청을 처리하는 데 문제가 발생했습니다. 다시 시도해 주세요."

```

-----

#### **새로운 `tools/open_banking_tools.py`**

```python
from langchain.tools import tool
from api_clients.open_banking import get_accounts as get_accounts_client
from api_clients.open_banking import get_loans as get_loans_client

@tool
def get_accounts():
    """사용자의 모든 은행 계좌 정보를 조회합니다. 잔액, 은행 이름 등을 확인할 수 있습니다."""
    return get_accounts_client()

@tool
def get_loans():
    """사용자의 모든 대출 정보를 조회합니다. 대출 유형, 잔액, 월 상환액 등을 확인할 수 있습니다."""
    return get_loans_client()
```

#### **새로운 `tools/financial_tools.py`**

```python
from langchain.tools import tool
from modules.debt_analyzer import analyze_debt as analyze_debt_logic
from modules.rag_retriever import query_rag_db
from langchain_community.chat_models import ChatOpenAI
from config import OPENAI_API_KEY

@tool
def analyze_debt_tool(income: float):
    """
    사용자의 소득 정보를 바탕으로 총 부채를 분석하고 월별 상환 계획을 수립합니다.
    Args:
        income (float): 사용자의 월 소득.
    """
    # 이 부분은 실제 대출 정보를 가져오는 로직으로 대체되어야 합니다.
    # 현재는 목업 데이터 사용
    loans = [
        {"type": "신용대출", "balance": 5000000},
        {"type": "주택담보대출", "balance": 10000000}
    ]
    analysis = analyze_debt_logic(loans, income)
    return analysis

@tool
def recommend_products_tool(query: str):
    """
    사용자 질문과 관련된 금융 상품을 추천합니다.
    Args:
        query (str): 사용자의 금융 상품에 대한 질문.
    """
    llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    
    # RAG 검색기를 호출하여 관련 문서 검색
    retrieved_docs = query_rag_db(query)
    
    # 검색된 문서를 바탕으로 LLM에게 응답 생성 요청
    prompt = f"""
    아래 금융 상품 정보를 참고하여 사용자 질문에 답변해주세요.
    사용자 질문: {query}
    ---
    참고 정보:
    {retrieved_docs}
    """
    return llm.invoke(prompt).content
```

이와 같이 에이전트를 구성하면, "내 계좌 잔액은 얼마야?"라는 질문에는 `get_accounts` 도구를 호출하고, "부채 상환 계획 좀 짜줘"라는 질문에는 `analyze_debt_tool`을 호출하는 등 **질문 의도에 맞는 동적인 동작**이 가능해집니다.

---------
`config.py`를 YAML 파일에서 설정을 읽어오는 방식으로 수정하고, OpenAI API의 무료 사용 방법을 안내해 드릴게요.

### 1\. `config.yml` 파일 생성

먼저, 설정 값을 담을 `config.yml` 파일을 프로젝트 루트 디렉터리에 생성하세요.

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
  model: "gpt-4o-mini"
```

  * **`api_key`**: 여기에 발급받은 OpenAI API 키를 입력합니다.
  * **`model`**: 비용 효율적이고 빠른 **`gpt-4o-mini`** 모델을 추천합니다. `gpt-4o`보다 훨씬 저렴합니다.

-----

### 2\. `config.py` 수정

`yaml` 라이브러리를 사용해 `config.yml` 파일에서 설정을 불러오도록 `config.py`를 수정합니다.

```python
import yaml
import os

def load_config():
    """Load configuration from config.yml."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# Load configuration
try:
    CONFIG = load_config()
    OPENAI_API_KEY = CONFIG['openai']['api_key']
    OPENAI_MODEL = CONFIG['openai']['model']
except FileNotFoundError as e:
    print(e)
    # Fallback or exit if config file is critical
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in config.yml or as an environment variable.")

if __name__ == "__main__":
    print(f"Loaded OpenAI API Key: {'*' * 10}") # Print masked key for security
    print(f"Loaded OpenAI Model: {OPENAI_MODEL}")

```

  * **`pyyaml` 라이브러리 설치**: `pip install pyyaml` 명령어를 실행하여 필요한 라이브러리를 설치해야 합니다.
  * **파일 경로**: `os.path.join(os.path.dirname(__file__), 'config.yml')`를 사용해 현재 파일(`config.py`)과 동일한 디렉터리에 있는 `config.yml` 파일을 안전하게 참조합니다.
  * **에러 핸들링**: `config.yml` 파일이 없거나 환경 변수가 설정되지 않았을 때를 대비한 예외 처리를 추가했습니다.

-----

### 3\. OpenAI 무료 API 키 얻는 방법 및 사용량 💰

OpenAI는 신규 사용자에게 일정량의 **무료 크레딧**을 제공하며, 이를 통해 API를 무료로 사용할 수 있습니다.

1.  **OpenAI 계정 가입**: [OpenAI 웹사이트](https://platform.openai.com/)에 접속하여 회원가입합니다.
2.  **API 키 발급**: 로그인 후, 오른쪽 상단의 **'Personal'** 메뉴에서 \*\*'View API keys'\*\*를 클릭합니다.
3.  **새 키 생성**: **'Create new secret key'** 버튼을 클릭하여 새로운 키를 생성합니다. 이 키는 **생성 시 한 번만 표시**되므로, 안전한 곳에 즉시 복사해 두어야 합니다.
4.  **키 사용**: 복사한 키를 앞서 만든 **`config.yml` 파일**의 `api_key` 필드에 붙여넣습니다.

#### **무료 사용량**

  * **가입 시 무료 크레딧**: 신규 계정에는 일반적으로 **5달러**의 무료 크레딧이 제공됩니다. 이 크레딧은 **발급일로부터 3개월 동안** 유효합니다.
  * **`gpt-4o-mini` 모델 기준 사용량**:
      * **입력 (Input)**: 100만 토큰 당 **$0.15**
      * **출력 (Output)**: 100만 토큰 당 **$0.60**
  * **5달러 크레딧으로 얼마나 사용할 수 있을까?**:
      * 평균적으로 약 **1,000만\~1,500만 토큰**을 사용할 수 있습니다.
      * 이는 채팅 메시지 기준으로 **수십만 건**의 질문과 답변을 처리할 수 있는 양으로, 학습 및 소규모 테스트에는 충분합니다.
      * 1 토큰은 한글 약 0.5\~0.7 글자 또는 영문 한 단어 정도의 양입니다.

무료 크레딧이 소진되거나 만료되면, 추가적인 결제 수단을 등록하지 않는 한 API 호출이 중단되어 **과금될 걱정 없이** 안전하게 사용할 수 있습니다.

-----------------------------------
좋습니다 👍 그럼 완전히 **OpenAI API 안 쓰고** → **로컬(혹은 HuggingFace 공개 모델)** 기반으로

* **임베딩(HuggingFaceEmbeddings)**
* **LLM(HuggingFace 모델)**
  두 개 다 교체하는 게 맞습니다.

---

## 🔎 한국어 잘 이해하는 오픈소스 모델

한글을 잘 처리하려면 영어 중심 모델(flan-t5 등) 말고, **한국어 특화된 모델**을 쓰는 게 좋아요.
HuggingFace에서 많이 쓰는 모델 몇 가지:

* `beomi/KcELECTRA-base` → 한국어 자연어 이해에 강점
* `snunlp/KR-BERT-char16424` → BERT 계열, 문장 분류/이해
* `beomi/KoAlpaca-Polyglot-12.8B` → LLaMA 기반 한국어 파인튜닝 (성능 좋음, GPU 필요)
* `nlpai-lab/kullm-polyglot-5.8b` → 한국어 대화 특화 LLM

👉 개발 테스트라면 **가볍고 무료 CPU에서도 돌릴 수 있는 모델** 추천:

* `"beomi/KcELECTRA-base"` (이해 중심, 답변 생성엔 약함)
* `"beomi/KoAlpaca-Polyglot-12.8B"` (자연스러운 대화, 하지만 GPU 필요)

---

## ✅ `main_agent.py` (한국어 모델 적용 예시)

```python
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.open_banking_tools import get_accounts, get_loans
from tools.financial_tools import analyze_debt_tool, recommend_products_tool

# 1. 도구(Tools) 정의
tools = [
    get_accounts,
    get_loans,
    analyze_debt_tool,
    recommend_products_tool
]

# 2. AI Agent용 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 금융과 관련된 모든 것을 도와주는 친절한 금융 AI Agent입니다. "
               "사용자의 질문에 맞춰 적절한 도구를 사용하거나 직접 답변하세요."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. HuggingFace 로컬 LLM 설정 (한국어 모델)
from langchain_community.llms import HuggingFaceHub

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("⚠️ HUGGINGFACEHUB_API_TOKEN 환경변수를 설정해주세요.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN

# 👉 CPU 가능 모델: beomi/KoAlpaca-Polyglot-12.8B (권장: GPU 환경)
# 👉 가벼운 CPU 모델: beomi/KcELECTRA-base (속도 빠름, 성능은 제한적)
llm = HuggingFaceHub(
    repo_id="beomi/KoAlpaca-Polyglot-12.8B",  
    model_kwargs={"temperature": 0.3, "max_length": 512}
)

# 4. Agent 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def handle_user_query(user_input):
    """
    사용자 질문을 받아 AgentExecutor를 실행하고 응답을 반환합니다.
    """
    try:
        response = await agent_executor.ainvoke({"input": user_input})
        return response["output"]
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return "죄송합니다. 요청을 처리하는 데 문제가 발생했습니다. 다시 시도해 주세요."
```

---

## 🚀 실행 준비

1. HuggingFace 토큰 발급: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```
2. 설치:

   ```bash
   pip install sentence-transformers
   pip install transformers accelerate
   ```

---

## ⚡️ 정리

* 개발 단계:

  * **임베딩** → `sentence-transformers/all-MiniLM-L6-v2` (빠르고 가볍고 한국어도 무난)
  * **LLM** → `beomi/KoAlpaca-Polyglot-12.8B` (대화형, 한국어 잘함)

* 운영 단계:

  * 필요시 OpenAI GPT-4o 등으로 교체

---
한국어 잘 이해하는 오픈소스 모델

한글을 잘 처리하려면 영어 중심 모델(flan-t5 등) 말고, 한국어 특화된 모델을 쓰는 게 좋아요.
HuggingFace에서 많이 쓰는 모델 몇 가지:

beomi/KcELECTRA-base → 한국어 자연어 이해에 강점

snunlp/KR-BERT-char16424 → BERT 계열, 문장 분류/이해

beomi/KoAlpaca-Polyglot-12.8B → LLaMA 기반 한국어 파인튜닝 (성능 좋음, GPU 필요)

nlpai-lab/kullm-polyglot-5.8b → 한국어 대화 특화 LLM

👉 개발 테스트라면 가볍고 무료 CPU에서도 돌릴 수 있는 모델 추천:

"beomi/KcELECTRA-base" (이해 중심, 답변 생성엔 약함)

"beomi/KoAlpaca-Polyglot-12.8B" (자연스러운 대화, 하지만 GPU 필요)

---
완료했어 — 전체 데모 프로젝트 코드를 캔버스(‘Financial Ai Agent Demo’)에 넣어뒀어. ✅

빠른 요약과 다음 단계:

1. 캔버스에서 코드 전체(파일별)가 보일 거야 — `mock_server`, `app.py`, `main_agent.py`, `rag/build_vector_db.py`, `modules/`, `tools/`, `api_clients/` 등 포함.
2. 실행 순서(캔버스 상단에도 있음):

   * 패키지 설치: `pip install -r requirements.txt`
   * Mock API 실행: `uvicorn mock_server.mock_api:app --reload --port 8001`
   * Vector DB 생성: `python -m rag.build_vector_db`
   * 데모 서버 실행: `uvicorn app:app --reload --port 8000`
   * 데모 호출: `curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"input":"내 집을 마련하고 싶어. 내집 마련 계획을 세워줘"}'`

주의/참고:

* LLM(문장 다듬기)은 HuggingFace 토큰이 있어야 작동하고, 로컬 CPU에서 무거운 모델은 느릴 수 있어. 데모는 LLM 없어도 로직(데이터 조회 → 계획 생성)으로 동작합니다.
* `rag/build_vector_db.py`는 기존 `data/vector_store`를 삭제 후 재생성하도록 구현했습니다.
* 필요하면 LLM 모델을 한국어 생성에 더 적합한 모델로 바꿔드릴게요 (GPU 없을 경우 속도/품질 고려 필요).

이제 원하는 걸 해줘:

* 바로 실행해볼래? (문제 생기면 에러 로그랑 함께 알려줘)
* 아니면 LLM 출력 형태(한국어 자연어 요약 샘플), UI(간단한 웹 페이지) 등 추가로 만들어줄까?

# financial\_ai\_agent — Demo prototype

프로젝트 전체 코드(개발용, 로컬 CPU 환경, Mock API 포함). 아래 파일들이 포함되어 있습니다. 각 파일은 실행 가능한 최소 데모 수준으로 구성되어 있습니다.

---

## 파일 목록 / 사용법 요약

1. `mock_server/mock_api.py` — 로컬 Mock API (FastAPI)
2. `config.py` — 환경변수 로드(간단)
3. `rag/build_vector_db.py` — product 문서로부터 Vector DB 생성 (중복 제거)
4. `api_clients/open_banking.py` — Mock API 클라이언트 (requests 기반)
5. `modules/debt_analyzer.py` — 핵심 비즈니스 로직: 내집마련 계획 계산
6. `modules/rag_retriever.py` — RAG용 벡터스토어 로더 (optional)
7. `tools/financial_tools.py` — 추천/계획 생성 유틸 (LLM 없이 로직 수행)
8. `main_agent.py` — orchestration: Mock API 조회 → 분석 → LLM(문장 다듬기)
9. `app.py` — FastAPI 서버: `/chat` 엔드포인트
10. `requirements.txt` — 필요한 패키지 목록
11. `data/product_docs/products.txt` — 상품 문서 (제공하신 내용 그대로)
12. `README.md` — 실행 가이드

---

## 실행 순서 (요약)

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. HuggingFace 토큰(LLM 사용 시, 권장):

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

(참고: 데모는 LLM 단계가 선택적입니다. LLM이 없더라도 로직 데모는 동작합니다.)

3. Mock API 실행 (새 터미널):

```bash
uvicorn mock_server.mock_api:app --reload --port 8001
```

4. Vector DB 생성 (상품 문서에서):

```bash
python -m rag.build_vector_db
```

5. 서버 실행 (새 터미널):

```bash
uvicorn app:app --reload --port 8000
```

6. 데모 요청 예시 (curl)

```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"input": "내 집을 마련하고 싶어. 내집 마련 계획을 세워줘"}'
```

---

---

# 코드: `mock_server/mock_api.py`

```python
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
```

---

# 코드: `config.py`

```python
import os

def load_config():
    # 데모용 간단한 config
    return {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY", ""), "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini")},
        "huggingface": {"token": os.getenv("HUGGINGFACEHUB_API_TOKEN", "")}
    }
```

---

# 코드: `rag/build_vector_db.py`

```python
import os
import shutil
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def build_vector_db():
    persist_dir = os.path.join("data", "vector_store")

    # 기존 DB 삭제 (중복 방지)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("🗑️ 기존 Vector DB 삭제 완료")

    # 문서 로드
    loader = TextLoader(os.path.join("data", "product_docs", "products.txt"), encoding="utf-8")
    docs = loader.load()

    # 다국어 한국어 지원 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 새로운 벡터 DB 생성
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("✅ 새로운 Vector DB build success (multilingual embeddings 사용)")


if __name__ == "__main__":
    build_vector_db()
```

---

# 코드: `api_clients/open_banking.py`

```python
import requests

BASE = "http://localhost:8001"  # mock server


def get_accounts():
    r = requests.get(f"{BASE}/accounts")
    r.raise_for_status()
    return r.json()


def get_savings():
    r = requests.get(f"{BASE}/savings")
    r.raise_for_status()
    return r.json()


def get_loans():
    r = requests.get(f"{BASE}/loans")
    r.raise_for_status()
    return r.json()


def get_subscription_score():
    r = requests.get(f"{BASE}/subscription_score")
    r.raise_for_status()
    return r.json()
```

---

# 코드: `modules/debt_analyzer.py`

```python
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
```

---

# 코드: `modules/rag_retriever.py` (optional)

```python
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


def load_retriever(persist_dir="data/vector_store"):
    if not os.path.exists(persist_dir):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever
```

---

# 코드: `tools/financial_tools.py`

```python
from api_clients.open_banking import get_accounts, get_savings, get_loans, get_subscription_score
from modules.debt_analyzer import plan_home_saving


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
```

---

# 코드: `main_agent.py`

```python
import os
from config import load_config
from tools.financial_tools import build_home_plan

# Optional: LLM로 최종 문장 다듬기 (있으면 더 자연스러움)
try:
    from langchain_community.llms import HuggingFaceHub
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if HF_TOKEN:
        llm = HuggingFaceHub(repo_id="beomi/KcELECTRA-base", model_kwargs={"temperature": 0.2, "max_length": 512})
    else:
        llm = None
except Exception:
    llm = None


def generate_response(user_input: str):
    # 데모 트리거
    if "내 집" in user_input or "내집" in user_input:
        result = build_home_plan()
        # LLM이 있으면 요약/문장 다듬기
        if llm:
            prompt = f"사용자 요청: {user_input}\n\n재무정보: {result['financials']}\n\n추천 계획: {result['plan']}\n\n위 정보를 바탕으로 한국어 자연스러운 요약 답변을 만들어줘."
            try:
                out = llm(prompt)
                return out
            except Exception as e:
                print("LLM 호출 실패:", e)
                # fallback: 구조화된 텍스트
        # LLM 없을 때 반환
        return {
            "text": "내집마련 계획(데모):",
            "financials": result["financials"],
            "plan": result["plan"]
        }
    else:
        return {"text": "죄송합니다. 현재 데모는 '내 집' 관련 시나리오만 지원합니다."}
```

---

# 코드: `app.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from main_agent import generate_response

app = FastAPI()

class ChatRequest(BaseModel):
    input: str

@app.post("/chat")
async def chat(req: ChatRequest):
    return generate_response(req.input)
```

---

# 코드: `requirements.txt`

```
fastapi
uvicorn
requests
langchain
langchain-embeddings
langchain-vectorstores
chromadb
sentence-transformers
transformers
langchain-community
```

---

# 데이터: `data/product_docs/products.txt`

```text
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

# README.md

```markdown
# financial_ai_agent Demo

로컬 CPU에서 동작하는 개발용 데모입니다. Mock API를 사용하여 계좌/저축/청약 점수 등을 조회하고 간단한 내집마련 계획을 생성합니다.

실행 방법은 문서 상단의 실행 순서를 참고하세요.
```

---

*파일 전체를 캔버스에 넣어두었습니다. 코드 확인 후 실행해보시고, 수정하거나 기능 추가를 원하시면 바로 반영해드릴게요.*




