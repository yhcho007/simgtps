# tester.py
import requests
import json

url = "http://localhost:8000/chat"
payload = {"input": "내 집을 마련하고 싶어. 내집 마련 계획을 세워줘"}

try:
    # json=payload 를 쓰면 requests가 Content-Type: application/json 을 자동으로 설정합니다.
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
else:
    # 가능한 경우 JSON으로 파싱해서 보기 좋게 출력
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except ValueError:
        # JSON이 아니면 그냥 텍스트 출력
        print(resp.text)
