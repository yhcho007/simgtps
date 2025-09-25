import requests

url = "http://localhost:8000/chat"
header = { "Content-Type" : 'application/json'}
data = {"message": "안녕하세요"}
response = requests.post(url=url, headers=header, data=data)
print(f"response:{response.json()}")

