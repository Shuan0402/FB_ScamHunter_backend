import requests

url = "http://127.0.0.1:8000/analyze"
data = {
    "ad_id": "123456789",
    "ad_context": "這是一則測試廣告內容",
    "ad_profile_name": "AI 測試粉專"
}

response = requests.post(url, json=data)
print(response.json())