import requests

url = "https://fb-scamhunter-backend.onrender.com/analyze"
data = {
    "ad_id": "123456789",
    "ad_context": "這是一則測試廣告內容",
    "ad_profile_name": "AI 測試粉專"
}

response = requests.post(url, json=data)
print(response.json())