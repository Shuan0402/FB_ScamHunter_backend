# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import AdData
from model import predict_category

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_ad_endpoint(data: AdData):
    result = predict_category(data)
    return {
        "ad_id": data.ad_id,
        "prediction": result,
        "status": "success"
    }