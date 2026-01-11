# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import AdData
from model import predict_category
from predict import ScamDetector
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):  # 伺服器啟動時載入模型
    print("伺服器啟動中，正在載入模型...")
    try:
        ml_models["detector"] = ScamDetector() 
        print("模型載入成功")
    except Exception as e:
        print(f"模型載入失敗: {e}")
    
    yield
    
    ml_models.clear()
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_ad_endpoint(data: AdData):
    detector = ml_models.get("detector")
    if not detector:
        return {"status": "error", "message": "Model not loaded"}

    ad_dict = data.model_dump()
    is_scam, score = detector.predict(ad_dict)

    return {
        "ad_id": data.ad_id,
        "prediction": is_scam,
        "confidence_score": score,   # 相似度分數
        "status": "success"
    }