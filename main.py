from fastapi import FastAPI
from schemas import AdData
from model import predict_category

app = FastAPI()

@app.post("/analyze")
async def analyze_ad_endpoint(data: AdData):
    result = predict_category(data)
    return {"ad_id": data.ad_id, "prediction": result}