from schemas import AdData

def predict_category(data: AdData) -> bool:
    print(f"正在處理廣告 ID: {data.ad_id}")
    return True