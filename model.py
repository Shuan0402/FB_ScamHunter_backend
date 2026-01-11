from schemas import AdData

def predict_category(data: AdData) -> bool:
    # 這裡可以自由存取 data.ad_id, data.ad_context 等
    print(f"正在處理廣告 ID: {data.ad_id}")
    return True