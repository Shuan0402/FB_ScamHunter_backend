import pandas as pd
import numpy as np
import xgboost as xgb
import os
import re
import jieba
import pickle
import tldextract
import json

class ScamDetector:
    def __init__(self, 
                 model_path="fraud_detection_nlp_model.json", 
                 tfidf_path="tfidf_vectorizer.pkl", 
                 keywords_path="risk_keywords.pkl"):
        
        # 1. 檢查檔案
        required_files = [model_path, tfidf_path, keywords_path]
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"❌ 找不到檔案: {f}，請確認是否已執行訓練程式。")

        # 2. 載入模型與工具
        # print("📥 載入模型中...") # 註解掉以保持 API 輸出乾淨
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        with open(tfidf_path, "rb") as f:
            self.vectorizer = pickle.load(f)
            
        with open(keywords_path, "rb") as f:
            self.dynamic_keywords = pickle.load(f)

        self.base_keywords = ["限時", "倒閉", "跑路", "加賴", "飆股", "免費", "領取", "老師"]
        self.all_risk_keywords = list(set(self.base_keywords + self.dynamic_keywords))

        self.base_features = [
            'ad_is_video', 'domain_spoofing', 'risk_keyword_count',
            'text_exclamation_count', 'text_question_count', 
            'text_digit_count', 'text_length', 'text_risk_emoji_count'
        ]
        self.risk_emojis = ["🚨", "⚠️", "❌", "🔥", "💣", "⚡", "💰", "💸", "💵", "💎", "💳", "📈", "📉", "👇", "👉", "🔗", "LINE"]

    # ... (中間的特徵工程函式保持不變，為節省篇幅省略) ...
    def _safe_bool(self, value):
        if isinstance(value, bool): return value
        return str(value).lower() in ["true", "1", "yes"]

    def _check_domain_mismatch(self, row):
        display_text = str(row.get('ad_caption', '')).lower().strip()
        real_url = str(row.get('ad_link_url', '')).lower().strip()
        if not real_url: return 0
        
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 'reurl', 'tr.ee', 'linktr.ee', 'line.me']
        for s in shorteners:
            if s in real_url: return 1
        try:
            extracted = tldextract.extract(real_url)
            real_domain = extracted.domain
            if real_domain in ["facebook", "instagram", "youtube", "google"]: return 0
            if "facebook" in display_text and "facebook" not in real_domain: return 1
            return 0
        except: return 1

    def _tokenize(self, text):
        text = re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
        return " ".join(jieba.lcut(text))

    def extract_features(self, ad_data):
        df = pd.DataFrame([ad_data])
        for col in ['ad_context', 'ad_link_url', 'ad_caption', 'ad_title']:
            if col not in df.columns: df[col] = ""
            else: df[col] = df[col].fillna("").astype(str)
        
        df['ad_is_video'] = df.get('ad_is_video', False).apply(self._safe_bool)
        df['domain_spoofing'] = df.apply(self._check_domain_mismatch, axis=1)

        def count_kw(text): return sum(1 for kw in self.all_risk_keywords if kw in str(text))
        df['risk_keyword_count'] = df['ad_context'].apply(count_kw)

        def count_pattern(text, pattern): return len(re.findall(pattern, str(text)))
        df['text_exclamation_count'] = df['ad_context'].apply(lambda x: count_pattern(x, r'!|！'))
        df['text_question_count'] = df['ad_context'].apply(lambda x: count_pattern(x, r'\?|？'))
        df['text_digit_count'] = df['ad_context'].apply(lambda x: count_pattern(x, r'\d+'))
        df['text_length'] = df['ad_context'].astype(str).apply(len)
        
        def count_emoji(text): return sum(str(text).count(e) for e in self.risk_emojis)
        df['text_risk_emoji_count'] = df['ad_context'].apply(count_emoji)

        tokenized_text = df['ad_context'].apply(self._tokenize)
        tfidf_matrix = self.vectorizer.transform(tokenized_text)
        tfidf_cols = [f"tfidf_{n}" for n in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols)

        final_df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        all_features = self.base_features + tfidf_cols
        for col in all_features:
            if col not in final_df.columns: final_df[col] = 0
                
        return final_df[all_features]

    def predict(self, json_data):
        """回傳: (is_fraud: bool, score: float, reasons: list)"""
        try:
            X_input = self.extract_features(json_data)
            dmatrix = xgb.DMatrix(X_input)
            score = self.model.predict(dmatrix)[0]
            is_fraud = score > 0.5
            
            reasons = [] # 這裡省略 reasons 生成邏輯，因為前端不需要，但保留變數
            return bool(is_fraud), float(score), reasons
        except Exception as e:
            print(f"Predict Error: {e}")
            return False, 0.0, []

# ==========================================
# 測試區 (模擬 API 行為)
# ==========================================
if __name__ == "__main__":
    predictor = ScamDetector()
    
    # 模擬前端傳來的資料
    mock_data = {
        "ad_id": "8888_test_ad",
        "ad_context": "限時免費領取飆股資訊！加賴領取：line.me/ti/p/123",
        "ad_link_url": "https://bit.ly/fake",
        "ad_caption": "facebook.com",
        "ad_is_video": False
    }
    
    # 1. 取得預測結果
    is_scam, score, _ = predictor.predict(mock_data)
    
    # 2. 格式化為前端要求的 JSON
    response = {
        "ad_id": str(mock_data.get("ad_id", "")), # str
        "prediction": is_scam,                    # bool
        "confidence_score": f"{score:.4f}",       # str
        "status": "success"
    }
    
    print("\n📦 前端回傳格式預覽:")
    print(json.dumps(response, indent=4, ensure_ascii=False))