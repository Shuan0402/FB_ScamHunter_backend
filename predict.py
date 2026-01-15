import numpy as np
import pandas as pd
import xgboost as xgb
import joblib  # 用來載入 TF-IDF
import jieba
import re
import tldextract
import json
from urllib.parse import urlparse

class ScamDetector:
    def __init__(self, 
                 model_path="fraud_detection_nlp_model.json", 
                 vectorizer_path="tfidf_vectorizer.pkl"):
        
        print("⏳ 正在載入 XGBoost 模型...")
        # 載入 XGBoost 模型
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        print("⏳ 正在載入 TF-IDF Vectorizer...")
        # 載入訓練好的 TF-IDF 設定
        self.vectorizer = joblib.load(vectorizer_path)
        
        # 定義風險關鍵字 (必須跟訓練時一樣)
        self.risk_keywords = ["限時", "倒閉", "跑路", "加賴", "飆股", "免費"]
        
        print("✅ 模型載入完成！")

    # =========================================
    # 特徵工程函式 (必須與訓練時完全一致)
    # =========================================
    def _check_domain_mismatch(self, row):
        display_text = str(row.get('ad_caption', '')).lower().strip()
        real_url = str(row.get('ad_link_url', '')).lower().strip()
        if not display_text or not real_url: return 0 
        shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 'reurl', 'tr.ee', 'linktr.ee']
        for s in shorteners:
            if s in real_url: return 1
        try:
            extracted = tldextract.extract(real_url)
            real_main_domain = extracted.domain
            display_main_domain = display_text.split('.')[0]
            if display_main_domain in real_main_domain: return 0
            if "facebook" in real_main_domain or "instagram" in real_main_domain: return 0
            return 1
        except:
            return 1

    def _check_page_anomaly(self, row):
        try: likes = int(row.get('ad_page_like_count', 0))
        except: likes = 0
        try: post_likes = int(row.get('ad_like_count', 0))
        except: post_likes = 0
        
        if likes < 10: return 1
        if likes > 0 and post_likes > 0:
            if (post_likes / likes) > 10: return 1
        return 0

    def _count_pattern(self, text, pattern):
        return len(re.findall(pattern, str(text)))

    def _tokenize(self, text):
        text = str(text)
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        words = jieba.lcut(text)
        return " ".join(words)

    # =========================================
    # 核心預測流程
    # =========================================
    def predict(self, ad_json):
        """
        Input: 單筆廣告 JSON
        Output: (is_scam: bool, prob: float)
        """
        # 1. 將 JSON 轉為 DataFrame (單列)
        # 確保數值欄位有預設值
        if "ad_page_like_count" not in ad_json: ad_json["ad_page_like_count"] = 0
        if "ad_like_count" not in ad_json: ad_json["ad_like_count"] = 0
        
        df = pd.DataFrame([ad_json])
        
        df['ad_page_like_count'] = pd.to_numeric(df['ad_page_like_count'], errors='coerce').fillna(0)
        df['ad_like_count'] = pd.to_numeric(df['ad_like_count'], errors='coerce').fillna(0)
        
        # 2. 基礎特徵工程
        df['is_page_anomaly'] = df.apply(self._check_page_anomaly, axis=1)
        df['domain_spoofing'] = df.apply(self._check_domain_mismatch, axis=1)
        
        df['interaction_ratio'] = df.apply(
            lambda r: r['ad_like_count'] / r['ad_page_like_count'] if r['ad_page_like_count'] > 0 else 0, 
            axis=1
        )

        # 關鍵字計數
        def count_keywords(text):
            count = 0
            for kw in self.risk_keywords:
                if kw in str(text): count += 1
            return count
        df['risk_keyword_count'] = df['ad_context'].fillna("").apply(count_keywords)

        # 3. 內文結構特徵
        context = df['ad_context'].fillna("").astype(str)
        df['text_exclamation_count'] = context.apply(lambda x: self._count_pattern(x, r'!|！'))
        df['text_question_count'] = context.apply(lambda x: self._count_pattern(x, r'\?|？'))
        df['text_digit_count'] = context.apply(lambda x: self._count_pattern(x, r'\d+'))
        df['text_length'] = context.apply(len)
        emoji_pattern = r'[^\w\s,\.，。！!\?\?]'
        df['text_emoji_count'] = context.apply(lambda x: self._count_pattern(x, emoji_pattern))
        df['text_all_emoji_count'] = context.apply(lambda x: self._count_pattern(x, emoji_pattern))
        df['text_risk_emoji_count'] = df['text_all_emoji_count']

        # 4. NLP TF-IDF 特徵
        # 先斷詞
        tokenized_text = context.apply(self._tokenize)
        # 使用載入的 Vectorizer 進行轉換 (注意：這裡是用 transform，不是 fit_transform)
        tfidf_matrix = self.vectorizer.transform(tokenized_text)
        
        # 轉成 DataFrame
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{name}" for name in feature_names])
        
        # 5. 合併所有特徵
        # 定義正確的特徵順序 (必須跟訓練時完全一樣)
        base_features = [
            'ad_page_like_count', 'ad_like_count', 'interaction_ratio', 'is_page_anomaly', 
            'domain_spoofing', 'risk_keyword_count', 'text_exclamation_count', 
            'text_question_count', 'text_digit_count', 'text_length', 
            'text_risk_emoji_count', 'text_all_emoji_count', 
            'tfidf_健康', 'tfidf_優惠', 'tfidf_免費', 'tfidf_全館', 'tfidf_分享', 
            'tfidf_加入', 'tfidf_台灣', 'tfidf_回饋', 'tfidf_我們', 'tfidf_折起', 
            'tfidf_新年', 'tfidf_日本', 'tfidf_最高', 'tfidf_生活', 'tfidf_立即', 
            'tfidf_系列', 'tfidf_自己', 'tfidf_設計', 'tfidf_限量', 'tfidf_領取'
        ]
        
        # 合併數值特徵與 TF-IDF 特徵
        combined_df = pd.concat([df, tfidf_df], axis=1)
        final_df = combined_df[base_features]
        
        # 6. 轉為 DMatrix 供 XGBoost 使用
        dmatrix = xgb.DMatrix(final_df)
        
        # 7. 預測
        # XGBoost predict 回傳的是機率值 (0~1)
        prob = self.model.predict(dmatrix)[0]
        is_scam = bool(prob > 0.5) # 門檻值設為 0.5
        
        return is_scam, float(prob)