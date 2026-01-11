import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

class ScamDetector:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2", 
                 center_path="scam_center.npy", 
                 threshold_path="threshold.txt"):
        
        print("正在載入 Sentence-BERT 模型...")
        self.model = SentenceTransformer(model_name)
        
        print("正在載入詐騙特徵中心...")
        self.scam_center = np.load(center_path)
        
        with open(threshold_path, "r") as f:
            self.threshold = float(f.read().strip())
            
        print(f"模型載入完成！(Threshold: {self.threshold:.4f})")

    def _ad_to_text(self, ad):
        """
        將 JSON 轉為結構化文字，邏輯必須跟訓練時一模一樣
        """
        parts = []
        if ad.get("ad_profile_name"):
            parts.append(f"[PAGE] {ad['ad_profile_name']}")
        if ad.get("ad_context"):
            parts.append(f"[CONTENT] {ad['ad_context']}")
        if ad.get("ad_cta_text"):
            parts.append(f"[CTA] {ad['ad_cta_text']}")
        if ad.get("ad_caption"):
            parts.append(f"[CAPTION] {ad['ad_caption']}")
        if ad.get("ad_page_categories"):
            # 處理可能是 list 或 string 的情況
            cats = ad["ad_page_categories"]
            if isinstance(cats, list):
                cats = " ".join(cats)
            parts.append(f"[CATEGORY] {cats}")
            
        return " ".join(parts)

    def predict(self, ad_json):
        """
        Input: 單筆廣告 JSON
        Output: (is_scam: bool, score: float, details: dict)
        """
        text = self._ad_to_text(ad_json)
        vector = self.model.encode([text], normalize_embeddings=True)
        similarity = cosine_similarity(vector, self.scam_center)[0][0]
        is_scam = bool(similarity >= self.threshold)
        
        return is_scam, float(similarity)