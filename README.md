# FB_ScamHunter_backend

FB Scam Hunter 是一個在 AIS3 Hackathon 中開發的即時防詐標記工具後端系統。本專案核心為提供高效能的 API 服務，接收來自瀏覽器擴充元件的廣告內容，並透過機器學習模型即時辨識 Facebook 上的詐騙廣告。

## 功能簡介
### 模型辨識與分析
- **即時詐騙判定**：接收前端擷取的廣告文字與內容，透過訓練好的 XGBoost 模型進行風險判定。
- **特徵工程處裡**：包含 NLP 語意分析、內文結構特徵提取、以及廣告連結與顯示網域的一致性檢查。
- **手動查驗支援**：提供 API 接口支援使用者手動輸入連結進行內文查驗。

### 模型成果 (Model 5)
- **高準確度**：模型準確度 (Accuracy) 達 0.92，精確率 (Precision) 達 0.93。
- **特徵重要性分析**：主要判定依據包含文字長度 (text_length)、數字出現頻率 (text_digit_count) 及風險關鍵字統計 (risk_keyword_count)。

## 使用技術
### 後端框架
- **FastAPI**：作為 API Gateway，負責處理高併發的請求與回應。

### 機器學習與數據處理
- **XGBoost Classifier**：核心分類模型，用於判斷廣告是否為惡意詐騙。
- **Feature Engineering**：包含 NLP 技術與圖像辨識輔助分析。

### 數據來源
- **詐騙數據**：由 AIS3 Hackathon 提供，共計 5501 筆詐騙廣告資料。
- **正常數據**：取自 Meta 廣告檔案庫之正常廣告。

## 系統架構概述

```css=
server/
├── api/
│   ├── main.py          # FastAPI 主程式入口
│   └── endpoints/       # API 路由 (判斷、查驗)
│
├── models/
│   ├── xgboost_model.bin # 訓練完成之 XGBoost 模型檔
│   └── processor.py      # 特徵工程與 NLP 處裡邏輯
│
├── core/
│   ├── config.py         # 系統設定
│   └── security.py       # 安全檢查邏輯
│
└── requirements.txt      # 專案依賴套件
```

## 運作流程
1. 前端擴充元件存取 Facebook 頁面 DOM 結構，判定「贊助」字樣擷取廣告 。
2. 前端發送 API Request 將內容傳送至本後端 。
3. 後端進行 Feature Engineering (如 NLP 分析、風險詞偵測) 。
4. XGBoost Classifier 進行分類並回傳 Verdict (JSON) 結果 。
5. 前端根據回傳結果，於高風險廣告處標記紅框與提醒標籤 。

## 未來可擴充項目
1. 多平台支援：擴充後端模型以支援 Instagram 或 TikTok 的詐騙偵測 。
2. 即時黑名單同步：整合 GASA 或 Whoscall 的即時詐騙資料庫 。
3. 圖像 OCR 強化：提升廣告附圖中文字的辨識精準度 。

## 授權
本專案為 AIS3 Hackathon 競賽作品，由隊伍「隊名沒想法」維護 。
