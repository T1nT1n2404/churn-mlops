from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import joblib
import numpy as np
import pickle

# 1. Khởi tạo App
app = FastAPI(title="Churn Prediction API")

print("🚀 Đang load model...")
try:
    model = joblib.load("models/model.pkl")
except FileNotFoundError:
    raise RuntimeError("❌ Không tìm thấy file models/model.pkl")

# 3. Định nghĩa dữ liệu đầu vào (Input Schema)
# Lưu ý: Model cần đúng thứ tự và số lượng cột như lúc train
# Để đơn giản cho bài học, ta dùng Dictionary linh hoạt
class CustomerData(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # 1. Chuyển dict thành DataFrame
        input_data = data.features
        df = pd.DataFrame([input_data])
        
        # --- ĐOẠN CODE QUAN TRỌNG ĐỂ SỬA LỖI 500 ---
        # Lấy danh sách cột mà model đã học lúc train
        if hasattr(model, "feature_names_in_"):
            expected_cols = model.feature_names_in_
            
            # Hàm reindex sẽ:
            # - Tự động thêm cột thiếu (điền số 0)
            # - Tự động bỏ cột thừa
            # - Sắp xếp lại đúng thứ tự
            df = df.reindex(columns=expected_cols, fill_value=0)
        # -------------------------------------------
        
        # 2. Dự đoán
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability[0]),
            "result": "Rời bỏ (Churn)" if prediction[0] == 1 else "Ở lại (Not Churn)"
        }
        
    except Exception as e:
        # Bắt lỗi và in ra để debug
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))