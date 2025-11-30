import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import joblib

input_path = os.path.join("data", "processed", "churn_processed.csv")
output_train_path = os.path.join("data", "processed", "train.csv")
output_test_path = os.path.join("data", "processed", "test.csv")

def train():
    # 1. Load dữ liệu đã chuẩn bị ở bước trước
    print("🚀 Đang load dữ liệu train/test...")
    train_df = pd.read_csv(output_train_path)
    test_df = pd.read_csv(output_test_path)
    
    # Tách Feature (X) và Target (y)
    # Target là cột 'churn', các cột còn lại là Feature
    X_train = train_df.drop('churn', axis=1)
    y_train = train_df['churn']
    X_test = test_df.drop('churn', axis=1)
    y_test = test_df['churn']
    
    # --- BẮT ĐẦU MLFLOW ---
    # set_experiment giúp gom nhóm các lần chạy lại cho gọn
    mlflow.set_experiment("churn-prediction-baseline")
    
    with mlflow.start_run():
        # 2. Định nghĩa tham số Model
        params = {
            "C": 0.01,           # Regularization strength
            "solver": "liblinear",
            "max_iter": 1000
        }
        
        # Log tham số lên MLflow (để sau này nhớ mình đã chỉnh gì)
        mlflow.log_params(params)
        
        # 3. Train Model
        print("🧠 Đang training model...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # 4. Đánh giá Model
        predictions = model.predict(X_test)
        predict_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, predict_proba)
        
        print(f"📊 Kết quả: Accuracy={acc:.4f}, AUC={auc:.4f}")
        
        # Log chỉ số (Metrics) lên MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        
        # 5. Lưu Model vào MLflow
        # Giúp bạn có thể tải lại model này ở bất kỳ đâu
        mlflow.sklearn.log_model(model, "model")
        
        print("✅ Đã log model và metrics lên MLflow!")

    # 6. Lưu model ra folder models/ để dùng sau này (ví dụ: deploy API)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Đã lưu model local tại: {model_path}")

if __name__ == "__main__":
    train()

#mlflow ui
