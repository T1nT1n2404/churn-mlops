# Customer Churn MLOps Project

Dự án MLOps end-to-end dự đoán khả năng rời bỏ của khách hàng (Customer Churn). Dự án bao gồm quy trình từ xử lý dữ liệu, huấn luyện mô hình, theo dõi thí nghiệm với MLflow, đóng gói mô hình với Docker và triển khai API với FastAPI.

## 📂 Cấu trúc dự án

```
customer-churn-mlops/
├── data/
│   ├── raw/                # Dữ liệu thô (được quản lý bởi DVC)
│   └── processed/          # Dữ liệu đã qua xử lý
├── models/                 # Chứa model đã huấn luyện (.pkl)
├── notebooks/              # Jupyter Notebooks cho phân tích & thử nghiệm
├── src/                    # Source code chính
│   ├── app.py              # FastAPI application
│   ├── load_data.py        # Script load và sơ chế dữ liệu
│   ├── make_dataset.py     # Script xử lý features và split train/test
│   └── train.py            # Script huấn luyện và log MLflow
├── Dockerfile              # Cấu hình Docker image
├── requirements.txt        # Các thư viện Python cần thiết
└── README.md               # Tài liệu dự án
```

## 🚀 Cài đặt môi trường

1.  **Clone repository:**
    ```bash
    git clone <your-repo-url>
    cd customer-churn-mlops
    ```

2.  **Tạo môi trường ảo (Khuyên dùng Conda hoặc venv):**
    ```bash
    conda create -n churn-env python=3.10
    conda activate churn-env
    ```

3.  **Cài đặt dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Quy trình chạy (Workflow)

### 1. Chuẩn bị dữ liệu
Load dữ liệu thô và thực hiện sơ chế ban đầu:
```bash
python src/load_data.py
```
*Output: `data/processed/churn_processed.csv`*

Xử lý đặc trưng (Feature Engineering) và chia tập Train/Test:
```bash
python src/make_dataset.py
```
*Output: `data/processed/train.csv`, `data/processed/test.csv`*

### 2. Huấn luyện mô hình (Training)
Huấn luyện mô hình Logistic Regression và log kết quả vào MLflow:
```bash
python src/train.py
```
*Output: Model được lưu tại `models/model.pkl` và log metrics trên MLflow.*

### 3. Theo dõi thí nghiệm (MLflow)
Xem giao diện MLflow để so sánh các lần chạy:
```bash
mlflow ui
```
Truy cập: `http://127.0.0.1:5000`

## 🌐 Triển khai API (Deployment)

### Chạy Local với Uvicorn
```bash
uvicorn src.app:app --reload
```
Truy cập API docs: `http://127.0.0.1:8000/docs`

### Chạy với Docker

1.  **Build Docker Image:**
    ```bash
    docker build -t churn-api:v1 .
    ```

2.  **Run Container:**
    ```bash
    docker run -p 8000:8000 churn-api:v1
    ```

## 🧪 API Endpoints

-   `GET /`: Kiểm tra trạng thái API.
-   `POST /predict`: Dự đoán churn.

**Ví dụ Body JSON:**
```json
{
  "features": {
    "seniorcitizen": 0,
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
    "gender_Male": 0,
    "partner_Yes": 1,
    "dependents_Yes": 0,
    "phoneservice_Yes": 0,
    "multiplelines_No phone service": 1,
    "multiplelines_Yes": 0,
    "internetservice_Fiber optic": 0,
    "internetservice_No": 0,
    "onlinesecurity_No internet service": 0,
    "onlinesecurity_Yes": 0,
    "onlinebackup_No internet service": 0,
    "onlinebackup_Yes": 1,
    "deviceprotection_No internet service": 0,
    "deviceprotection_Yes": 0,
    "techsupport_No internet service": 0,
    "techsupport_Yes": 0,
    "streamingtv_No internet service": 0,
    "streamingtv_Yes": 0,
    "streamingmovies_No internet service": 0,
    "streamingmovies_Yes": 0,
    "contract_One year": 0,
    "contract_Two year": 0,
    "paperlessbilling_Yes": 1,
    "paymentmethod_Credit card (automatic)": 0,
    "paymentmethod_Electronic check": 1,
    "paymentmethod_Mailed check": 0
  }
}
```

## 🔧 Công nghệ sử dụng
-   **Python 3.10**
-   **Pandas, Scikit-learn**: Xử lý dữ liệu & Modeling.
-   **MLflow**: Quản lý vòng đời ML (Tracking, Models).
-   **DVC**: Quản lý phiên bản dữ liệu.
-   **FastAPI**: Xây dựng REST API hiệu năng cao.
-   **Docker**: Containerization.
