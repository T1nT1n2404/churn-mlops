# 1. Chọn hệ điều hành Python 3.10 (Sửa dòng này)
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements và cài thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code và model
COPY src/ src/
COPY models/ models/

# 5. Mở cổng
EXPOSE 8000

# 6. Chạy server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]