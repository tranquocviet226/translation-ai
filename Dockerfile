# Sử dụng image Python chính thức
FROM python:3.13-slim


# Cài đặt các công cụ cần thiết để cài đặt PyTorch và các thư viện khác
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements.txt vào container
COPY requirements.txt /app/

# Cài đặt các thư viện cần thiết từ requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . /app/

# Mở cổng mà API sử dụng
EXPOSE 5001

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
