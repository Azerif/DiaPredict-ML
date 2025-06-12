# Menggunakan base image Python resmi
FROM python:3.10-slim

# Mengatur direktori kerja di dalam container
WORKDIR /app

# Install system dependencies yang diperlukan untuk TensorFlow
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Menyalin file requirements.txt dan menginstal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file aplikasi ke dalam container
COPY . .

# Mengekspos port 7860 (sesuai dengan konfigurasi di main.py)
EXPOSE 7860

# Menjalankan aplikasi menggunakan uvicorn (konsisten dengan main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]