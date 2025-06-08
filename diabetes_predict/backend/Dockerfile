# Gunakan base image Python yang kompatibel dengan TensorFlow di requirements.txt Anda
# Python 3.10 atau 3.11 direkomendasikan untuk TensorFlow versi terbaru.
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan cache Docker
COPY requirements.txt .

# Instal semua dependensi Python dari requirements.txt
# --no-cache-dir digunakan untuk mengurangi ukuran image
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek lainnya (main.py, model.h5, scaler.joblib, dll.)
# ke dalam direktori kerja (/app) di container.
COPY . .

# Ekspos port tempat aplikasi FastAPI akan berjalan (sesuai dengan CMD di bawah)
EXPOSE 8000

# Perintah untuk menjalankan aplikasi FastAPI menggunakan Uvicorn
# "--host 0.0.0.0" membuat aplikasi dapat diakses dari luar container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]