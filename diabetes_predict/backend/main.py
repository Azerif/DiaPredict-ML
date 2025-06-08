import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd
import joblib # Untuk memuat scaler
import numpy as np
import tensorflow as tf # Untuk memuat model .h5

# --- Konfigurasi Path Artefak ---
# Pastikan file-file ini ada di direktori yang sama dengan main.py
# atau sesuaikan path-nya jika berbeda.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GANTI NAMA FILE MODEL DAN SCALER SESUAI DENGAN MILIK ANDA
MODEL_H5_FILENAME = "keras_model.h5" # Ganti dengan nama file .h5 Anda
SCALER_FILENAME = "scaler.joblib" # Ganti dengan nama file .joblib scaler Anda

MODEL_H5_PATH = os.path.join(BASE_DIR, MODEL_H5_FILENAME)
SCALER_PATH = os.path.join(BASE_DIR, SCALER_FILENAME)

# --- 0. Definisi Variabel Global & Memuat Artefak ---
model = None
scaler = None
training_columns = []

try:
    # Memuat model Keras .h5
    if not os.path.exists(MODEL_H5_PATH):
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_H5_PATH}. Pastikan file ada dan nama sudah benar.")
    model = tf.keras.models.load_model(MODEL_H5_PATH)
    print(f"Model Keras '{MODEL_H5_FILENAME}' berhasil dimuat.")
    # Anda bisa uncomment baris berikut untuk melihat ringkasan model saat startup jika perlu
    # model.summary()

    # Memuat scaler (misalnya, dari scikit-learn, disimpan dengan joblib)
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"File scaler tidak ditemukan: {SCALER_PATH}. Pastikan file ada dan nama sudah benar.")
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler '{SCALER_FILENAME}' berhasil dimuat.")
    
    # !!! PENTING: Definisikan training_columns Anda di sini !!!
    # Daftar ini HARUS sama persis (nama dan urutan) dengan kolom yang digunakan 
    # untuk melatih model Keras Anda SETELAH semua proses preprocessing.
    # GANTI DAFTAR INI DENGAN DAFTAR KOLOM ANDA YANG SEBENARNYA!
    training_columns = [
        'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
        'blood_glucose_level', 'female', 'male', 'no_info_smoke',
        'current_smoke', 'ever_smoke', 'former_smoke', 'never_smoke',
        'no_current_smoke'
    ] 
    print(f"Daftar 'training_columns' telah ditentukan. Model mengharapkan {len(training_columns)} fitur.")
    if hasattr(scaler, 'n_features_in_') and len(training_columns) != scaler.n_features_in_:
        print(f"PERINGATAN: Jumlah 'training_columns' ({len(training_columns)}) tidak cocok dengan fitur yang diharapkan scaler ({scaler.n_features_in_}). Periksa kembali!")


except FileNotFoundError as e:
    print(f"KESALAHAN SAAT MEMUAT ARTEFAK: {e}")
    print("Pastikan file model (.h5) dan scaler (.joblib) ada di direktori yang benar dan nama file di skrip sudah sesuai.")
    # Aplikasi mungkin tidak berfungsi dengan benar jika artefak tidak dimuat
except Exception as e:
    print(f"KESALAHAN TIDAK DIKENALI SAAT MEMUAT ARTEFAK: {e}")
    import traceback
    traceback.print_exc()

# --- 1. Definisikan Model Input Data menggunakan Pydantic ---
# Model ini mendefinisikan struktur data yang akan dikirim oleh pengguna
class GenderEnum(str, Enum):
    FEMALE = "Female"
    MALE = "Male"

class SmokingHistoryEnum(str, Enum):
    NEVER = "never"
    NO_INFO = "No Info"
    CURRENT = "current"
    EVER = "ever"
    FORMER = "former"
    NOT_CURRENT = "not current"

class UserDataInput(BaseModel):
    age: int = Field(..., example=35, description="Usia pengguna dalam tahun.")
    hypertension: int = Field(..., example=0, ge=0, le=1, description="Riwayat hipertensi (0: Tidak, 1: Ya).")
    heart_disease: int = Field(..., example=0, ge=0, le=1, description="Riwayat penyakit jantung (0: Tidak, 1: Ya).")
    bmi: float = Field(..., example=22.5, description="Body Mass Index pengguna.")
    HbA1c_level: float = Field(..., example=5.0, description="Level HbA1c pengguna.")
    blood_glucose_level: int = Field(..., example=90, description="Level glukosa darah pengguna.")
    gender: GenderEnum = Field(..., example=GenderEnum.FEMALE, description="Jenis kelamin pengguna.")
    smoking_history: SmokingHistoryEnum = Field(..., example=SmokingHistoryEnum.NEVER, description="Riwayat merokok pengguna.")

    class Config:
        json_schema_extra = { # Ini untuk contoh di dokumentasi Swagger UI
            "example": {
                "age": 52,
                "hypertension": 0,
                "heart_disease": 0,
                "bmi": 27.32,
                "HbA1c_level": 6.6,
                "blood_glucose_level": 140,
                "gender": "Female",
                "smoking_history": "never"
            }
        }

# --- 2. Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="API Prediksi Diabetes (Model Keras)",
    description="API ini menerima data pengguna, melakukan preprocessing, dan mengembalikan prediksi risiko diabetes menggunakan model Keras (.h5).",
    version="1.2.0"
)

# --- 3. Definisikan Endpoint Prediksi ---
@app.post("/predict/", 
          summary="Prediksi Risiko Diabetes", 
          description="Masukkan data pengguna untuk mendapatkan prediksi risiko diabetes.")
async def predict_diabetes_endpoint(user_input: UserDataInput):
    """
    Endpoint untuk prediksi diabetes:
    - Menerima data pengguna sesuai skema `UserDataInput`.
    - Melakukan preprocessing.
    - Menggunakan model Keras yang sudah dimuat untuk membuat prediksi.
    - Mengembalikan probabilitas dan kelas prediksi.
    """
    if not model or not scaler or not training_columns:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Model, scaler, atau konfigurasi kolom tidak berhasil dimuat. Silakan periksa log server atau hubungi administrator."
        )

    try:
        # Langkah 1: Ubah data input Pydantic menjadi Pandas DataFrame
        user_data_dict = user_input.model_dump()
        user_df = pd.DataFrame([user_data_dict]) # DataFrame memerlukan list of dicts

        # --- Langkah 2: Preprocessing data pengguna (sesuai dengan proses training) ---
        
        # 2a: One-hot encode fitur kategorikal ('gender', 'smoking_history')
        # Pastikan prefix sesuai dengan yang mungkin dihasilkan dan akan di-rename
        user_one_hot = pd.get_dummies(user_df, columns=["gender", "smoking_history"], prefix=["gender", "smoking_history"])
        
        # 2b: Ganti nama kolom hasil one-hot encoding agar sesuai dengan `training_columns`
        rename_map = {
            f"gender_{GenderEnum.FEMALE.value}": "female",
            f"gender_{GenderEnum.MALE.value}": "male",
            f"smoking_history_{SmokingHistoryEnum.NO_INFO.value}": "no_info_smoke",
            f"smoking_history_{SmokingHistoryEnum.CURRENT.value}": "current_smoke",
            f"smoking_history_{SmokingHistoryEnum.EVER.value}": "ever_smoke",
            f"smoking_history_{SmokingHistoryEnum.FORMER.value}": "former_smoke",
            f"smoking_history_{SmokingHistoryEnum.NEVER.value}": "never_smoke",
            f"smoking_history_{SmokingHistoryEnum.NOT_CURRENT.value}": "no_current_smoke"
        }
        
        # Hanya rename kolom yang benar-benar ada setelah get_dummies
        actual_rename_map = {k: v for k, v in rename_map.items() if k in user_one_hot.columns}
        user_one_hot = user_one_hot.rename(columns=actual_rename_map)

        # 2c: Pastikan semua kolom di `training_columns` ada. Jika tidak, tambahkan dengan nilai 0.
        #    Lalu, susun ulang kolom sesuai urutan di `training_columns`.
        for col in training_columns:
            if col not in user_one_hot.columns:
                user_one_hot[col] = 0 # Tambahkan kolom yang hilang (karena kategori tidak ada di input ini)
        
        try:
            # Ambil hanya kolom yang ada di training_columns dan dalam urutan yang benar
            user_preprocessed = user_one_hot[training_columns]
        except KeyError as e:
            missing_expected_cols = [col for col in training_columns if col not in user_one_hot.columns]
            raise HTTPException(status_code=500, detail=f"Kesalahan penyelarasan kolom setelah one-hot encoding. Kolom yang hilang dari ekspektasi: {missing_expected_cols}. Error: {e}")

        # 2d: Scaling fitur numerik menggunakan scaler yang sudah di-load
        if user_preprocessed.shape[1] != scaler.n_features_in_:
             raise HTTPException(
                status_code=500,
                detail=f"Jumlah fitur setelah preprocessing ({user_preprocessed.shape[1]}) tidak cocok dengan yang diharapkan oleh scaler ({scaler.n_features_in_}). Periksa daftar 'training_columns'."
            )
        user_scaled_data = scaler.transform(user_preprocessed)
        
        # Hasil scaler.transform() adalah NumPy array, bisa langsung digunakan untuk model Keras
        input_array_for_model = user_scaled_data 
        # Jika Anda ingin kembali ke DataFrame (opsional, tidak wajib untuk Keras):
        # user_preprocessed_scaled_df = pd.DataFrame(user_scaled_data, columns=user_preprocessed.columns)
        # input_array_for_model = user_preprocessed_scaled_df.to_numpy()


        # --- Langkah 3: Prediksi menggunakan model Keras --
        # Model Keras.predict() mengembalikan NumPy array
        raw_predictions = model.predict(input_array_for_model)

        # Interpretasi output tergantung pada layer terakhir model Keras Anda:
        # Asumsi umum untuk klasifikasi biner:
        prob_diabetes = 0.0
        prob_no_diabetes = 0.0
        predicted_class = 0

        if raw_predictions.shape[1] == 1: # Kasus 1: Layer terakhir Sigmoid (1 output neuron) -> P(diabetes)
            prob_diabetes = float(raw_predictions[0][0])
            prob_no_diabetes = 1.0 - prob_diabetes
            predicted_class = 1 if prob_diabetes > 0.5 else 0
        elif raw_predictions.shape[1] == 2: # Kasus 2: Layer terakhir Softmax (2 output neuron) -> [P(tidak_diabetes), P(diabetes)]
            prob_no_diabetes = float(raw_predictions[0][0])
            prob_diabetes = float(raw_predictions[0][1])
            predicted_class = int(np.argmax(raw_predictions[0]))
        else:
            # Jika output model tidak sesuai dengan yang diharapkan
            raise HTTPException(
                status_code=500,
                detail=f"Format output model Keras tidak terduga. Shape: {raw_predictions.shape}. Harap periksa konfigurasi model Anda."
            )

        return {
            "message": "Prediksi berhasil diproses.",
            "input_data_received": user_input.model_dump(),
            "probabilities": {
                "tidak_diabetes": round(prob_no_diabetes, 4),
                "diabetes": round(prob_diabetes, 4)
            },
            "predicted_class": predicted_class,
            "interpretation": "0: Tidak Diabetes, 1: Diabetes"
        }

    except HTTPException: # Re-raise HTTPException agar FastAPI menanganinya
        raise
    except Exception as e:
        # Tangani error tidak terduga lainnya
        import traceback
        print(f"KESALAHAN INTERNAL SERVER SAAT PREDIKSI: {e}")
        traceback.print_exc() # Cetak traceback lengkap ke log server untuk debugging
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal pada server saat memproses prediksi: {str(e)}")

# --- 4. Endpoint root (untuk mengecek apakah API berjalan) ---
@app.get("/", summary="Status API", description="Endpoint root untuk mengecek apakah API Prediksi Diabetes (Keras) aktif.")
async def root():
    """
    Endpoint root sederhana untuk verifikasi bahwa API aktif dan berjalan.
    """
    status_model = "Model Keras dimuat" if model else "Model Keras GAGAL dimuat"
    status_scaler = "Scaler dimuat" if scaler else "Scaler GAGAL dimuat"
    status_columns = f"{len(training_columns)} kolom training dikonfigurasi" if training_columns else "Kolom training BELUM dikonfigurasi"
    
    return {
        "message": "Selamat datang di API Prediksi Diabetes (Model Keras)!",
        "status_api": "Aktif dan berjalan",
        "status_model": status_model,
        "status_scaler": status_scaler,
        "status_training_columns": status_columns,
        "dokumentasi_api": "/docs"
    }

# --- Cara menjalankan aplikasi ini (biasanya dari terminal) ---
# 1. Pastikan semua file (main.py, model.h5, scaler.joblib) ada di direktori yang sama.
# 2. Pastikan semua dependensi di requirements.txt sudah terinstal.
# 3. Jalankan dari terminal:
#    uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
#    Lalu akses API melalui browser atau Postman, misalnya:
#    - http://127.0.0.1:8000/ (untuk root)
#    - http://127.0.0.1:8000/docs (untuk dokumentasi interaktif Swagger UI)