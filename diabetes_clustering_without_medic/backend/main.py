import pandas as pd
import numpy as np
import tensorflow as tf
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib

app = FastAPI(title="Diabetes Cluster Prediction API", version="1.1.0")

# --- Konfigurasi ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessors.joblib')

# Mapping input ke format model
SMOKING_MAPPING = {
    # Bahasa Indonesia
    "tidak": "never_smoke", "tidak_pernah": "never_smoke",
    "mantan": "former_smoke", "bekas": "former_smoke",
    "aktif": "current_smoke", "merokok": "current_smoke",
    "tidak_lagi": "no_current_smoke", "tidak_diketahui": "no_info_smoke",
    "pernah": "ever_smoke",
    # English pendek
    "never": "never_smoke", "former": "former_smoke", "current": "current_smoke",
    "not current": "no_current_smoke", "no info": "no_info_smoke", "ever": "ever_smoke",
    # Format asli
    "never_smoke": "never_smoke", "former_smoke": "former_smoke",
    "current_smoke": "current_smoke", "no_current_smoke": "no_current_smoke",
    "ever_smoke": "ever_smoke", "no_info_smoke": "no_info_smoke"
}

GENDER_MAPPING = {
    # Bahasa Indonesia
    "pria": "Male", "wanita": "Female", "laki-laki": "Male", "perempuan": "Female",
    "cowok": "Male", "cewek": "Female",
    # English
    "male": "Male", "female": "Female", "man": "Male", "woman": "Female",
    "m": "Male", "f": "Female", "Male": "Male", "Female": "Female"
}

CLUSTER_NAMES = {
    0: 'Populasi Lansia dengan Riwayat Kesehatan Kompleks',
    1: 'Anak dan Remaja dalam Kondisi Fisik Optimal',
    2: 'Dewasa dengan Pola Hidup Kurang Sehat',
    3: 'Dewasa Muda Aktif dan Relatif Sehat'
}

# Global variables
predictor = None
gender_categories = []
smoking_categories = []

# --- Models ---


class PredictionInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    smoking_history: str

    @validator('hypertension', 'heart_disease')
    def check_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError('Must be 0 or 1')
        return v

    @validator('gender')
    def normalize_gender(cls, v):
        v_clean = str(v).strip().lower()
        for key, value in GENDER_MAPPING.items():
            if v_clean == key.lower():
                return value
        raise ValueError(
            f'Invalid gender. Accepted: {list(GENDER_MAPPING.keys())}')

    @validator('smoking_history')
    def normalize_smoking(cls, v):
        v_clean = str(v).strip().lower()
        for key, value in SMOKING_MAPPING.items():
            if v_clean == key.lower():
                return value
        raise ValueError(
            f'Invalid smoking_history. Accepted: {list(SMOKING_MAPPING.keys())}')


class PredictionOutput(BaseModel):
    predicted_cluster: int
    cluster_name: str
    probabilities: list[float]
    confidence: float


class StatusOutput(BaseModel):
    status: str
    model_loaded: bool
    preprocessors_loaded: bool
    accepted_gender_inputs: list[str]
    accepted_smoking_inputs: list[str]
    cluster_names: dict

# --- Predictor Class ---


class ClusterPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")

        self.label_encoder = None
        self.onehot_encoder = None
        self.scaler = None
        self.numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi']

    def set_preprocessors(self, preprocessors):
        self.label_encoder = preprocessors['label_encoder']
        self.onehot_encoder = preprocessors['onehot_encoder']
        self.scaler = preprocessors['scaler']

        # Get feature names
        try:
            self.onehot_features = self.onehot_encoder.get_feature_names_out([
                                                                             'smoking_history'])
        except:
            num_cats = len(self.onehot_encoder.categories_[0])
            self.onehot_features = [
                f"smoking_history_cat{i}" for i in range(num_cats)]

        self.feature_order = ['gender'] + \
            self.numeric_features + list(self.onehot_features)
        print(f"✅ Preprocessors loaded. Features: {len(self.feature_order)}")

    def predict(self, gender, age, hypertension, heart_disease, bmi, smoking_history):
        # Create DataFrame
        data = pd.DataFrame({
            'gender': [gender], 'age': [float(age)], 'hypertension': [int(hypertension)],
            'heart_disease': [int(heart_disease)], 'bmi': [float(bmi)],
            'smoking_history': [smoking_history]
        })

        # Preprocess
        data['gender'] = self.label_encoder.transform(data['gender'])
        smoking_encoded = self.onehot_encoder.transform(
            data[['smoking_history']])
        smoking_df = pd.DataFrame(
            smoking_encoded, columns=self.onehot_features, index=data.index)
        data = pd.concat(
            [data.drop(columns=['smoking_history']), smoking_df], axis=1)
        data[self.numeric_features] = self.scaler.transform(
            data[self.numeric_features])

        # Predict
        X = data[self.feature_order].values.astype(np.float32)
        prediction = self.model.predict(X)
        predicted_cluster = np.argmax(prediction[0])
        probabilities = prediction[0]

        return predicted_cluster, probabilities

# --- Startup ---


@app.on_event("startup")
async def startup():
    global predictor, gender_categories, smoking_categories
    print("🚀 Starting application...")

    try:
        # Load model
        predictor = ClusterPredictor(MODEL_PATH)

        # Load preprocessors
        if not os.path.exists(PREPROCESSOR_PATH):
            raise RuntimeError(
                f"Preprocessor file not found: {PREPROCESSOR_PATH}")

        preprocessors = joblib.load(PREPROCESSOR_PATH)
        predictor.set_preprocessors(preprocessors)

        # Set global categories
        gender_categories = list(predictor.label_encoder.classes_)
        smoking_categories = list(predictor.onehot_encoder.categories_[0])

        print(f"✅ Ready! Gender: {gender_categories}")
        print(f"✅ Ready! Smoking: {smoking_categories}")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

# --- Endpoints ---


@app.post("/predict", response_model=PredictionOutput)
async def predict_cluster(input_data: PredictionInput):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Validate against model categories
    if input_data.gender not in gender_categories:
        raise HTTPException(
            status_code=400, detail=f"Gender must be one of: {gender_categories}")
    if input_data.smoking_history not in smoking_categories:
        raise HTTPException(
            status_code=400, detail=f"Smoking history must be one of: {smoking_categories}")

    try:
        predicted_cluster, probabilities = predictor.predict(
            input_data.gender, input_data.age, input_data.hypertension,
            input_data.heart_disease, input_data.bmi,
            input_data.smoking_history
        )

        cluster_int = int(predicted_cluster)
        return PredictionOutput(
            predicted_cluster=cluster_int,
            cluster_name=CLUSTER_NAMES.get(cluster_int, "Unknown Cluster"),
            probabilities=probabilities.tolist(),
            confidence=float(probabilities[cluster_int])
        )
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusOutput)
async def get_status():
    model_ready = predictor is not None and predictor.model is not None
    preprocessors_ready = (predictor is not None and
                           all([predictor.label_encoder, predictor.onehot_encoder, predictor.scaler]))

    status = "Ready" if model_ready and preprocessors_ready else "Not ready"

    return StatusOutput(
        status=status,
        model_loaded=model_ready,
        preprocessors_loaded=preprocessors_ready,
        accepted_gender_inputs=list(GENDER_MAPPING.keys()),
        accepted_smoking_inputs=list(SMOKING_MAPPING.keys()),
        cluster_names=CLUSTER_NAMES
    )


@app.get("/")
async def home():
    return {"message": "Diabetes Cluster Prediction API - Visit /docs for documentation"}

# --- Main ---
if __name__ == '__main__':
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Preprocessor: {PREPROCESSOR_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
    elif not os.path.exists(PREPROCESSOR_PATH):
        print(f"❌ Preprocessor file not found: {PREPROCESSOR_PATH}")
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
