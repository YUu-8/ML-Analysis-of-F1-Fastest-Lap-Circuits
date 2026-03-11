import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="F1 Lap Speed Predictor", version="1.0")

model = joblib.load("model/best_xgboost_model.pkl")

class CircuitFeatures(BaseModel):
    sector_straight_ratio_S1: float
    sector_straight_ratio_S2: float
    sector_straight_ratio_S3: float
    sector_slow_corner_ratio_S1: float
    sector_slow_corner_ratio_S2: float
    sector_length_km_S1: float
    sector_length_km_S2: float
    sector_length_km_S3: float

@app.get("/")
def root():
    return {"message": "F1 Fastest-Lap Speed Predictor API"}

@app.post("/predict")
def predict(features: CircuitFeatures):
    X = np.array([[
        features.sector_straight_ratio_S1,
        features.sector_straight_ratio_S2,
        features.sector_straight_ratio_S3,
        features.sector_slow_corner_ratio_S1,
        features.sector_slow_corner_ratio_S2,
        features.sector_length_km_S1,
        features.sector_length_km_S2,
        features.sector_length_km_S3,
    ]])
    prediction = model.predict(X)[0]
    return {
        "predicted_avg_speed_kmh": round(float(prediction), 3),
        "model": "XGBoost (Optimized)"
    }

@app.get("/health")
def health():
    return {"status": "ok"}