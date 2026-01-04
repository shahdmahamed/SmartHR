from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from backend.ML.input_preprocess import input_preprocess 
# =========================
# Load trained model
# =========================
#model = joblib.load("ML/model.pkl")
model_path = os.path.join("backend", "models", "model.pkl")
model = joblib.load(model_path)
# =========================
# FastAPI App
# =========================
app = FastAPI(title="Smart HR - Salary Prediction API")

# =========================
# Input Schema
# =========================
class CandidateInput(BaseModel):
    years_experience: int
    skill_count: int
    job_state: str

# =========================
# Prediction endpoint
# =========================
@app.post("/predict-salary")
def predict_salary(data: CandidateInput):
    X = input_preprocess(data, model.feature_names_in_)
    prediction = model.predict(X)[0]
    return {"predicted_salary": round(float(prediction), 2)}

# =========================
# Root endpoint
# =========================
@app.get("/")
def home():
    return {"status": "Smart HR API is running"}

#uvicorn ML.app:app --reload
