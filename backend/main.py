from fastapi.middleware.cors import CORSMiddleware   # <-- add this linefrom fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
from typing import Dict, Any

app = FastAPI(title="Fraud Detection API")
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fraud-detection-six-nu.vercel.app"],   # <-- put your Vercel URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... rest of your code (health, predict, etc.)

# Load model at startup
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
model_path = "model/fraud_model.pkl"

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class Transaction(BaseModel):
    amount: float
    hour: int
    merchant_category: int
    distance_from_home: float
    num_transactions_last_24h: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 150.0,
                "hour": 14,
                "merchant_category": 3,
                "distance_from_home": 5.2,
                "num_transactions_last_24h": 2
            }
        }

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_version: str

@app.get("/")
async def root():
    return {"message": "Fraud Detection API", "version": MODEL_VERSION}

@app.get("/health")
async def health():
    return {
        "status": "ok" if model else "degraded",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to dataframe
    input_data = pd.DataFrame([transaction.dict()])
    
    # Predict
    fraud_prob = float(model.predict_proba(input_data)[0, 1])
    is_fraud = fraud_prob > 0.5
    
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        model_version=MODEL_VERSION
    )

@app.post("/reload-model")
async def reload_model():
    """Reload the model from disk without restarting the server"""
    global model
    try:
        model = joblib.load(model_path)
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {e}")