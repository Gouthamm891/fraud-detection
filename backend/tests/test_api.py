import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_valid_input():
    transaction = {
        "amount": 150.0,
        "hour": 14,
        "merchant_category": 3,
        "distance_from_home": 5.2,
        "num_transactions_last_24h": 2
    }
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "model_version" in data

def test_predict_invalid_input():
    transaction = {
        "amount": "invalid",
        "hour": 14,
        "merchant_category": 3,
        "distance_from_home": 5.2,
        "num_transactions_last_24h": 2
    }
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422

def test_predict_missing_field():
    transaction = {
        "amount": 150.0,
        "hour": 14,
        "distance_from_home": 5.2,
        "num_transactions_last_24h": 2
    }
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422

def test_model_output_range():
    transaction = {
        "amount": 150.0,
        "hour": 14,
        "merchant_category": 3,
        "distance_from_home": 5.2,
        "num_transactions_last_24h": 2
    }
    response = client.post("/predict", json=transaction)
    data = response.json()
    assert 0 <= data["fraud_probability"] <= 1

def test_model_version_present():
    response = client.get("/health")
    assert "model_version" in response.json()
