"""
FastAPI Backend Integration & Unit Test Suite.
Tests health endpoint, valid prediction, invalid input validation, model loading,
confidence map responses, and explainability endpoints.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "baseline"))

from app.main import app
from app.services.model_service import get_model_service

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health and GET /api/v1/health return 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 200


def test_model_loading_service():
    """Test model service singleton initializes and loads trained model."""
    svc = get_model_service()
    assert svc is not None
    assert hasattr(svc, "model")
    assert svc.model.is_trained is True


def test_valid_prediction():
    """Test POST /api/v1/predict with valid input features."""
    payload = {
        "lead_time_hours": 48.0,
        "latitude": 19.0,
        "longitude": 73.0,
        "forecast_precipitation": 35.5,
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "bust_probability" in data
    assert "confidence" in data
    assert "is_bust_predicted" in data
    assert "risk_category" in data
    assert 0.0 <= data["bust_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["is_bust_predicted"] is True  # Heavy rain (35.5mm) yields high bust prob
    assert data["risk_category"] == "High Bust Risk"


def test_invalid_input_validation():
    """Test POST /api/v1/predict returns 422 Unprocessable Entity on invalid inputs."""
    # Invalid latitude out of bounds (> 40.0)
    invalid_lat = {
        "lead_time_hours": 48.0,
        "latitude": 85.0,  # Invalid
        "longitude": 73.0,
        "forecast_precipitation": 10.0,
    }
    res1 = client.post("/api/v1/predict", json=invalid_lat)
    assert res1.status_code == 422

    # Negative precipitation (invalid)
    invalid_precip = {
        "lead_time_hours": 48.0,
        "latitude": 19.0,
        "longitude": 73.0,
        "forecast_precipitation": -15.0,  # Invalid
    }
    res2 = client.post("/api/v1/predict", json=invalid_precip)
    assert res2.status_code == 422


def test_confidence_map_endpoint():
    """Test GET /api/v1/confidence-map returns gridded confidence payload."""
    response = client.get("/api/v1/confidence-map?lead_time_hours=24&stride=8")
    assert response.status_code == 200

    data = response.json()
    assert data["lead_time_hours"] == 24
    assert data["total_points"] > 0
    assert len(data["points"]) == data["total_points"]

    pt = data["points"][0]
    assert "bust_probability" in pt
    assert "confidence" in pt
    assert "expected_error_mm" in pt
    assert "latitude" in pt
    assert "longitude" in pt


def test_forecast_endpoint():
    """Test GET /api/v1/forecast returns gridded forecast payload."""
    response = client.get("/api/v1/forecast?lead_time_hours=24&stride=8")
    assert response.status_code == 200

    data = response.json()
    assert data["lead_time_hours"] == 24
    assert data["total_points"] > 0


def test_explanation_endpoint():
    """Test GET /api/v1/explanation returns feature importances."""
    response = client.get("/api/v1/explanation")
    assert response.status_code == 200

    data = response.json()
    assert "global_feature_importances" in data
    assert "latitude" in data["global_feature_importances"]
    assert "top_driving_factors" in data


def test_valid_prediction_cnn():
    """Test POST /api/v1/predict?model_type=cnn with CNN model."""
    payload = {
        "lead_time_hours": 48.0,
        "latitude": 19.0,
        "longitude": 73.0,
        "forecast_precipitation": 35.5,
    }

    response = client.post("/api/v1/predict?model_type=cnn", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "bust_probability" in data
    assert "confidence" in data
    assert "is_bust_predicted" in data
    assert "risk_category" in data
    assert 0.0 <= data["bust_probability"] <= 1.0


def test_confidence_map_endpoint_cnn():
    """Test GET /api/v1/confidence-map with model_type=cnn."""
    response = client.get("/api/v1/confidence-map?lead_time_hours=24&stride=8&model_type=cnn")
    assert response.status_code == 200

    data = response.json()
    assert data["lead_time_hours"] == 24
    assert data["total_points"] > 0
    assert len(data["points"]) == data["total_points"]

    pt = data["points"][0]
    assert "bust_probability" in pt
    assert "confidence" in pt

