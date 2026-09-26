"""
Test Suite for Meteorological Forecast Explorer & Forecast Bust Engine API Endpoints.
Tests all live and calculated endpoints across the India domain.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_layers_catalog():
    """Test GET /api/v1/forecast/layers returns metadata catalog."""
    response = client.get("/api/v1/forecast/layers")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "category" in data[0]
    assert "layers" in data[0]


def test_layer_data_endpoint():
    """Test GET /api/v1/forecast/layer/{layer_id} returns gridded data."""
    response = client.get("/api/v1/forecast/layer/precip_surface?lead_time_hours=24&model=gefs")
    assert response.status_code == 200
    data = response.json()
    assert data["layer_id"] == "precip_surface"
    assert "points" in data
    assert len(data["points"]) > 0
    assert "statistics" in data


def test_vector_field_endpoint():
    """Test GET /api/v1/forecast/vectors/{vector_id} returns U and V components."""
    response = client.get("/api/v1/forecast/vectors/wind_850hpa?lead_time_hours=48&model=gefs")
    assert response.status_code == 200
    data = response.json()
    assert data["vector_id"] == "wind_850hpa"
    assert "vectors" in data
    assert len(data["vectors"]) > 0
    assert "u" in data["vectors"][0]
    assert "v" in data["vectors"][0]


def test_forecast_confidence():
    """Test GET /api/v1/forecast/confidence."""
    response = client.get("/api/v1/forecast/confidence?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "mean_confidence_pct" in data
    assert "high_confidence_area_pct" in data
    assert "points" in data
    assert len(data["points"]) > 0
    assert 0 <= data["mean_confidence_pct"] <= 100


def test_bust_probability():
    """Test GET /api/v1/forecast/bust-probability."""
    response = client.get("/api/v1/forecast/bust-probability?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "mean_bust_probability_pct" in data
    assert "high_risk_points_count" in data
    assert "points" in data
    assert len(data["points"]) > 0
    assert 0 <= data["mean_bust_probability_pct"] <= 100


def test_expected_error():
    """Test GET /api/v1/forecast/error."""
    response = client.get("/api/v1/forecast/error?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "mean_expected_error_mm" in data
    assert "max_expected_error_mm" in data
    assert "points" in data
    assert len(data["points"]) > 0


def test_model_disagreement():
    """Test GET /api/v1/forecast/model-disagreement."""
    response = client.get("/api/v1/forecast/model-disagreement?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "mean_disagreement_index" in data
    assert "high_disagreement_area_pct" in data
    assert "points" in data
    assert len(data["points"]) > 0


def test_weather_regime():
    """Test GET /api/v1/forecast/regime."""
    response = client.get("/api/v1/forecast/regime?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "detected_regime" in data
    assert "confidence_pct" in data
    assert "webster_yang_index_mps" in data


def test_forecast_explanation():
    """Test GET /api/v1/forecast/explanation."""
    response = client.get("/api/v1/forecast/explanation?lead_time_hours=24&latitude=23.0&longitude=86.0")
    assert response.status_code == 200
    data = response.json()
    assert "confidence_pct" in data
    assert "bust_probability_pct" in data
    assert "feature_attributions" in data
    assert "meteorological_reasons" in data
    assert len(data["feature_attributions"]) > 0


def test_ensemble_analysis():
    """Test GET /api/v1/forecast/ensemble."""
    response = client.get("/api/v1/forecast/ensemble?lead_time_hours=48&threshold_mm=25.0")
    assert response.status_code == 200
    data = response.json()
    assert data["total_members"] == 21
    assert "overall_mean" in data
    assert "overall_spread" in data
    assert "points" in data
    assert len(data["points"]) > 0


def test_point_profile():
    """Test GET /api/v1/forecast/point-profile."""
    response = client.get("/api/v1/forecast/point-profile?latitude=19.07&longitude=72.87&lead_time_hours=48")
    assert response.status_code == 200
    data = response.json()
    assert "vertical_sounding" in data
    assert "surface_metrics" in data
    assert len(data["vertical_sounding"]) > 0


def test_ai_comparison():
    """Test GET /api/v1/forecast/ai-comparison."""
    response = client.get("/api/v1/forecast/ai-comparison?lead_time_hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "points" in data
    assert len(data["points"]) > 0


def test_live_status():
    """Test GET /api/v1/forecast/live-status."""
    response = client.get("/api/v1/forecast/live-status")
    assert response.status_code == 200
    data = response.json()
    assert "is_online" in data
    assert "stream_source" in data
    assert "status" in data
    assert data["status"] == "LIVE ONLINE"
