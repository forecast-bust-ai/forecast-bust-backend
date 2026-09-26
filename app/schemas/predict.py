"""
Pydantic Schemas for Bust Prediction Endpoint (/api/v1/predict).
"""

import datetime
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    lead_time_hours: float = Field(
        ...,
        ge=6.0,
        le=240.0,
        description="Forecast lead time in hours (e.g. 24.0, 48.0, 72.0, 96.0, 120.0)",
        examples=[48.0],
    )
    latitude: float = Field(
        ...,
        ge=0.0,
        le=40.0,
        description="Latitude coordinate in degrees north (India region 0°N to 40°N)",
        examples=[19.0],
    )
    longitude: float = Field(
        ...,
        ge=60.0,
        le=100.0,
        description="Longitude coordinate in degrees east (India region 60°E to 100°E)",
        examples=[73.0],
    )
    forecast_precipitation: float = Field(
        ...,
        ge=0.0,
        description="Predicted GFS precipitation accumulation in mm",
        examples=[25.5],
    )


class PredictionResponse(BaseModel):
    bust_probability: float = Field(..., description="Predicted probability of forecast bust (0.0 to 1.0)")
    confidence: float = Field(..., description="Model prediction confidence score (0.0 to 1.0)")
    is_bust_predicted: bool = Field(..., description="True if bust_probability >= decision threshold")
    risk_category: str = Field(..., description="Human-readable risk label (Low Risk, Moderate Risk, High Bust Risk)")
    threshold_applied_mm: float = Field(..., description="Historical forecast error bust threshold in mm (11.05 mm)")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
