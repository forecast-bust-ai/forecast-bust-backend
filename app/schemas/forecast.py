"""
Pydantic Schemas for Gridded Confidence Map and Forecast Endpoints.
"""

from pydantic import BaseModel, Field


class ConfidenceMapPoint(BaseModel):
    lead_time_hours: int = Field(..., description="Forecast lead time in hours")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    bust_probability: float = Field(..., description="Predicted forecast bust probability")
    confidence: float = Field(..., description="Model confidence score")
    expected_error_mm: float = Field(..., description="Estimated forecast error in mm")
    forecast_precipitation_mm: float = Field(..., description="GFS forecasted precipitation in mm")


class ConfidenceMapResponse(BaseModel):
    lead_time_hours: int = Field(..., description="Lead time in hours for this map layer")
    total_points: int = Field(..., description="Total spatial grid points in payload")
    grid_resolution_deg: float = Field(..., description="Spatial grid step in degrees")
    bounding_box: dict = Field(..., description="Region bounding box bounds")
    points: list[ConfidenceMapPoint] = Field(..., description="Grid points containing confidence & bust estimates")


class ForecastPoint(BaseModel):
    latitude: float
    longitude: float
    forecast_precipitation_mm: float
    verification_precipitation_mm: float | None = None
    absolute_error_mm: float | None = None
    is_bust_actual: bool | None = None


class ForecastMapResponse(BaseModel):
    lead_time_hours: int
    total_points: int
    initialization_time: str
    valid_time: str
    points: list[ForecastPoint]
