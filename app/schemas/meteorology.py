"""
Pydantic Schemas for Meteorological Forecast Explorer API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayerMetadataItem(BaseModel):
    id: str
    name: str
    category: str
    units: str
    palette: str
    min: float
    max: float
    description: str
    has_levels: Optional[bool] = False
    icon: Optional[str] = "Cloud"


class CategoryResponseItem(BaseModel):
    category: str
    count: int
    layers: List[LayerMetadataItem]


class SpatialPoint(BaseModel):
    latitude: float
    longitude: float
    value: float


class LayerDataResponse(BaseModel):
    layer_id: str
    title: str
    units: str
    palette: str
    lead_time_hours: int
    level_hpa: int
    initialization_time: str
    valid_time: str
    grid_resolution_deg: float
    bounding_box: Dict[str, float]
    statistics: Dict[str, float]
    total_points: int
    points: List[SpatialPoint]


class VectorItem(BaseModel):
    latitude: float
    longitude: float
    u: float
    v: float
    speed_mps: float
    speed_kts: float
    direction_deg: float


class VectorFieldResponse(BaseModel):
    vector_id: str
    lead_time_hours: int
    level_hpa: int
    units: str
    total_vectors: int
    vectors: List[VectorItem]


class SoundingLevel(BaseModel):
    level_hpa: int
    geopotential_height_gpm: float
    u_wind_mps: float
    v_wind_mps: float
    wind_speed_kts: float
    relative_humidity_pct: float


class PointProfileResponse(BaseModel):
    latitude: float
    longitude: float
    lead_time_hours: int
    surface_metrics: Dict[str, Any]
    ensemble_distribution: Dict[str, Any]
    vertical_sounding: List[SoundingLevel]


class RegimeIndicator(BaseModel):
    name: str
    value: str
    status: str


class MonsoonRegimeResponse(BaseModel):
    lead_time_hours: int
    detected_regime: str
    confidence_pct: float
    regime_color: str
    description: str
    webster_yang_index_mps: float
    central_india_mean_rain_mm: float
    bay_of_bengal_depression_prob_pct: float
    mean_olr_w_m2: float
    indicators: List[RegimeIndicator]


class EnsemblePoint(BaseModel):
    latitude: float
    longitude: float
    mean: float
    spread: float
    p10: float
    p50: float
    p90: float
    prob_exceedance: float


class EnsembleAnalysisResponse(BaseModel):
    lead_time_hours: int
    threshold_mm: float
    total_members: int
    overall_mean: float
    overall_spread: float
    total_points: int
    points: List[EnsemblePoint]


class ComparisonPoint(BaseModel):
    latitude: float
    longitude: float
    raw_forecast: float
    ai_forecast: float
    observed: float
    raw_bias: float
    corrected_bias: float
    error_reduction_pct: float


class AiComparisonResponse(BaseModel):
    lead_time_hours: int
    metrics: Dict[str, float]
    total_points: int
    points: List[ComparisonPoint]


class ProviderStatusItem(BaseModel):
    provider_id: str
    name: str
    status: str
    is_configured: bool
    initialization_time: Optional[str] = None
    domain: Optional[str] = None
    members: Optional[int] = 1
    resolution_deg: Optional[float] = 0.5
    vertical_levels_hpa: Optional[List[int]] = None


# Core Forecast Bust & Confidence Schemas
class ForecastConfidencePoint(BaseModel):
    latitude: float
    longitude: float
    confidence_pct: float
    bust_probability_pct: float
    expected_error_mm: float
    is_error_prone: bool
    category: str


class ForecastConfidenceResponse(BaseModel):
    lead_time_hours: int
    lead_day: int
    mean_confidence_pct: float
    high_confidence_area_pct: float
    error_prone_area_pct: float
    total_points: int
    grid_resolution_deg: float
    points: List[ForecastConfidencePoint]


class BustProbabilityPoint(BaseModel):
    latitude: float
    longitude: float
    bust_probability_pct: float
    risk_category: str
    ensemble_spread_mm: float
    model_disagreement_index: float


class BustProbabilityResponse(BaseModel):
    lead_time_hours: int
    lead_day: int
    mean_bust_probability_pct: float
    high_risk_points_count: int
    total_points: int
    grid_resolution_deg: float
    points: List[BustProbabilityPoint]


class ExpectedErrorPoint(BaseModel):
    latitude: float
    longitude: float
    expected_error_mm: float
    forecast_precipitation_mm: float
    ensemble_std_mm: float


class ExpectedErrorResponse(BaseModel):
    lead_time_hours: int
    lead_day: int
    mean_expected_error_mm: float
    max_expected_error_mm: float
    units: str
    total_points: int
    points: List[ExpectedErrorPoint]


class ModelDisagreementPoint(BaseModel):
    latitude: float
    longitude: float
    disagreement_index: float
    gfs_ecmwf_diff_mm: float
    max_model_spread_mm: float


class ModelDisagreementResponse(BaseModel):
    lead_time_hours: int
    lead_day: int
    mean_disagreement_index: float
    high_disagreement_area_pct: float
    models_compared: List[str]
    total_points: int
    points: List[ModelDisagreementPoint]


class ForecastExplanationResponse(BaseModel):
    latitude: float
    longitude: float
    lead_time_hours: int
    lead_day: int
    confidence_category: str
    confidence_color: str
    confidence_pct: float
    bust_probability_pct: float
    weather_regime: str
    feature_attributions: Dict[str, float]
    meteorological_reasons: List[str]

