"""
Meteorological Forecast Explorer API Routes.
Provides interactive layer data, regime classification, ensemble dispersion,
multi-map AI post-processing comparison, and vector streamlines.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.providers.registry import get_provider_registry, ProviderRegistry
from app.services.layer_catalog import get_all_categories, get_layer_metadata
from app.schemas.meteorology import (
    CategoryResponseItem,
    LayerDataResponse,
    VectorFieldResponse,
    PointProfileResponse,
    MonsoonRegimeResponse,
    EnsembleAnalysisResponse,
    AiComparisonResponse,
    ProviderStatusItem,
    ForecastConfidenceResponse,
    BustProbabilityResponse,
    ExpectedErrorResponse,
    ModelDisagreementResponse,
    ForecastExplanationResponse,
)

router = APIRouter(prefix="/api/v1/forecast", tags=["Meteorological Forecast Explorer"])


@router.get(
    "/live-status",
    summary="Get Real-Time Live Online Data Ingestion Status",
    description="Returns live operational streaming status, NOAA/ECMWF protocol, and last sync timestamp.",
)
def get_live_online_status() -> Dict[str, Any]:
    from app.services.live_nwp_service import get_live_nwp_service
    return get_live_nwp_service().get_live_status()


@router.get(
    "/models",
    response_model=List[ProviderStatusItem],
    summary="Get Available Forecast Models & Providers",
    description="Returns metadata and configuration status for GEFS, GFS, ECMWF, IMD, and ERA5 providers.",
)
def get_forecast_models(
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> List[ProviderStatusItem]:
    return registry.list_providers()


@router.get(
    "/times",
    response_model=List[int],
    summary="Get Available Forecast Lead Times",
    description="Returns list of available forecast lead-time hours (0h to 240h).",
)
def get_forecast_times(
    model: str = Query("gefs", description="Model provider ID (gefs, gfs, ecmwf, imd, era5)"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> List[int]:
    provider = registry.get_provider(model)
    return provider.get_available_lead_times()


@router.get(
    "/layers",
    response_model=List[CategoryResponseItem],
    summary="Get All Meteorological Layer Categories",
    description="Returns the 10 collapsible categories and layer catalog with units, palettes, and descriptions.",
)
def get_meteorological_layers() -> List[CategoryResponseItem]:
    return get_all_categories()


@router.get(
    "/layer/{layer_id}",
    response_model=LayerDataResponse,
    summary="Get Gridded Meteorological Layer Data",
    description="Returns spatial grid data, bounding box, units, dynamic palette, and statistics for selected layer.",
)
def get_layer_data(
    layer_id: str,
    lead_time_hours: int = Query(24, ge=0, le=240, description="Forecast lead time in hours"),
    level_hpa: int = Query(850, description="Pressure level in hPa (1000, 850, 700, 500, 300)"),
    threshold_mm: float = Query(25.0, description="Rainfall threshold in mm for exceedance probability"),
    model: str = Query("gefs", description="Forecast model provider (gefs, gfs, ecmwf, imd, era5)"),
    stride: int = Query(1, ge=1, le=8, description="Spatial subsampling stride for performance"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> LayerDataResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_layer_data(
            layer_id=layer_id,
            lead_time_hours=lead_time_hours,
            level_hpa=level_hpa,
            threshold_mm=threshold_mm,
            stride=stride,
        )
        return LayerDataResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch layer '{layer_id}': {str(e)}",
        )


@router.get(
    "/vectors/{vector_id}",
    response_model=VectorFieldResponse,
    summary="Get Atmospheric Flow & Wind Vector Field",
    description="Returns u and v vector components, speed, and angle for GPU animated particle streamlines.",
)
def get_vector_field(
    vector_id: str = "wind_850hpa",
    lead_time_hours: int = Query(24, ge=0, le=240),
    level_hpa: int = Query(850, description="Atmospheric level (1000, 850, 700, 500, 300)"),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> VectorFieldResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_vector_field(
            vector_id=vector_id,
            lead_time_hours=lead_time_hours,
            level_hpa=level_hpa,
            stride=stride,
        )
        return VectorFieldResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract vector field: {str(e)}",
        )


@router.get(
    "/point-profile",
    response_model=PointProfileResponse,
    summary="Get Atmospheric Column & Surface Sounding Profile",
    description="Returns multi-level thermodynamic and kinematic profile, ensemble dispersion, and surface diagnostics at coordinates.",
)
def get_point_profile(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> PointProfileResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_point_profile(
            latitude=latitude,
            longitude=longitude,
            lead_time_hours=lead_time_hours,
        )
        return PointProfileResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract point sounding profile: {str(e)}",
        )


@router.get(
    "/regime",
    response_model=MonsoonRegimeResponse,
    summary="Get Monsoon Weather Regime Classification",
    description="Returns detected Monsoon Regime (Active, Break, Normal), Webster-Yang index, confidence %, and diagnostic metrics.",
)
def get_monsoon_regime(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> MonsoonRegimeResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_monsoon_regime(lead_time_hours=lead_time_hours)
        return MonsoonRegimeResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate monsoon regime: {str(e)}",
        )


@router.get(
    "/ensemble",
    response_model=EnsembleAnalysisResponse,
    summary="Get Ensemble Dispersion & Probability Metrics",
    description="Returns ensemble mean, spread, range, quantiles (P10, P50, P90), and exceedance probability grid.",
)
def get_ensemble_analysis(
    lead_time_hours: int = Query(24, ge=0, le=240),
    threshold_mm: float = Query(25.0, ge=0.0),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> EnsembleAnalysisResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_ensemble_analysis(
            lead_time_hours=lead_time_hours,
            threshold_mm=threshold_mm,
            stride=stride,
        )
        return EnsembleAnalysisResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate ensemble analysis: {str(e)}",
        )


@router.get(
    "/ai-comparison",
    response_model=AiComparisonResponse,
    summary="Get Raw NWP vs Regime-Aware AI Post-Processing Comparison Grid",
    description="Returns spatial grid comparing Raw NWP vs Regime AI vs Observed Reference with bias and MAE reduction %.",
)
def get_ai_comparison(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> AiComparisonResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_ai_postprocessing_comparison(
            lead_time_hours=lead_time_hours,
            stride=stride,
        )
        return AiComparisonResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate AI comparison: {str(e)}",
        )


@router.get(
    "/confidence",
    response_model=ForecastConfidenceResponse,
    summary="Get Spatial Forecast Confidence & Error-Prone Map",
    description="Returns spatial grid of calibrated forecast confidence, bust probabilities, and error-prone zones.",
)
def get_forecast_confidence(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ForecastConfidenceResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_confidence_map(lead_time_hours=lead_time_hours, stride=stride)
        return ForecastConfidenceResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate confidence map: {str(e)}",
        )


@router.get(
    "/bust-probability",
    response_model=BustProbabilityResponse,
    summary="Get Calibrated Forecast Bust Probability Map",
    description="Returns calibrated probability grid of NWP forecast failure for Day 1 to Day 10.",
)
def get_bust_probability(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> BustProbabilityResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_bust_probability_map(lead_time_hours=lead_time_hours, stride=stride)
        return BustProbabilityResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate bust probability map: {str(e)}",
        )


@router.get(
    "/error",
    response_model=ExpectedErrorResponse,
    summary="Get Expected Forecast Error Continuous Field",
    description="Returns expected quantitative error magnitude regression field for selected lead time.",
)
def get_expected_error(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ExpectedErrorResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_expected_error_map(lead_time_hours=lead_time_hours, stride=stride)
        return ExpectedErrorResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate expected error field: {str(e)}",
        )


@router.get(
    "/model-disagreement",
    response_model=ModelDisagreementResponse,
    summary="Get Multi-Model Disagreement Index (MDI) Map",
    description="Returns inter-model spread and normalized disagreement index comparing GFS, ECMWF, and GEFS.",
)
def get_model_disagreement(
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    stride: int = Query(2, ge=1, le=8),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ModelDisagreementResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_model_disagreement_map(lead_time_hours=lead_time_hours, stride=stride)
        return ModelDisagreementResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate model disagreement index: {str(e)}",
        )


@router.get(
    "/explanation",
    response_model=ForecastExplanationResponse,
    summary="Get Explainable AI SHAP Attribution for Coordinate",
    description="Returns transparent SHAP feature contributions and meteorological reasons for why confidence is low.",
)
def get_forecast_explanation(
    latitude: float = Query(21.5, ge=-90.0, le=90.0),
    longitude: float = Query(78.5, ge=-180.0, le=180.0),
    lead_time_hours: int = Query(24, ge=0, le=240),
    model: str = Query("gefs"),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> ForecastExplanationResponse:
    try:
        provider = registry.get_provider(model)
        data = provider.get_explanation(
            latitude=latitude,
            longitude=longitude,
            lead_time_hours=lead_time_hours,
        )
        return ForecastExplanationResponse(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SHAP explanation: {str(e)}",
        )

