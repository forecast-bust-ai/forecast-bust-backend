"""
Gridded Forecast & Confidence Map Route Handlers.
GET /api/v1/confidence-map
GET /api/v1/forecast
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from app.schemas.forecast import ConfidenceMapResponse, ForecastMapResponse
from app.services.model_service import get_model_service, ModelService
from app.services.forecast_service import get_forecast_service, ForecastService

router = APIRouter(prefix="/api/v1", tags=["Forecast Layers"])


@router.get(
    "/confidence-map",
    response_model=ConfidenceMapResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Gridded Forecast Bust Confidence Map",
    description="Returns spatial grid containing lead time, latitude, longitude, bust probability, confidence score, and expected error.",
)
def get_confidence_map(
    lead_time_hours: int = Query(24, ge=24, le=120, description="Lead time in hours (24, 48, 72, 96, 120)"),
    stride: int = Query(4, ge=1, le=16, description="Grid spatial subsampling stride"),
    model_type: str = Query("xgboost", description="Model selection: 'xgboost' or 'cnn'"),
    model_svc: ModelService = Depends(get_model_service),
) -> ConfidenceMapResponse:
    try:
        return model_svc.generate_confidence_map(lead_time_hours=lead_time_hours, stride=stride, model_type=model_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate confidence map layer: {str(e)}",
        )


@router.get(
    "/forecast",
    response_model=ForecastMapResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Gridded Forecast Precipitation Layer",
    description="Returns spatial GFS precipitation forecast values across India region.",
)
def get_forecast_layer(
    lead_time_hours: int = Query(24, ge=24, le=120, description="Lead time in hours (24, 48, 72, 96, 120)"),
    stride: int = Query(4, ge=1, le=16, description="Grid spatial subsampling stride"),
    fcst_svc: ForecastService = Depends(get_forecast_service),
) -> ForecastMapResponse:
    try:
        return fcst_svc.get_forecast_map(lead_time_hours=lead_time_hours, stride=stride)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch forecast layer: {str(e)}",
        )
