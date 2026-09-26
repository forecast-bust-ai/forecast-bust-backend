"""
Prediction Route Handler.
POST /api/v1/predict
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from app.schemas.predict import PredictionRequest, PredictionResponse
from app.services.model_service import get_model_service, ModelService

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Forecast Bust Probability",
    description="Calculates probability of medium-range forecast bust using trained XGBoost baseline model.",
)
def predict_forecast_bust(
    request: PredictionRequest,
    model_type: str = Query("xgboost", description="Model choice: 'xgboost' or 'cnn'"),
    model_svc: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    try:
        response = model_svc.predict_single(request, model_type=model_type)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine failure: {str(e)}",
        )
