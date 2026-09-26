"""
Model Explainability Route Handler.
GET /api/v1/explanation
"""

from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.explanation import ExplanationResponse
from app.services.model_service import get_model_service, ModelService

router = APIRouter(prefix="/api/v1", tags=["Explainability"])


@router.get(
    "/explanation",
    response_model=ExplanationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Explainability & Feature Importance Breakdown",
    description="Returns global feature importances, top driving meteorological factors, and leakage audit verification.",
)
def get_model_explanation(
    model_svc: ModelService = Depends(get_model_service),
) -> ExplanationResponse:
    try:
        return model_svc.get_explanation()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model explanation: {str(e)}",
        )
