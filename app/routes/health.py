"""
Health Check Route Handler.
GET /health & GET /api/v1/health
"""

from fastapi import APIRouter, Depends
from app.config import APP_NAME, APP_VERSION
from app.services.model_service import get_model_service, ModelService

router = APIRouter(tags=["Health"])


@router.get("/health", summary="System Health Check")
@router.get("/HEALTH", include_in_schema=False)
@router.get("/api/v1/health", summary="System Health Check v1")
@router.get("/api/v1/HEALTH", include_in_schema=False)
def health_check(model_svc: ModelService = Depends(get_model_service)):
    """Check API status and model readiness."""
    model_loaded = hasattr(model_svc, "model") and model_svc.model.is_trained

    return {
        "status": "ok",
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": model_loaded,
        "model_type": "XGBoost Baseline Classifier",
    }
