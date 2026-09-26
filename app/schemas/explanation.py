"""
Pydantic Schemas for Model Explainability Endpoint (/api/v1/explanation).
"""

from pydantic import BaseModel, Field


class ExplanationResponse(BaseModel):
    model_type: str = Field(..., description="ML model architecture name")
    threshold_applied_mm: float = Field(..., description="Bust threshold in mm")
    global_feature_importances: dict[str, float] = Field(..., description="Global feature importance percentages")
    top_driving_factors: list[str] = Field(..., description="Top features contributing to forecast bust risk")
    leakage_prevention_status: str = Field(..., description="Confirmation of zero verification leakage")
    description: str = Field(..., description="Detailed explanation text")
