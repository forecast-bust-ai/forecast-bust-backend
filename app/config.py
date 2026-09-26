"""
FastAPI Backend Configuration.
Loads paths, CORS configurations, environment variables, and settings.
"""

import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BACKEND_DIR.parent

# Resolve model path
MODEL_PATH = BACKEND_DIR / "models" / "trained" / "baseline_xgboost_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = WORKSPACE_ROOT / "forecast-bust-model" / "artifacts" / "xgboost" / "baseline_xgboost_model.joblib"

METRICS_PATH = WORKSPACE_ROOT / "forecast-bust-model" / "artifacts" / "xgboost" / "baseline_metrics.json"
FEATURE_CONFIG_PATH = WORKSPACE_ROOT / "forecast-bust-model" / "artifacts" / "xgboost" / "feature_config.json"

# Resolve processed dataset path
DATASET_PATH = WORKSPACE_ROOT / "forecast-bust-model" / "data" / "processed" / "training_dataset_20240601.nc"
if not DATASET_PATH.exists():
    DATASET_PATH = WORKSPACE_ROOT / "forecast-bust-model" / "data" / "processed" / "aligned_forecast_verification_20240601.nc"

# App Metadata
APP_NAME = "AI-Based Forecast Bust Detection API"
APP_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# CORS Configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "*",
]
