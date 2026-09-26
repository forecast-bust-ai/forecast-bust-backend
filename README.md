# Forecast Bust AI — FastAPI REST API Backend (`forecast-bust-backend`)

**Organization:** `forecast-bust-ai`  
**Problem Statement ID:** 26079 (NCMRWF / Ministry of Earth Sciences)

## Overview
This repository contains the FastAPI REST API backend service for medium-range NWP forecast bust probability inference, gridded spatial confidence map generation, and model explainability.

## API Endpoints
- `GET /health`: Server health check & model readiness status.
- `POST /api/v1/predict?model_type=xgboost|cnn`: Single-point bust probability inference.
- `GET /api/v1/confidence-map?lead_time_hours=24&stride=4&model_type=xgboost|cnn`: Gridded spatial map payload over India region.
- `GET /api/v1/forecast`: Gridded GFS forecast precipitation payload.
- `GET /api/v1/explanation`: Global feature importances and explainability breakdown.

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend development server
python3 -m uvicorn app.main:app --reload --port 8000
```

## Running Integration Tests
```bash
python3 -m pytest tests/
```
