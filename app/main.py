"""
FastAPI Main Application Entry Point.
Configures CORS, routes, middleware, and startup verification.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "baseline"))

from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from app.routes import health, predict, forecast, explanation

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI-Based Forecast Bust Detection REST API for NCMRWF / MoES",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration suitable for Next.js / React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(forecast.router)
app.include_router(explanation.router)


@app.get("/", include_in_schema=False)
def root_redirect():
    return {
        "message": f"Welcome to {APP_NAME} API v{APP_VERSION}",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
