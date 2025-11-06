# healthcare_model/api.py
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conint, confloat, field_validator

import joblib
import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path

# ------------------------------------------------------------------
#  NEW:  monitoring & validation imports
# ------------------------------------------------------------------
from monitoring import initialize_monitor, model_monitor
from data_validation import validate_incoming_data
from error_handling import handle_prediction_with_fallback, error_handler, get_system_health
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#  FIX: make repo root visible → config.py  can  be  imported
# ------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent.parent          # ExplainableAI-Project
sys.path.insert(0, str(repo_root))                          # add once, first
# ------------------------------------------------------------------

# ----------  project-specific imports  ----------
from config import settings               # central config
# ----------------------------------------------

# ---------------  logging setup  ----------------
log_level = getattr(logging, getattr(settings, "LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)
# ----------------------------------------------

# ======  security: rate-limit storage  =======
# (in production replace with Redis)
request_times: Dict[str, list] = {}

# ======  lifespan: secure model loading + monitoring  ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Secure startup / shutdown lifecycle."""
    global model
    try:
        from utils import get_model_path

        model_path = get_model_path("pipeline_heart_optimized.joblib")
        if not model_path.exists():
            model_path = get_model_path("pipeline_heart.joblib")

        # basic integrity check: model age
        model_age_days = (datetime.now().timestamp() - model_path.stat().st_mtime) / 86400
        if model_age_days > getattr(settings, "MAX_MODEL_AGE_DAYS", 365):
            logger.warning(f"Model is {model_age_days:.0f} days old – consider retraining.")

        model = joblib.load(model_path)

        # INITIALIZE MONITORING SYSTEM
        initialize_monitor()

        logger.info("✅  Model loaded successfully (secure lifecycle).")
        logger.info("✅  Monitoring system initialized.")
    except Exception as e:
        logger.error(f"❌  Failed to start API: {e}")
        raise RuntimeError("API startup failed") from e

    yield       # application running

    logger.info("🛑  Application shutdown complete.")


# ==========  FastAPI app (with security)  ==========
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Secure ML API for heart-disease risk prediction with explainable-AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ----------------  CORS  -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["http://localhost:7860",
                                                     "http://127.0.0.1:7860"]),
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# ==========  secure Pydantic models  ==========
class PatientData(BaseModel):
    age:     conint(ge=1, le=120)
    sex:     conint(ge=0, le=1)
    cp:      conint(ge=0, le=3)
    trestbps:conint(ge=50, le=250)
    chol:    conint(ge=100, le=600)
    fbs:     conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalach: conint(ge=50, le=220)
    exang:   conint(ge=0, le=1)
    oldpeak: confloat(ge=0.0, le=10.0)
    slope:   conint(ge=0, le=2)
    ca:      conint(ge=0, le=3)
    thal:    conint(ge=1, le=3)

    @field_validator("*")
    @classmethod
    def medical_sanity_check(cls, v, info):
        """Extra medical-range guard."""
        field_name = info.field_name
        hard_ranges = {
            "age": (1, 120),
            "trestbps": (50, 250),
            "chol": (100, 600),
            "thalach": (50, 220)
        }
        if field_name in hard_ranges:
            low, high = hard_ranges[field_name]
            if not (low <= v <= high):
                raise ValueError(f"{field_name} must be between {low} and {high}")
        return v


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    confidence: str
    advice: str
    timestamp: str
    success: bool


# ==========  security middleware (rate-limit + logging)  ==========
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enhanced security middleware with error handling."""
    client_ip = request.client.host
    now = time.time()

    try:
        # Rate limiting with error handling
        window = [t for t in request_times.get(client_ip, []) if now - t < 60]
        if len(window) >= 10:
            logger.warning(f"Rate-limit hit by {client_ip}")
            error_handler.record_error('rate_limit', f"IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again in 60 seconds."}
            )
        request_times[client_ip] = window + [now]

        # Request logging
        logger.info(f"{request.method} {request.url} from {client_ip}")

        # Process request with error handling
        response = await call_next(request)
        return response

    except Exception as e:
        # Catch any middleware errors
        error_handler.record_error('middleware', str(e))
        logger.error(f"Middleware error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error in request processing"}
        )


# ----------------  globals  -----------------
model = None          # loaded in lifespan


# ----------------  endpoints  ----------------
@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "status": "healthy",
        "version": "2.0.0",
        "security": "enabled"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "security": "active",
        "timestamp": datetime.now().isoformat()
    }


# ------------------------------------------------------------------
#  NEW:  monitored + validated prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData, request: Request):
    try:
        client_ip = request.client.host

        # Convert to dict for validation and logging
        patient_dict = patient.model_dump()
        logger.info(f"Prediction request from {client_ip}: {patient_dict}")

        # DATA VALIDATION
        is_valid, validation_errors = validate_incoming_data(patient_dict)
        if not is_valid:
            logger.warning(f"Data validation failed: {validation_errors}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input data: {', '.join(validation_errors)}"
            )

        # CREATE INPUT DATA
        input_df = pd.DataFrame([patient_dict])

        # ADVANCED PREDICTION WITH ERROR HANDLING
        prediction_result = handle_prediction_with_fallback(model, input_df)

        if not prediction_result.get('success', False):
            # Fallback response was used
            return PredictionResponse(
                **prediction_result,
                timestamp=datetime.now().isoformat()
            )

        # Extract results from successful prediction
        prob = prediction_result['probability']
        pred = prediction_result['prediction']

        # Risk assessment
        if prob < 0.2:
            risk_level, confidence, advice = "very_low", "high", "Maintain a healthy lifestyle."
        elif prob < 0.4:
            risk_level, confidence, advice = "low", "medium", "Regular checkups recommended."
        elif prob < 0.6:
            risk_level, confidence, advice = "medium", "medium", "Consult your doctor."
        elif prob < 0.8:
            risk_level, confidence, advice = "high", "high", "Schedule a cardiologist visit."
        else:
            risk_level, confidence, advice = "very_high", "high", "Seek medical attention soon."

        logger.info(f"Prediction complete – risk: {risk_level}, confidence: {confidence}")

        return PredictionResponse(
            prediction=pred,
            probability=prob,
            risk_level=risk_level,
            confidence=confidence,
            advice=advice,
            timestamp=datetime.now().isoformat(),
            success=True
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        logger.error(f"Unexpected prediction error from {client_ip}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


# ------------------------------------------------------------------
#  NEW:  advanced monitoring health endpoint
# ------------------------------------------------------------------
@app.get("/monitoring/health")
async def monitoring_health():
    """Advanced system health monitoring endpoint"""
    try:
        # Get system health from error handler
        system_health = get_system_health()

        # Get model monitoring data if available
        model_health = {}
        if model_monitor and hasattr(model_monitor, 'metrics_history'):
            if model_monitor.metrics_history:
                latest_metrics = model_monitor.metrics_history[-1]
                model_health = {
                    'latest_performance': latest_metrics,
                    'model_age_days': model_monitor.get_model_age(),
                    'performance_trend': model_monitor.analyze_performance_trend()
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "model_health": model_health,
            "monitoring_status": "active"
        }
    except Exception as e:
        logger.error(f"Monitoring health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {"overall_status": "unknown"},
            "model_health": {},
            "monitoring_status": "error",
            "error": str(e)
        }


# ----------------  dev entry-point  ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)