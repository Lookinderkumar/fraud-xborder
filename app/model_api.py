"""
app/model_api.py
FastAPI real-time fraud scoring service.

Run:
    uvicorn app.model_api:app --reload

Endpoints:
    POST /score      — score a single transaction
    GET  /health     — liveness + model info
    GET  /stats      — rolling performance statistics
    GET  /metrics    — Prometheus text format
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap
import structlog
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.feature_state import FeatureState
from src.anomaly import AnomalyDetector
from src.reasons import generate_reasons

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
_API_KEY = os.getenv("API_KEY", "changeme")

MODELS_DIR    = Path("models")
DATA_DIR      = Path("data")
ALERTS_FILE   = DATA_DIR / "alerts.jsonl"
FEATURE_NAMES = [
    "velocity_1h", "velocity_24h", "amount_zscore_30d", "log_amount",
    "is_night", "hour_of_day", "day_of_week", "is_cross_border",
    "corridor_risk", "device_changed_24h", "ip_mismatch", "mcc_risk",
    "receiver_fan_in_24h", "rule_night_high", "rule_corridor_high",
    # 5 advanced features — set to 0 for real-time (no history available)
    "inter_arrival_seconds", "amount_round_flag", "benford_deviation",
    "receiver_country_novelty", "time_since_last_high_value",
]

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
_log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Model version
# ---------------------------------------------------------------------------

def _model_version() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    try:
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        short = "unknown"
    return f"v{date_str}-{short}"


MODEL_VERSION = _model_version()

# ---------------------------------------------------------------------------
# Rate limiter — 100 req/min per IP via middleware (not per-endpoint decorator,
# which would hide Pydantic body parameters from FastAPI's signature inspection)
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

# ---------------------------------------------------------------------------
# Lifespan — load all artefacts once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models
    app.state.xgb      = joblib.load(MODELS_DIR / "xgb.pkl")
    app.state.detector = AnomalyDetector.load(MODELS_DIR / "iforest.pkl")
    with open(MODELS_DIR / "threshold.json", encoding="utf-8") as f:
        app.state.thresholds = json.load(f)

    # SHAP explainer — cache once, never recreate per request
    app.state.explainer = shap.TreeExplainer(app.state.xgb)

    # Online feature state
    app.state.feature_state = FeatureState()

    # Runtime stats
    app.state.latency_deque: deque[float] = deque(maxlen=1_000)
    app.state.total_scored: int = 0
    app.state.alert_timestamps: deque[float] = deque(maxlen=10_000)
    app.state.uptime_start: float = time.monotonic()

    DATA_DIR.mkdir(exist_ok=True)
    _log.info("startup", model_version=MODEL_VERSION,
              threshold=app.state.thresholds.get("fpr_threshold"))
    yield
    _log.info("shutdown")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Hybrid XGBoost + IsolationForest scorer for SWIFT/SEPA payments.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: str | None = Depends(_api_key_header)) -> str:
    if api_key is None or api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TransactionIn(BaseModel):
    txn_id:           str
    timestamp:        datetime
    amount:           float       = Field(gt=0, le=10_000_000)
    currency:         str         = Field(min_length=3, max_length=3)
    sender_id:        str
    receiver_id:      str
    sender_country:   str         = Field(min_length=2, max_length=2)
    receiver_country: str         = Field(min_length=2, max_length=2)
    device_id:        str
    ip_country:       str         = Field(min_length=2, max_length=2)
    channel:          str
    mcc:              str         = Field(min_length=4, max_length=4)
    is_cross_border:  bool


class ScoreResponse(BaseModel):
    txn_id:        str
    score:         float
    is_fraud:      bool
    iso_score:     float
    reasons:       list[str]
    latency_ms:    float
    model_version: str


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_transaction(
    txn_dict: dict[str, Any],
    xgb,
    detector: AnomalyDetector,
    explainer,
    feature_state: FeatureState,
    thresholds: dict,
) -> dict[str, Any]:
    """
    Core scoring pipeline:
      1. get_features (stateful) → feature_dict
      2. Build 20-feature numpy array (advanced features zeroed for real-time)
      3. XGBoost probability
      4. Hybrid = max(xgb_prob, rule_score)
      5. IsolationForest anomaly score
      6. Ensemble = 0.7 * hybrid + 0.3 * iso
      7. SHAP → reasons
      8. update() feature state
    """
    feat_dict = feature_state.get_features(txn_dict)

    # Build feature array in training order (advanced features = 0 for real-time)
    feat_vec = np.array(
        [feat_dict.get(f, 0.0) for f in FEATURE_NAMES],
        dtype=np.float64,
    ).reshape(1, -1)

    # XGBoost
    xgb_prob = float(xgb.predict_proba(feat_vec)[0, 1])

    # Rule score
    rule_score = float(max(
        feat_dict.get("rule_night_high",   0.0),
        feat_dict.get("rule_corridor_high", 0.0),
    ))
    hybrid = max(xgb_prob, rule_score)

    # Isolation Forest
    iso = float(detector.score(feat_vec)[0])

    # Ensemble
    ensemble = 0.7 * hybrid + 0.3 * iso

    # Threshold
    threshold = float(thresholds.get("fpr_threshold", 0.5))
    is_fraud = ensemble >= threshold

    # SHAP — pass all 20 features to match model's input dimensionality,
    # then use only the first 15 SHAP values for reason codes (advanced
    # features are 0 in real-time and contribute negligible SHAP).
    shap_vals = explainer.shap_values(feat_vec)[0]
    reasons = generate_reasons(shap_vals[:15], FEATURE_NAMES[:15], {**txn_dict, **feat_dict})

    # Update stateful features after scoring (offline semantics: state = prior txns)
    feature_state.update(txn_dict)

    return {
        "score":     ensemble,
        "is_fraud":  is_fraud,
        "iso_score": iso,
        "reasons":   reasons,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoreResponse)
def score(
    request: Request,
    txn: TransactionIn,
    _key: str = Depends(_require_api_key),
) -> ScoreResponse:
    t0 = time.perf_counter()

    txn_dict = txn.model_dump()
    result = _score_transaction(
        txn_dict=txn_dict,
        xgb=request.app.state.xgb,
        detector=request.app.state.detector,
        explainer=request.app.state.explainer,
        feature_state=request.app.state.feature_state,
        thresholds=request.app.state.thresholds,
    )

    latency_ms = (time.perf_counter() - t0) * 1_000

    # Update runtime stats
    request.app.state.latency_deque.append(latency_ms)
    request.app.state.total_scored += 1
    if result["is_fraud"]:
        request.app.state.alert_timestamps.append(time.time())

    # Structured log + alerts file
    log_entry = {
        "txn_id":        txn.txn_id,
        "timestamp":     txn.timestamp.isoformat(),
        "score":         result["score"],
        "is_fraud":      result["is_fraud"],
        "latency_ms":    latency_ms,
        "reasons":       result["reasons"],
        "model_version": MODEL_VERSION,
        "client_ip":     request.client.host if request.client else "unknown",
    }
    _log.info("score", **log_entry)

    if result["is_fraud"]:
        with open(ALERTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")

    return ScoreResponse(
        txn_id=txn.txn_id,
        score=result["score"],
        is_fraud=result["is_fraud"],
        iso_score=result["iso_score"],
        reasons=result["reasons"],
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )


@app.get("/health")
def health(request: Request) -> dict:
    uptime = time.monotonic() - request.app.state.uptime_start
    return {
        "status":        "ok",
        "model_loaded":  True,
        "uptime_s":      round(uptime, 1),
        "version":       MODEL_VERSION,
        "threshold":     request.app.state.thresholds.get("fpr_threshold"),
    }


@app.get("/stats")
def stats(request: Request) -> dict:
    lats = list(request.app.state.latency_deque)
    now  = time.time()
    alerts_1h = sum(1 for t in request.app.state.alert_timestamps if now - t <= 3600)
    return {
        "total_scored":  request.app.state.total_scored,
        "alerts_1h":     alerts_1h,
        "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
        "p50_ms":         float(np.percentile(lats, 50)) if lats else 0.0,
        "p95_ms":         float(np.percentile(lats, 95)) if lats else 0.0,
    }


@app.get("/metrics")
def metrics(request: Request) -> Response:
    lats = list(request.app.state.latency_deque)
    p50  = float(np.percentile(lats, 50)) if lats else 0.0
    p95  = float(np.percentile(lats, 95)) if lats else 0.0
    body = (
        f"# HELP fraud_scored_total Total transactions scored\n"
        f"# TYPE fraud_scored_total counter\n"
        f"fraud_scored_total {request.app.state.total_scored}\n"
        f"# HELP fraud_latency_p50_ms Scoring latency p50 ms\n"
        f"# TYPE fraud_latency_p50_ms gauge\n"
        f"fraud_latency_p50_ms {p50:.2f}\n"
        f"# HELP fraud_latency_p95_ms Scoring latency p95 ms\n"
        f"# TYPE fraud_latency_p95_ms gauge\n"
        f"fraud_latency_p95_ms {p95:.2f}\n"
    )
    return Response(content=body, media_type="text/plain")
