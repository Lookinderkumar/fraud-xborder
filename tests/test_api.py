"""
tests/test_api.py
Integration tests for app/model_api.py.

Uses httpx.AsyncClient with ASGITransport + asgi_lifespan to properly
trigger FastAPI lifespan startup (which loads models and app state).
Uses anyio's asyncio backend via @pytest.mark.anyio.

Run:
    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.model_api import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_KEY = "changeme"
HEADERS = {"X-API-Key": API_KEY}

LEGIT_TXN = {
    "txn_id":           "test_legit_001",
    "timestamp":        "2026-01-15T10:30:00+00:00",
    "amount":           5000.00,
    "currency":         "EUR",
    "sender_id":        "sender_1",
    "receiver_id":      "receiver_1",
    "sender_country":   "DE",
    "receiver_country": "FR",
    "device_id":        "dev_1_0",
    "ip_country":       "DE",
    "channel":          "SEPA",
    "mcc":              "5411",
    "is_cross_border":  False,
}

FRAUD_NIGHT_TXN = {
    "txn_id":           "test_fraud_night_001",
    "timestamp":        "2026-01-15T02:30:00+00:00",
    "amount":           800_000.00,
    "currency":         "EUR",
    "sender_id":        "sender_fraud_1",
    "receiver_id":      "receiver_fraud_1",
    "sender_country":   "IE",
    "receiver_country": "NG",
    "device_id":        "dev_fraud_new_999",
    "ip_country":       "CN",
    "channel":          "SWIFT",
    "mcc":              "4829",
    "is_cross_border":  True,
}

# ---------------------------------------------------------------------------
# Async client fixture — triggers lifespan via LifespanManager
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    """Create async test client with FastAPI lifespan events triggered."""
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app),
            base_url="http://test",
        ) as c:
            yield c


# ---------------------------------------------------------------------------
# Health tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_health_returns_ok(client: AsyncClient):
    resp = await client.get("/health", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# Score tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_score_valid_legitimate(client: AsyncClient):
    resp = await client.post("/score", json=LEGIT_TXN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] < 0.7, f"Legit txn scored high: {body['score']}"


@pytest.mark.anyio
async def test_score_fraud_night_high(client: AsyncClient):
    resp = await client.post("/score", json=FRAUD_NIGHT_TXN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["score"] > 0.5, f"Night+high fraud scored low: {body['score']}"


@pytest.mark.anyio
async def test_score_response_schema(client: AsyncClient):
    resp = await client.post("/score", json=LEGIT_TXN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    for field in ("txn_id", "score", "is_fraud", "reasons", "latency_ms", "model_version"):
        assert field in body, f"Missing field: {field}"


@pytest.mark.anyio
async def test_score_reasons_non_empty(client: AsyncClient):
    resp = await client.post("/score", json=FRAUD_NIGHT_TXN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["reasons"], list)
    assert len(body["reasons"]) >= 1, f"Expected reasons, got: {body['reasons']}"
    assert all(isinstance(r, str) for r in body["reasons"])


@pytest.mark.anyio
async def test_score_latency_under_1s(client: AsyncClient):
    resp = await client.post("/score", json=LEGIT_TXN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["latency_ms"] < 1000, f"Latency {body['latency_ms']:.1f}ms exceeds 1s"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_score_invalid_amount(client: AsyncClient):
    bad = {**LEGIT_TXN, "amount": -100.0}
    resp = await client.post("/score", json=bad, headers=HEADERS)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_score_invalid_country_code(client: AsyncClient):
    bad = {**LEGIT_TXN, "sender_country": "INVALID"}
    resp = await client.post("/score", json=bad, headers=HEADERS)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_score_missing_field(client: AsyncClient):
    bad = {k: v for k, v in LEGIT_TXN.items() if k != "txn_id"}
    resp = await client.post("/score", json=bad, headers=HEADERS)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_stats_after_scoring(client: AsyncClient):
    for i in range(3):
        txn = {**LEGIT_TXN, "txn_id": f"stats_test_{i}"}
        await client.post("/score", json=txn, headers=HEADERS)
    resp = await client.get("/stats", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_scored"] >= 3


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unauthorized_no_key(client: AsyncClient):
    resp = await client.post("/score", json=LEGIT_TXN)
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_unauthorized_wrong_key(client: AsyncClient):
    resp = await client.post("/score", json=LEGIT_TXN,
                             headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
