# Real-Time Fraud Detection -- SWIFT/SEPA

![Tests](https://github.com/Lookinderkumar/fraud-xborder/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue)

Production-grade, explainable fraud detection for high-value cross-border SWIFT/SEPA
payments -- XGBoost + Isolation Forest ensemble, SHAP reason codes, FastAPI scoring
API, and a live Streamlit monitoring dashboard.

---

## Results

| Model             | ROC-AUC | PR-AUC | Brier  | Recall@FPR0.5% | Prec@Top1% |
|-------------------|---------|--------|--------|----------------|------------|
| LR Baseline       |  0.949  |  0.333 | 0.0748 |         15.5%  |      40.0% |
| XGBoost only      |  0.992  |  0.807 | 0.0342 |         70.9%  |      95.0% |
| Hybrid (R+X)      |  0.973  |  0.290 | 0.0526 |          0.0%  |      27.5% |
| Ensemble (XGB+IF) |  0.991  |  0.805 | 0.0374 |         72.5%  |      94.0% |

**Latency**: p50=17ms, p95=20ms | **Throughput**: 57 txn/s

---

## Architecture

```
Transactions
     |
     v
FastAPI /score  <-- X-API-Key auth, 100 req/min rate limit
     |
     +-- FeatureState (online, in-memory Welford stats + velocity deques)
     |
     +-- XGBoost predict_proba  --|
     +-- Rule engine (night/corridor) -|-> max() -> hybrid
     +-- IsolationForest (8 key features, legit-only trained)
     |
     +-- Ensemble: 0.7 * hybrid + 0.3 * iso
     +-- SHAP TreeExplainer --> reason codes (top 3)
     |
     v
ScoreResponse {score, is_fraud, reasons, latency_ms}
     |
     +-- data/alerts.jsonl  --> Streamlit dashboard (auto-refresh 2s)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python -m data_gen.synth_payments

# 3. Train models
python -m src.train_pipeline

# 4. Start the API
uvicorn app.model_api:app --reload

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

Score a transaction:

```bash
curl -X POST http://localhost:8000/score \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "txn_id": "test_001",
    "timestamp": "2026-01-15T02:30:00+00:00",
    "amount": 800000,
    "currency": "EUR",
    "sender_id": "sender_1",
    "receiver_id": "receiver_1",
    "sender_country": "IE",
    "receiver_country": "NG",
    "device_id": "new_device_999",
    "ip_country": "CN",
    "channel": "SWIFT",
    "mcc": "4829",
    "is_cross_border": true
  }'
```

---

## Business Impact (Citi-scale estimate)

| Metric                    | Rules Only         | ML Ensemble          |
|---------------------------|--------------------|----------------------|
| Fraud caught/day          | 1,000              | 1,813                |
| Losses prevented/day      | EUR 250,000,000    | EUR 453,125,000      |
| FP review cost/day        | EUR 111,290        | EUR 28,450           |
| Net saving/year           | EUR 62.5 billion   | EUR 113.3 billion    |

Incremental value of ML vs rules-only: **+EUR 50.8 billion/year**

See `reports/business.md` for full assumptions and calculations.

---

## Project Structure

```
fraud-xborder/
+-- data_gen/synth_payments.py   # 100k synthetic SWIFT/SEPA payments
+-- src/
|   +-- features.py              # 15+5 feature pipeline (offline batch)
|   +-- anomaly.py               # Isolation Forest layer
|   +-- reasons.py               # SHAP -> human-readable reason codes
|   +-- monitoring.py            # PSI drift detection
+-- app/
|   +-- model_api.py             # FastAPI scoring service
|   +-- feature_state.py         # Online in-memory feature state
|   +-- dashboard.py             # Streamlit real-time dashboard
+-- models/
|   +-- threshold.json           # FPR + cost-optimised thresholds
+-- reports/
|   +-- metrics.md               # All model metrics
|   +-- business.md              # ROI analysis
|   +-- monitoring.md            # PSI drift report
+-- tests/
|   +-- test_features.py         # 30 feature unit tests
|   +-- test_api.py              # 12 API integration tests
|   +-- test_reasons.py          # 16 SHAP reason code tests
+-- .github/workflows/ci.yml     # GitHub Actions CI
```

---

## Fraud Patterns Detected

| Pattern                   | % of Fraud | Signal                                      |
|---------------------------|------------|---------------------------------------------|
| Night high-value          | 30%        | amount > EUR 500k, hour in {22,23,0,1,2,3}  |
| High-risk corridor        | 25%        | IE->NG, EUR->KES, USD->NGN, EUR->PK         |
| Device + IP change (ATO)  | 20%        | New device AND ip_country != sender_country  |
| Structuring burst         | 15%        | 5-10 txns in 40min just below thresholds    |
| Mule account              | 10%        | 20+ distinct senders -> receiver in 2h      |

---

## Explainability

Every scored transaction includes up to 3 SHAP-based reason codes, e.g.:

- "Amount 8x customer 30-day average"
- "High-risk corridor IE->NG"
- "High-value payment at 02:30 (outside business hours)"

Global SHAP importance: `reports/img/shap_summary.png`

---

## Limitations and Next Steps

- Synthetic data only (90-day window, 100k transactions) -- production deployment
  requires real transaction history and compliance review
- Feature state is in-memory -- a production deployment should use Redis for
  multi-instance deployments
- The 5 advanced features (Benford, inter-arrival, etc.) are zeroed at inference
  time due to cold-start; a pre-loaded sender history cache would improve coverage
- Rate limit (100 req/min) is per-IP only -- production would need per-API-key limits

---
