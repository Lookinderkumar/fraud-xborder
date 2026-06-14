# Business Case — Real-Time Fraud Detection

## Assumptions (Citi-scale)

| Parameter                     | Value                  |
|-------------------------------|------------------------|
| Daily SWIFT/SEPA volume       | 500,000 transactions   |
| Fraud rate                    | 0.5% = 2,500 fraud/day |
| Average fraudulent amount     | EUR 250,000            |
| Daily fraud exposure          | EUR 625,000,000        |
| Analyst cost per FP reviewed  | EUR 50 (30 min @ EUR 100/hr) |
| Rule-only baseline recall     | 40%                    |

## Model Performance (from reports/metrics.md)

Ensemble model (XGBoost + Isolation Forest):
- Recall @ FPR=0.5%: **72.5%**
- Precision @ Top 1%: **94.0%**

Using FPR threshold (0.793):
- Recall: 72.5%
- Precision: 76.1%

## Daily ROI Calculations

### ML Model (Ensemble)

```
recall         = 0.725
precision      = 0.761

fraud_caught_day      = 2,500 x 0.725             = 1,813 txns
losses_prevented_day  = 1,813 x EUR 250,000        = EUR 453,125,000

total_alerts_day      = 2,500 x 0.005 / 0.761     = 16 alerts (approx.)
                        (using: fraud_caught / precision)
                      = 1,813 / 0.761              = 2,382 total alerts
fp_alerts_day         = 2,382 x (1 - 0.761)       = 569 false positives
fp_review_cost_day    = 569 x EUR 50               = EUR 28,450

net_saving_day        = EUR 453,125,000 - EUR 28,450 = EUR 453,096,550
net_saving_year       = EUR 453,096,550 x 250        = EUR 113,274,137,500
```

### Rules-Only Baseline (40% recall, 31% precision)

```
fraud_caught_day      = 2,500 x 0.40              = 1,000 txns
losses_prevented_day  = 1,000 x EUR 250,000        = EUR 250,000,000

fp_alerts_day         = (1,000 / 0.31) x (1-0.31) = 2,226 false positives
fp_review_cost_day    = 2,226 x EUR 50             = EUR 111,290

net_saving_day        = EUR 250,000,000 - EUR 111,290 = EUR 249,888,710
```

### Incremental Value of ML over Rules

```
incremental_daily     = EUR 453,096,550 - EUR 249,888,710 = EUR 203,207,840
incremental_annual    = EUR 203,207,840 x 250             = EUR 50,801,960,000
```

## Summary Table

| Metric                    | Rules Only       | ML Ensemble (XGB+IF) | Uplift        |
|---------------------------|------------------|----------------------|---------------|
| Recall                    | 40.0%            | 72.5%                | +32.5 pp      |
| Precision                 | 31.0%            | 76.1%                | +45.1 pp      |
| Fraud caught/day          | 1,000            | 1,813                | +813          |
| Losses prevented/day      | EUR 250,000,000  | EUR 453,125,000      | +EUR 203,125,000 |
| FP review cost/day        | EUR 111,290      | EUR 28,450           | -EUR 82,840   |
| Net saving/day            | EUR 249,888,710  | EUR 453,096,550      | +EUR 203,207,840 |
| Net saving/year           | EUR 62,472,177,500 | EUR 113,274,137,500 | +EUR 50,801,960,000 |

## Cost Model Parameters

- cost_FN = EUR 250,000 (average fraud loss per transaction)
- cost_FP = EUR 50 (analyst review cost per false positive)
- cost_threshold = 0.113 (minimises expected daily cost)
- fpr_threshold = 0.793 (maintains FPR <= 0.5%)

## Latency SLA

- p50 latency: < 1,000 ms (target met)
- p95 latency: < 2,000 ms (target met)
- Throughput: > 10 txn/s (target met)

All model metrics and thresholds are sourced from `reports/metrics.md` and
`models/threshold.json`. Recall/precision figures are from the ensemble model
(XGBoost + Isolation Forest) evaluated on the held-out temporal test set
(last 20% of 90-day synthetic dataset by timestamp).
