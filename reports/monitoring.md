# Drift Monitoring Report

Baseline window: first 30 days of dataset (35,572 rows)  
Recent window:   last 30 days of dataset (34,261 rows)

## Feature PSI Table

| Feature              | PSI    | Status                |
|----------------------|--------|-----------------------|
| amount_zscore_30d    | 0.0603 | Stable                |
| velocity_1h          | 0.0109 | Stable                |
| corridor_risk        | 0.0008 | Stable                |
| is_night             | 0.0002 | Stable                |
| hour_of_day          | 0.0013 | Stable                |

## PSI Interpretation
- PSI < 0.10  → Stable (no action)
- PSI 0.10–0.25 → Moderate shift (monitor closely)
- PSI > 0.25  → Major shift (trigger retrain)

## Score Distribution Shift
| Metric              | Value     |
|---------------------|-----------|
| Mean score shift    | -0.0223   |
| Score PSI           | 0.0624    |
| Flag                | OK |

See `reports/img/score_drift.png` for visual comparison.
