"""
src/monitoring.py
Population Stability Index (PSI) drift detection for fraud features.

Usage:
    python -m src.monitoring

Computes PSI for the 5 key features using first 30 days as baseline vs
last 30 days of the dataset, then writes reports/monitoring.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import compute_features, compute_advanced_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
IMG_DIR     = REPORTS_DIR / "img"

# Features to monitor for PSI
MONITOR_FEATURES = [
    "amount_zscore_30d",
    "velocity_1h",
    "corridor_risk",
    "is_night",
    "hour_of_day",
]

# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI).

    PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)

    Interpretation:
        PSI < 0.10  → Stable (no action needed)
        0.10–0.25   → Moderate shift (monitor closely)
        PSI > 0.25  → Major shift (trigger retrain)

    Args:
        expected:  Baseline distribution (e.g. first 30 days).
        actual:    Current distribution (e.g. last 30 days).
        bins:      Number of equal-width bins.
        epsilon:   Small constant to avoid division by zero / log(0).

    Returns:
        PSI as a float.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Build bins from the combined range
    combined = np.concatenate([expected, actual])
    bin_edges = np.linspace(combined.min(), combined.max() + 1e-9, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual,   bins=bin_edges)

    exp_pct = exp_counts / max(len(expected), 1)
    act_pct = act_counts / max(len(actual),   1)

    # Avoid log(0) and division by zero
    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def _psi_label(psi: float) -> str:
    if psi < 0.10:
        return "Stable"
    if psi < 0.25:
        return "Moderate shift"
    return "Major shift — retrain"


# ---------------------------------------------------------------------------
# Score distribution monitoring
# ---------------------------------------------------------------------------

def plot_score_drift(
    baseline_scores: np.ndarray,
    recent_scores: np.ndarray,
    path: Path,
) -> float:
    """
    Plot score distribution comparison (baseline vs recent).
    Returns the mean score shift.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(baseline_scores, bins=50, alpha=0.6, label="Baseline (days 1-30)",
            density=True, color="steelblue")
    ax.hist(recent_scores,   bins=50, alpha=0.6, label="Recent (days 61-90)",
            density=True, color="tomato")
    ax.set_xlabel("Ensemble fraud score")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: baseline vs recent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    shift = float(np.mean(recent_scores) - np.mean(baseline_scores))
    return shift


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_monitoring_md(
    psi_results: dict[str, float],
    score_shift: float,
    score_psi: float,
    n_baseline: int,
    n_recent: int,
) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    lines = [
        "# Drift Monitoring Report\n",
        f"Baseline window: first 30 days of dataset ({n_baseline:,} rows)  ",
        f"Recent window:   last 30 days of dataset ({n_recent:,} rows)\n",
        "## Feature PSI Table\n",
        "| Feature              | PSI    | Status                |",
        "|----------------------|--------|-----------------------|",
    ]
    for feat, psi in psi_results.items():
        label = _psi_label(psi)
        lines.append(f"| {feat:<20} | {psi:.4f} | {label:<21} |")

    lines += [
        "",
        "## PSI Interpretation",
        "- PSI < 0.10  → Stable (no action)",
        "- PSI 0.10–0.25 → Moderate shift (monitor closely)",
        "- PSI > 0.25  → Major shift (trigger retrain)",
        "",
        "## Score Distribution Shift",
        f"| Metric              | Value     |",
        f"|---------------------|-----------|",
        f"| Mean score shift    | {score_shift:+.4f}   |",
        f"| Score PSI           | {score_psi:.4f}    |",
    ]
    flag = " **FLAG: mean shift > 0.05**" if abs(score_shift) > 0.05 else " OK"
    lines.append(f"| Flag                |{flag} |")
    lines += [
        "",
        "See `reports/img/score_drift.png` for visual comparison.",
    ]

    path = REPORTS_DIR / "monitoring.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 50)
    print("DRIFT MONITORING — PSI REPORT")
    print("=" * 50)

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data and computing features ...")
    df = pd.read_parquet("data/payments.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feat_df = compute_features(df)
    adv_df  = compute_advanced_features(df).fillna(0.0)
    feat_df = pd.concat([feat_df, adv_df], axis=1)
    feat_df["timestamp"] = df["timestamp"].values

    # Define windows
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    baseline_end  = t_min + pd.Timedelta(days=30)
    recent_start  = t_max - pd.Timedelta(days=30)

    mask_baseline = df["timestamp"] <= baseline_end
    mask_recent   = df["timestamp"] >= recent_start

    feat_baseline = feat_df[mask_baseline]
    feat_recent   = feat_df[mask_recent]
    n_base = len(feat_baseline)
    n_rec  = len(feat_recent)
    print(f"  Baseline: {n_base:,} rows  |  Recent: {n_rec:,} rows")

    # PSI per feature
    print("\nComputing PSI for top 5 features ...")
    psi_results: dict[str, float] = {}
    for feat in MONITOR_FEATURES:
        exp = feat_baseline[feat].values.astype(float)
        act = feat_recent[feat].values.astype(float)
        psi = compute_psi(exp, act)
        label = _psi_label(psi)
        psi_results[feat] = psi
        print(f"  {feat:<25}: PSI={psi:.4f}  [{label}]")

    # Score distribution shift
    # Load model and compute scores
    print("\nComputing score distribution shift ...")
    try:
        import joblib
        xgb      = joblib.load("models/xgb.pkl")
        detector = joblib.load("models/iforest.pkl")
        import json
        with open("models/threshold.json") as f:
            thresholds = json.load(f)

        from src.anomaly import AnomalyDetector
        feature_cols = [c for c in feat_df.columns if c != "timestamp"]
        X_base = feat_df.loc[mask_baseline, feature_cols].values.astype(np.float64)
        X_rec  = feat_df.loc[mask_recent,  feature_cols].values.astype(np.float64)

        xgb_base = xgb.predict_proba(X_base)[:, 1]
        xgb_rec  = xgb.predict_proba(X_rec)[:,  1]
        iso_base = detector.score(X_base)
        iso_rec  = detector.score(X_rec)

        scores_base = 0.7 * xgb_base + 0.3 * iso_base
        scores_rec  = 0.7 * xgb_rec  + 0.3 * iso_rec

        score_shift = plot_score_drift(scores_base, scores_rec, IMG_DIR / "score_drift.png")
        score_psi   = compute_psi(scores_base, scores_rec, bins=20)
        print(f"  Mean score shift: {score_shift:+.4f}  {'FLAG' if abs(score_shift) > 0.05 else 'OK'}")
        print(f"  Score PSI:        {score_psi:.4f}  [{_psi_label(score_psi)}]")

        _write_monitoring_md(psi_results, score_shift, score_psi, n_base, n_rec)

    except FileNotFoundError as e:
        print(f"  WARNING: Could not load model ({e}). Writing PSI-only report.")
        _write_monitoring_md(psi_results, 0.0, 0.0, n_base, n_rec)


if __name__ == "__main__":
    main()
