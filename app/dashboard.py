"""
app/dashboard.py
Streamlit real-time fraud monitoring dashboard.

Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALERTS_FILE    = Path("data/alerts.jsonl")
THRESHOLD_FILE = Path("models/threshold.json")
SHAP_FILE      = Path("reports/shap_importance.csv")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="",
)

st_autorefresh(interval=2000, limit=None, key="fraud_dashboard_refresh")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=2)
def load_alerts() -> pd.DataFrame:
    """Load alerts from JSONL file. Returns empty DataFrame if file absent."""
    if not ALERTS_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "txn_id", "score", "is_fraud",
                                     "latency_ms", "reasons", "model_version"])
    rows = []
    with open(ALERTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_shap_importance() -> pd.DataFrame:
    if SHAP_FILE.exists():
        return pd.read_csv(SHAP_FILE)
    return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def load_thresholds() -> dict:
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"fpr_threshold": 0.5, "cost_threshold": 0.3}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

thresholds = load_thresholds()
default_thr = float(thresholds.get("fpr_threshold", 0.5))
cost_thr    = float(thresholds.get("cost_threshold", 0.3))

with st.sidebar:
    st.header("Controls")

    mode = st.radio(
        "Threshold mode",
        options=["FPR-based", "Cost-optimised"],
        index=0,
    )
    active_threshold = default_thr if mode == "FPR-based" else cost_thr

    threshold = st.slider(
        "Score threshold",
        min_value=0.40,
        max_value=0.95,
        value=float(active_threshold),
        step=0.01,
    )

    corridor_filter = st.multiselect(
        "Filter by corridor (receiver country)",
        options=["KE", "NG", "PK", "VN", "CM", "All"],
        default=["All"],
    )

    date_range = st.date_input(
        "Date range",
        value=(
            (datetime.now(timezone.utc) - timedelta(days=7)).date(),
            datetime.now(timezone.utc).date(),
        ),
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

alerts_df = load_alerts()
shap_df   = load_shap_importance()

# ---------------------------------------------------------------------------
# Row 1 — KPI metric cards
# ---------------------------------------------------------------------------

st.title(" Real-Time Fraud Detection Dashboard")

now = datetime.now(timezone.utc)
cutoff_5m = now - timedelta(minutes=5)

if not alerts_df.empty:
    total_scored = len(alerts_df)
    alerts_fired = int((alerts_df["score"] >= threshold).sum() if "score" in alerts_df.columns else 0)
    alert_rate   = alerts_fired / total_scored * 100 if total_scored > 0 else 0
    avg_latency  = float(alerts_df["latency_ms"].mean()) if "latency_ms" in alerts_df.columns else 0.0

    # Deltas vs last 5 minutes
    recent  = alerts_df[alerts_df["timestamp"] >= cutoff_5m]
    prev_5m = alerts_df[
        (alerts_df["timestamp"] < cutoff_5m) &
        (alerts_df["timestamp"] >= cutoff_5m - timedelta(minutes=5))
    ]

    r_total  = len(recent)
    p_total  = len(prev_5m)
    r_alerts = int((recent["score"] >= threshold).sum()) if not recent.empty else 0
    p_alerts = int((prev_5m["score"] >= threshold).sum()) if not prev_5m.empty else 0
    r_latency = float(recent["latency_ms"].mean()) if not recent.empty else 0.0
    p_latency = float(prev_5m["latency_ms"].mean()) if not prev_5m.empty else 0.0

    delta_total   = r_total - p_total
    delta_alerts  = r_alerts - p_alerts
    delta_rate    = (r_alerts / r_total * 100 - p_alerts / p_total * 100
                     if r_total > 0 and p_total > 0 else 0)
    delta_latency = r_latency - p_latency
else:
    total_scored = alerts_fired = 0
    alert_rate   = avg_latency = 0.0
    delta_total  = delta_alerts = delta_rate = delta_latency = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Txns Scored",  f"{total_scored:,}",
            delta=f"{delta_total:+d}" if delta_total != 0 else None)
col2.metric("Alerts Fired",       f"{alerts_fired:,}",
            delta=f"{delta_alerts:+d}" if delta_alerts != 0 else None,
            delta_color="inverse")
col3.metric("Alert Rate %",       f"{alert_rate:.2f}%",
            delta=f"{delta_rate:+.2f}%" if delta_rate != 0 else None,
            delta_color="inverse")
col4.metric("Avg Latency ms",     f"{avg_latency:.0f}",
            delta=f"{delta_latency:+.0f}" if delta_latency != 0 else None,
            delta_color="inverse")

# ---------------------------------------------------------------------------
# Row 2 — Real-time alerts per minute (rolling 30 min)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Alerts / minute — rolling 30-minute window")

if not alerts_df.empty and "score" in alerts_df.columns:
    window_start = now - timedelta(minutes=30)
    window_df = alerts_df[alerts_df["timestamp"] >= window_start].copy()
    if not window_df.empty:
        flagged = window_df[window_df["score"] >= threshold].copy()
        flagged["minute"] = flagged["timestamp"].dt.floor("T")
        alerts_per_min = (
            flagged.groupby("minute").size()
            .reindex(
                pd.date_range(window_start.replace(second=0, microsecond=0),
                              now, freq="T", tz="UTC"),
                fill_value=0,
            )
            .reset_index(name="alerts")
        )
        alerts_per_min.columns = ["time", "alerts"]
        st.line_chart(alerts_per_min.set_index("time")["alerts"])
    else:
        st.info("No alerts in the last 30 minutes.")
else:
    st.info("No alert data yet. Start the API and score some transactions.")

# ---------------------------------------------------------------------------
# Row 3 — SHAP feature importance bar chart
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Top fraud drivers (SHAP)")

if not shap_df.empty:
    top_shap = shap_df.head(15).sort_values("mean_abs_shap")
    st.bar_chart(
        top_shap.set_index("feature")["mean_abs_shap"],
        horizontal=True,
    )
else:
    st.info("SHAP importance not available. Run `src/train_pipeline.py` first.")

# ---------------------------------------------------------------------------
# Row 4 — Alert table (last 50, color-coded)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Recent Alerts (last 50, sorted by score)")

if not alerts_df.empty and "score" in alerts_df.columns:
    # Apply corridor filter
    display_df = alerts_df.copy()

    # Sort by score desc and take last 50
    display_df = display_df.sort_values("score", ascending=False).head(50)

    # Format for display
    display_df["Time"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["Score"] = display_df["score"].round(4)
    display_df["Fraud?"] = display_df.get("is_fraud", False).map({True: "YES", False: "no"})
    display_df["Reasons"] = display_df.get("reasons", pd.Series()).apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )

    cols = ["Time", "Score", "Fraud?", "Reasons"]
    table_df = display_df[[c for c in cols if c in display_df.columns]].copy()

    def _highlight(row):
        score = row.get("Score", 0)
        if score >= 0.9:
            return ["background-color: #FDECEA"] * len(row)
        if score >= 0.7:
            return ["background-color: #FEF9E7"] * len(row)
        return ["background-color: #FFFDE7"] * len(row)

    styled = table_df.style.apply(_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, height=400)

    # CSV export
    csv_bytes = alerts_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export all alerts to CSV",
        data=csv_bytes,
        file_name="alerts_export.csv",
        mime="text/csv",
    )
else:
    st.info("No alert data yet.")

# ---------------------------------------------------------------------------
# Row 5 — Business impact footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Estimated Business Impact")

COST_FN    = 250_000   # avg fraud loss
COST_FP    = 50        # analyst review cost
PRECISION  = 0.76      # from reports/metrics.md (ensemble Prec@top1%)

losses_prevented = alerts_fired * PRECISION * COST_FN
fp_review_cost   = alerts_fired * (1 - PRECISION) * COST_FP
net_saving       = losses_prevented - fp_review_cost

col_a, col_b, col_c = st.columns(3)
col_a.metric("Est. losses prevented", f"€{losses_prevented:,.0f}")
col_b.metric("Est. FP review cost",   f"€{fp_review_cost:,.0f}")
col_c.metric("Est. net saving",       f"€{net_saving:,.0f}")

st.caption(
    f"Assumptions: avg fraud = €{COST_FN:,}, analyst review = €{COST_FP}, "
    f"model precision ~{PRECISION:.0%}. "
    "Source: reports/business.md"
)
