"""
src/features.py
Offline feature pipeline — batch feature computation over a sorted DataFrame.

All 15 required features plus 5 advanced features.
Online mirrors are in app/feature_state.py — logic MUST stay identical.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lookup dictionaries (shared with app/feature_state.py — keep in sync)
# ---------------------------------------------------------------------------

CORRIDOR_RISK: dict[str, float] = {
    # Key format: f"{sender_country}_{receiver_country}"
    # EUR senders → KES (Kenya)
    "DE_KE": 1.0, "FR_KE": 1.0, "NL_KE": 1.0, "BE_KE": 1.0, "IE_KE": 1.0,
    "LU_KE": 1.0, "AT_KE": 1.0, "FI_KE": 1.0,
    # USD/US sender → NGN (Nigeria)
    "US_NG": 0.9,
    # IE → NG (direct high-risk)
    "IE_NG": 1.0,
    # EUR senders → NG
    "DE_NG": 0.9, "FR_NG": 0.9, "NL_NG": 0.9, "BE_NG": 0.9,
    # GBP → NG
    "GB_NG": 0.9,
    # EUR senders → PK (Pakistan)
    "DE_PK": 0.85, "FR_PK": 0.85, "NL_PK": 0.85, "BE_PK": 0.85, "IE_PK": 0.85,
    # USD → VN (Vietnam)
    "US_VN": 0.7,
    # EUR senders → CM (Cameroon)
    "DE_CM": 0.8, "FR_CM": 0.8, "NL_CM": 0.8, "BE_CM": 0.8,
}
CORRIDOR_RISK_DEFAULT = 0.1

MCC_RISK: dict[str, float] = {
    "6012": 1.0,   # financial institutions
    "6051": 1.0,   # quasi-cash
    "4829": 0.8,   # wire transfer
    "6211": 0.7,   # securities brokers
}
MCC_RISK_DEFAULT = 0.1

NIGHT_HOURS: frozenset[int] = frozenset({22, 23, 0, 1, 2, 3})


def _corridor_key(sender_country: str, receiver_country: str) -> str:
    """Build corridor key: f'{sender_country}_{receiver_country}'."""
    return f"{str(sender_country).upper()}_{str(receiver_country).upper()}"


# ---------------------------------------------------------------------------
# Core 15 features — offline batch computation
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 required features for a time-sorted DataFrame.

    The DataFrame MUST be sorted by timestamp in ascending order before calling
    this function (enforced by an assertion).

    Returns a new DataFrame with one column per feature.
    """
    assert df["timestamp"].is_monotonic_increasing, (
        "DataFrame must be sorted by timestamp before compute_features()"
    )

    n = len(df)

    # Output arrays — pre-allocate
    velocity_1h        = np.zeros(n, dtype=np.float64)
    velocity_24h       = np.zeros(n, dtype=np.float64)
    amount_zscore_30d  = np.zeros(n, dtype=np.float64)
    log_amount         = np.zeros(n, dtype=np.float64)
    is_night           = np.zeros(n, dtype=np.int8)
    hour_of_day        = np.zeros(n, dtype=np.int8)
    day_of_week        = np.zeros(n, dtype=np.int8)
    is_cross_border    = np.zeros(n, dtype=np.int8)
    corridor_risk      = np.zeros(n, dtype=np.float64)
    device_changed_24h = np.zeros(n, dtype=np.int8)
    ip_mismatch        = np.zeros(n, dtype=np.int8)
    mcc_risk           = np.zeros(n, dtype=np.float64)
    receiver_fan_in_24h = np.zeros(n, dtype=np.float64)
    rule_night_high    = np.zeros(n, dtype=np.int8)
    rule_corridor_high = np.zeros(n, dtype=np.int8)

    # Per-sender rolling state
    # deque of timestamps for velocity windows
    sender_ts_1h:  dict[str, deque]  = defaultdict(lambda: deque())
    sender_ts_24h: dict[str, deque]  = defaultdict(lambda: deque())
    # Welford online stats for amount_zscore_30d
    sender_count:  dict[str, int]    = defaultdict(int)
    sender_mean:   dict[str, float]  = defaultdict(float)
    sender_M2:     dict[str, float]  = defaultdict(float)
    sender_ts_30d: dict[str, deque]  = defaultdict(lambda: deque())  # (ts, amount) pairs
    # Last device per sender in 24h
    sender_last_device: dict[str, tuple] = {}   # {sender_id: (ts, device_id)}
    # Per-receiver: set of senders in last 24h
    receiver_senders_24h: dict[str, deque] = defaultdict(lambda: deque())  # (ts, sender_id)

    timestamps = df["timestamp"].values  # numpy datetime64
    amounts    = df["amount"].values.astype(float)
    sender_ids   = df["sender_id"].values.astype(str)
    receiver_ids = df["receiver_id"].values.astype(str)
    sender_countries  = df["sender_country"].values.astype(str)
    receiver_countries = df["receiver_country"].values.astype(str)
    device_ids   = df["device_id"].values.astype(str)
    ip_countries = df["ip_country"].values.astype(str)
    mccs         = df["mcc"].values.astype(str)

    # Convert timestamps to Python datetimes for timedelta arithmetic
    ts_dt = pd.to_datetime(timestamps).to_pydatetime()

    _1h  = timedelta(hours=1)
    _24h = timedelta(hours=24)
    _30d = timedelta(days=30)

    for i in range(n):
        ts  = ts_dt[i]
        sid = sender_ids[i]
        rid = receiver_ids[i]
        amt = amounts[i]
        sc  = sender_countries[i]
        rc  = receiver_countries[i]
        dev = device_ids[i]
        ipc = ip_countries[i]
        mcc_val = mccs[i]

        # ---- Velocity 1h -------------------------------------------------------
        q1h = sender_ts_1h[sid]
        while q1h and (ts - q1h[0]) > _1h:
            q1h.popleft()
        q1h.append(ts)
        velocity_1h[i] = len(q1h)

        # ---- Velocity 24h ------------------------------------------------------
        q24h = sender_ts_24h[sid]
        while q24h and (ts - q24h[0]) > _24h:
            q24h.popleft()
        q24h.append(ts)
        velocity_24h[i] = len(q24h)

        # ---- Amount z-score 30d (Welford online algorithm) ---------------------
        # First, expire amounts older than 30 days
        q30d = sender_ts_30d[sid]
        while q30d and (ts - q30d[0][0]) > _30d:
            old_ts, old_amt = q30d.popleft()
            # Welford downdate (remove old observation)
            old_count = sender_count[sid]
            if old_count > 1:
                old_mean = sender_mean[sid]
                new_count = old_count - 1
                new_mean = (old_mean * old_count - old_amt) / new_count
                sender_M2[sid] -= (old_amt - old_mean) * (old_amt - new_mean)
                sender_M2[sid] = max(0.0, sender_M2[sid])
                sender_count[sid] = new_count
                sender_mean[sid] = new_mean
            else:
                sender_count[sid] = 0
                sender_mean[sid] = 0.0
                sender_M2[sid] = 0.0

        # Welford update (add current observation)
        sender_count[sid] += 1
        cnt = sender_count[sid]
        delta = amt - sender_mean[sid]
        sender_mean[sid] += delta / cnt
        delta2 = amt - sender_mean[sid]
        sender_M2[sid] += delta * delta2
        q30d.append((ts, amt))

        if cnt >= 2:
            variance = sender_M2[sid] / (cnt - 1)
            std = variance ** 0.5
            amount_zscore_30d[i] = (amt - sender_mean[sid]) / std if std > 1e-9 else 0.0
        else:
            amount_zscore_30d[i] = 0.0

        # ---- Simple inline features --------------------------------------------
        log_amount[i]   = float(np.log1p(amt))
        h = ts.hour
        is_night[i]     = 1 if h in NIGHT_HOURS else 0
        hour_of_day[i]  = h
        day_of_week[i]  = ts.weekday()  # 0=Mon
        is_cross_border[i] = 1 if sc != rc else 0

        # ---- Corridor risk -----------------------------------------------------
        ck = _corridor_key(sc, rc)
        corridor_risk[i] = CORRIDOR_RISK.get(ck, CORRIDOR_RISK_DEFAULT)

        # ---- Device changed in 24h --------------------------------------------
        prev = sender_last_device.get(sid)
        if prev is None:
            device_changed_24h[i] = 0
        else:
            prev_ts, prev_dev = prev
            if (ts - prev_ts) <= _24h and prev_dev != dev:
                device_changed_24h[i] = 1
            else:
                device_changed_24h[i] = 0
        sender_last_device[sid] = (ts, dev)

        # ---- IP mismatch -------------------------------------------------------
        ip_mismatch[i] = 1 if ipc != sc else 0

        # ---- MCC risk ----------------------------------------------------------
        mcc_risk[i] = MCC_RISK.get(mcc_val, MCC_RISK_DEFAULT)

        # ---- Receiver fan-in 24h -----------------------------------------------
        qr = receiver_senders_24h[rid]
        while qr and (ts - qr[0][0]) > _24h:
            qr.popleft()
        qr.append((ts, sid))
        # count distinct senders in window
        distinct_senders = len({s for _, s in qr})
        receiver_fan_in_24h[i] = distinct_senders

        # ---- Rule features (derived) ------------------------------------------
        rule_night_high[i]    = 1 if (is_night[i] == 1 and amt > 500_000) else 0
        rule_corridor_high[i] = 1 if corridor_risk[i] > 0.7 else 0

    return pd.DataFrame({
        "velocity_1h":         velocity_1h,
        "velocity_24h":        velocity_24h,
        "amount_zscore_30d":   amount_zscore_30d,
        "log_amount":          log_amount,
        "is_night":            is_night,
        "hour_of_day":         hour_of_day,
        "day_of_week":         day_of_week,
        "is_cross_border":     is_cross_border,
        "corridor_risk":       corridor_risk,
        "device_changed_24h":  device_changed_24h,
        "ip_mismatch":         ip_mismatch,
        "mcc_risk":            mcc_risk,
        "receiver_fan_in_24h": receiver_fan_in_24h,
        "rule_night_high":     rule_night_high,
        "rule_corridor_high":  rule_corridor_high,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 5 Advanced features (add-on after core 15)
# ---------------------------------------------------------------------------

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5 advanced features for a time-sorted DataFrame.
    Returns a DataFrame with 5 additional columns to be concatenated with
    the output of compute_features().
    """
    assert df["timestamp"].is_monotonic_increasing

    n = len(df)
    inter_arrival_seconds    = np.zeros(n, dtype=np.float64)
    amount_round_flag        = np.zeros(n, dtype=np.int8)
    benford_deviation        = np.zeros(n, dtype=np.float64)
    receiver_country_novelty = np.zeros(n, dtype=np.int8)
    time_since_last_high_value = np.full(n, np.nan, dtype=np.float64)

    timestamps  = df["timestamp"].values
    amounts     = df["amount"].values.astype(float)
    sender_ids  = df["sender_id"].values.astype(str)
    rcv_countries = df["receiver_country"].values.astype(str)

    ts_dt = pd.to_datetime(timestamps).to_pydatetime()

    sender_last_ts:   dict[str, object]   = {}
    sender_amounts:   dict[str, list]     = defaultdict(list)
    sender_rcv_ctry:  dict[str, set]      = defaultdict(set)
    sender_last_hv:   dict[str, object]   = {}   # last ts of high-value txn (>100k)

    for i in range(n):
        ts  = ts_dt[i]
        sid = sender_ids[i]
        amt = amounts[i]
        rc  = rcv_countries[i]

        # inter_arrival_seconds
        if sid in sender_last_ts:
            inter_arrival_seconds[i] = (ts - sender_last_ts[sid]).total_seconds()
        else:
            inter_arrival_seconds[i] = 0.0
        sender_last_ts[sid] = ts

        # amount_round_flag
        amount_round_flag[i] = 1 if (amt > 10_000 and amt % 1_000 == 0) else 0

        # benford_deviation — deviation from Benford's Law leading digit
        sender_amounts[sid].append(amt)
        hist = sender_amounts[sid]
        if len(hist) >= 20:
            # expected Benford P(d) = log10(1 + 1/d)
            benford_exp = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
            leading_digits = np.array([int(str(abs(a)).lstrip("0").lstrip(".")[0])
                                        for a in hist if a > 0])
            if len(leading_digits) > 0:
                counts = np.bincount(leading_digits, minlength=10)[1:10]
                observed = counts / counts.sum() if counts.sum() > 0 else benford_exp
                benford_deviation[i] = float(np.sum(np.abs(observed - benford_exp)))
        # else default 0

        # receiver_country_novelty
        if rc not in sender_rcv_ctry[sid]:
            receiver_country_novelty[i] = 1
            sender_rcv_ctry[sid].add(rc)
        else:
            receiver_country_novelty[i] = 0

        # time_since_last_high_value (hours since last txn > €100,000)
        if sid in sender_last_hv:
            time_since_last_high_value[i] = (ts - sender_last_hv[sid]).total_seconds() / 3600.0
        else:
            time_since_last_high_value[i] = np.nan
        if amt > 100_000:
            sender_last_hv[sid] = ts

    return pd.DataFrame({
        "inter_arrival_seconds":       inter_arrival_seconds,
        "amount_round_flag":           amount_round_flag,
        "benford_deviation":           benford_deviation,
        "receiver_country_novelty":    receiver_country_novelty,
        "time_since_last_high_value":  time_since_last_high_value,
    }, index=df.index)
