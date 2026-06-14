"""
app/feature_state.py
Online in-memory feature state — mirrors src/features.py exactly.

All rolling-window logic (velocity, z-score, device, fan-in) must produce
results identical to the offline batch pipeline to within 1e-6.

Design:
  - collections.deque for velocity windows (auto-expiry on access)
  - Welford online algorithm for running mean/std (amount_zscore_30d)
  - Dict of last-device for device_changed_24h
  - Dict of deque[(ts, sender)] for receiver_fan_in_24h
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

# Import lookup dicts from features.py — single source of truth
from src.features import (
    CORRIDOR_RISK,
    CORRIDOR_RISK_DEFAULT,
    MCC_RISK,
    MCC_RISK_DEFAULT,
    NIGHT_HOURS,
    _corridor_key,
)

import numpy as np

_1H  = timedelta(hours=1)
_24H = timedelta(hours=24)
_30D = timedelta(days=30)


class FeatureState:
    """
    Maintains rolling per-sender and per-receiver state for online scoring.

    Usage:
        state = FeatureState()
        # For each incoming transaction (in chronological order):
        features = state.get_features(txn_dict)
        state.update(txn_dict)

    Note: get_features() must be called BEFORE update() so that the stateful
    features reflect state PRIOR to the current transaction (matching offline
    batch semantics where the current row reads state built from prior rows).
    """

    def __init__(self) -> None:
        # --- Velocity 1h: sender → deque of timestamps
        self._v1h:  dict[str, deque] = defaultdict(deque)
        # --- Velocity 24h: sender → deque of timestamps
        self._v24h: dict[str, deque] = defaultdict(deque)

        # --- Welford stats for amount z-score 30d ---
        # sender → (count, mean, M2)
        self._w_count: dict[str, int]   = defaultdict(int)
        self._w_mean:  dict[str, float] = defaultdict(float)
        self._w_M2:    dict[str, float] = defaultdict(float)
        # sender → deque of (ts, amount) for expiry
        self._w_hist:  dict[str, deque] = defaultdict(deque)

        # --- Device tracking: sender → (ts, device_id)
        self._last_device: dict[str, tuple] = {}

        # --- Receiver fan-in: receiver → deque of (ts, sender_id)
        self._fan_in: dict[str, deque] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self, txn: dict[str, Any]) -> dict[str, float]:
        """
        Compute the 5 stateful features for a transaction using current state.
        Call this BEFORE update() to match offline semantics.

        txn must contain:
            timestamp      : datetime (tz-aware or naive UTC)
            amount         : float
            sender_id      : str
            receiver_id    : str
            sender_country : str
            receiver_country: str
            device_id      : str
            ip_country     : str
            mcc            : str

        Returns dict with all 15 feature names.
        """
        ts  = _coerce_ts(txn["timestamp"])
        sid = str(txn["sender_id"])
        rid = str(txn["receiver_id"])
        amt = float(txn["amount"])
        sc  = str(txn["sender_country"])
        rc  = str(txn["receiver_country"])
        dev = str(txn["device_id"])
        ipc = str(txn["ip_country"])
        mcc_val = str(txn["mcc"])

        # --- velocity_1h (BEFORE adding current ts)
        q1h = self._v1h[sid]
        self._expire(q1h, ts, _1H)
        velocity_1h = float(len(q1h)) + 1.0  # +1 for current txn

        # --- velocity_24h
        q24h = self._v24h[sid]
        self._expire(q24h, ts, _24H)
        velocity_24h = float(len(q24h)) + 1.0

        # --- amount_zscore_30d ---
        # Matches offline semantics: z-score is computed AFTER including the
        # current amount. We do a hypothetical Welford step (not persisted)
        # so that update() can do the real one later.
        self._expire_welford(sid, ts)
        cnt_before = self._w_count[sid]
        mean_before = self._w_mean[sid]
        M2_before = self._w_M2[sid]
        cnt_hyp = cnt_before + 1
        delta_hyp = amt - mean_before
        mean_hyp = mean_before + delta_hyp / cnt_hyp
        delta2_hyp = amt - mean_hyp
        M2_hyp = M2_before + delta_hyp * delta2_hyp
        if cnt_hyp >= 2:
            variance_hyp = M2_hyp / (cnt_hyp - 1)
            std_hyp = variance_hyp ** 0.5
            amount_zscore_30d = (amt - mean_hyp) / std_hyp if std_hyp > 1e-9 else 0.0
        else:
            amount_zscore_30d = 0.0

        # --- device_changed_24h
        prev = self._last_device.get(sid)
        if prev is None:
            device_changed_24h = 0.0
        else:
            prev_ts, prev_dev = prev
            if (ts - prev_ts) <= _24H and prev_dev != dev:
                device_changed_24h = 1.0
            else:
                device_changed_24h = 0.0

        # --- receiver_fan_in_24h (BEFORE adding current sender)
        qr = self._fan_in[rid]
        self._expire_tuples(qr, ts, _24H)
        distinct = len({s for _, s in qr}) + (
            0 if any(s == sid for _, s in qr) else 1
        )
        receiver_fan_in_24h = float(distinct)

        # --- Inline (non-stateful) features ---
        h = ts.hour
        is_night    = 1 if h in NIGHT_HOURS else 0
        ck          = _corridor_key(sc, rc)
        corr_risk   = CORRIDOR_RISK.get(ck, CORRIDOR_RISK_DEFAULT)

        return {
            "velocity_1h":          velocity_1h,
            "velocity_24h":         velocity_24h,
            "amount_zscore_30d":    amount_zscore_30d,
            "log_amount":           float(np.log1p(amt)),
            "is_night":             float(is_night),
            "hour_of_day":          float(h),
            "day_of_week":          float(ts.weekday()),
            "is_cross_border":      float(1 if sc != rc else 0),
            "corridor_risk":        corr_risk,
            "device_changed_24h":   device_changed_24h,
            "ip_mismatch":          float(1 if ipc != sc else 0),
            "mcc_risk":             MCC_RISK.get(mcc_val, MCC_RISK_DEFAULT),
            "receiver_fan_in_24h":  receiver_fan_in_24h,
            "rule_night_high":      float(1 if is_night and amt > 500_000 else 0),
            "rule_corridor_high":   float(1 if corr_risk > 0.7 else 0),
        }

    def update(self, txn: dict[str, Any]) -> None:
        """
        Update rolling state with the current transaction.
        Must be called AFTER get_features().
        """
        ts  = _coerce_ts(txn["timestamp"])
        sid = str(txn["sender_id"])
        rid = str(txn["receiver_id"])
        amt = float(txn["amount"])
        dev = str(txn["device_id"])

        # velocity 1h / 24h — append current ts
        self._v1h[sid].append(ts)
        self._v24h[sid].append(ts)

        # Welford update
        self._welford_add(sid, ts, amt)

        # device update
        self._last_device[sid] = (ts, dev)

        # fan-in update
        self._fan_in[rid].append((ts, sid))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expire(q: deque, ts: datetime, window: timedelta) -> None:
        """Remove datetime entries older than window from the left of q."""
        while q and (ts - q[0]) > window:
            q.popleft()

    @staticmethod
    def _expire_tuples(q: deque, ts: datetime, window: timedelta) -> None:
        """Remove (ts, ...) tuple entries older than window from the left of q."""
        while q and (ts - q[0][0]) > window:
            q.popleft()

    def _expire_welford(self, sid: str, ts: datetime) -> None:
        """Expire Welford history older than 30 days and downdate stats."""
        q = self._w_hist[sid]
        while q and (ts - q[0][0]) > _30D:
            old_ts, old_amt = q.popleft()
            cnt = self._w_count[sid]
            if cnt > 1:
                old_mean = self._w_mean[sid]
                new_cnt  = cnt - 1
                new_mean = (old_mean * cnt - old_amt) / new_cnt
                self._w_M2[sid] -= (old_amt - old_mean) * (old_amt - new_mean)
                self._w_M2[sid]  = max(0.0, self._w_M2[sid])
                self._w_count[sid] = new_cnt
                self._w_mean[sid]  = new_mean
            else:
                self._w_count[sid] = 0
                self._w_mean[sid]  = 0.0
                self._w_M2[sid]    = 0.0

    def _welford_add(self, sid: str, ts: datetime, amt: float) -> None:
        """Add one observation to Welford running stats."""
        self._w_count[sid] += 1
        cnt = self._w_count[sid]
        delta = amt - self._w_mean[sid]
        self._w_mean[sid] += delta / cnt
        delta2 = amt - self._w_mean[sid]
        self._w_M2[sid] += delta * delta2
        self._w_hist[sid].append((ts, amt))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _coerce_ts(ts: Any) -> datetime:
    """Ensure ts is a tz-aware datetime (UTC if naive)."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    # pandas Timestamp
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
