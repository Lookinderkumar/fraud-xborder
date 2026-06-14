"""
tests/test_features.py
Full test suite for src/features.py (offline) and app/feature_state.py (online).

Covers:
  - Velocity 1h / 24h accumulation, expiry, and edge cases
  - Amount z-score (stable sender, outlier)
  - Night flag boundaries
  - Corridor risk lookups
  - Rule features
  - Offline / online parity (100 transactions, all stateful features, delta < 1e-6)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features import compute_features, CORRIDOR_RISK, MCC_RISK
from app.feature_state import FeatureState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_row(
    sender_id: str = "S1",
    receiver_id: str = "R1",
    amount: float = 1000.0,
    ts: datetime | None = None,
    sender_country: str = "DE",
    receiver_country: str = "FR",
    device_id: str = "dev_S1_0",
    ip_country: str = "DE",
    mcc: str = "5999",
    channel: str = "SWIFT",
) -> dict:
    return {
        "txn_id":          "t0",
        "timestamp":       ts or _BASE_TS,
        "amount":          amount,
        "currency":        "EUR",
        "sender_id":       sender_id,
        "receiver_id":     receiver_id,
        "sender_country":  sender_country,
        "receiver_country": receiver_country,
        "device_id":       device_id,
        "ip_country":      ip_country,
        "channel":         channel,
        "mcc":             mcc,
        "is_cross_border": int(sender_country != receiver_country),
        "label_fraud":     0,
    }


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _run_offline(rows: list[dict]) -> pd.DataFrame:
    """Run the offline batch feature pipeline on a list of row dicts."""
    df = _make_df(rows)
    return compute_features(df)


def _run_online(rows: list[dict]) -> list[dict]:
    """Run FeatureState on rows in timestamp order; return list of feature dicts."""
    rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
    state = FeatureState()
    results = []
    for row in rows_sorted:
        feats = state.get_features(row)
        state.update(row)
        results.append(feats)
    return results


# ===========================================================================
# Velocity tests
# ===========================================================================

class TestVelocity:

    def test_velocity_empty_sender(self):
        """First txn for a new sender: velocity_1h=1, velocity_24h=1."""
        rows = [_make_row(sender_id="NEW_SENDER")]
        feats = _run_offline(rows)
        assert feats["velocity_1h"].iloc[0] == 1
        assert feats["velocity_24h"].iloc[0] == 1

    def test_velocity_accumulates(self):
        """3 txns within 30 min → velocity_1h=3 on the 3rd."""
        rows = [
            _make_row(ts=_BASE_TS + timedelta(minutes=0)),
            _make_row(ts=_BASE_TS + timedelta(minutes=10)),
            _make_row(ts=_BASE_TS + timedelta(minutes=20)),
        ]
        feats = _run_offline(rows)
        assert feats["velocity_1h"].iloc[2] == 3

    def test_velocity_window_expires(self):
        """Txn at t=0, then at t=61min: 1st falls out → velocity_1h=1 again."""
        rows = [
            _make_row(ts=_BASE_TS),
            _make_row(ts=_BASE_TS + timedelta(minutes=61)),
        ]
        feats = _run_offline(rows)
        assert feats["velocity_1h"].iloc[1] == 1

    def test_velocity_24h_accumulates(self):
        """3 txns across 12 hours → velocity_24h=3 on the 3rd."""
        rows = [
            _make_row(ts=_BASE_TS + timedelta(hours=0)),
            _make_row(ts=_BASE_TS + timedelta(hours=6)),
            _make_row(ts=_BASE_TS + timedelta(hours=12)),
        ]
        feats = _run_offline(rows)
        assert feats["velocity_24h"].iloc[2] == 3

    def test_velocity_24h_expires(self):
        """Txn at t=0, then at t=25h: 1st falls out → velocity_24h=1."""
        rows = [
            _make_row(ts=_BASE_TS),
            _make_row(ts=_BASE_TS + timedelta(hours=25)),
        ]
        feats = _run_offline(rows)
        assert feats["velocity_24h"].iloc[1] == 1

    def test_velocity_independent_senders(self):
        """Two senders each with 2 txns — velocities stay separate."""
        rows = [
            _make_row(sender_id="S_A", ts=_BASE_TS),
            _make_row(sender_id="S_B", ts=_BASE_TS + timedelta(minutes=1)),
            _make_row(sender_id="S_A", ts=_BASE_TS + timedelta(minutes=2)),
            _make_row(sender_id="S_B", ts=_BASE_TS + timedelta(minutes=3)),
        ]
        feats = _run_offline(rows)
        # Row 2 is S_A's second txn → velocity_1h=2
        assert feats["velocity_1h"].iloc[2] == 2
        # Row 3 is S_B's second txn → velocity_1h=2
        assert feats["velocity_1h"].iloc[3] == 2


# ===========================================================================
# Z-score tests
# ===========================================================================

class TestZScore:

    def test_zscore_stable_sender(self):
        """30 txns all same amount → zscore ≈ 0."""
        rows = [
            _make_row(amount=5000.0, ts=_BASE_TS + timedelta(hours=i))
            for i in range(30)
        ]
        feats = _run_offline(rows)
        assert abs(feats["amount_zscore_30d"].iloc[-1]) < 0.1

    def test_zscore_outlier(self):
        """29 txns at €1,000, then 1 at €10,000 (10× mean) → zscore > 3."""
        rows = [
            _make_row(amount=1000.0, ts=_BASE_TS + timedelta(hours=i))
            for i in range(29)
        ]
        rows.append(_make_row(amount=10_000.0, ts=_BASE_TS + timedelta(hours=29)))
        feats = _run_offline(rows)
        assert feats["amount_zscore_30d"].iloc[-1] > 3.0

    def test_zscore_first_txn_is_zero(self):
        """First txn from a sender: z-score is 0 (no prior history)."""
        rows = [_make_row(sender_id="BRAND_NEW")]
        feats = _run_offline(rows)
        assert feats["amount_zscore_30d"].iloc[0] == 0.0


# ===========================================================================
# Night flag tests
# ===========================================================================

class TestNightFlag:

    def test_night_flag_midnight(self):
        """00:30 → is_night=1."""
        ts = _BASE_TS.replace(hour=0, minute=30)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 1

    def test_night_flag_noon(self):
        """12:00 → is_night=0."""
        ts = _BASE_TS.replace(hour=12, minute=0)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 0

    def test_night_flag_boundary_2159(self):
        """21:59 → is_night=0 (night starts at 22:00)."""
        ts = _BASE_TS.replace(hour=21, minute=59)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 0

    def test_night_flag_22_is_night(self):
        """22:00 → is_night=1."""
        ts = _BASE_TS.replace(hour=22, minute=0)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 1

    def test_night_flag_3am(self):
        """03:00 → is_night=1."""
        ts = _BASE_TS.replace(hour=3, minute=0)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 1

    def test_night_flag_4am_not_night(self):
        """04:00 → is_night=0 (night ends after 03:xx)."""
        ts = _BASE_TS.replace(hour=4, minute=0)
        feats = _run_offline([_make_row(ts=ts)])
        assert feats["is_night"].iloc[0] == 0


# ===========================================================================
# Corridor risk tests
# ===========================================================================

class TestCorridorRisk:

    def test_corridor_risk_eur_kes(self):
        """EUR→KES: corridor_risk=1.0."""
        rows = [_make_row(sender_country="DE", receiver_country="KE")]
        feats = _run_offline(rows)
        assert feats["corridor_risk"].iloc[0] == pytest.approx(1.0)

    def test_corridor_risk_ie_ng(self):
        """IE→NG: corridor_risk=1.0 (direct key IE_NG)."""
        rows = [_make_row(sender_country="IE", receiver_country="NG")]
        feats = _run_offline(rows)
        assert feats["corridor_risk"].iloc[0] == pytest.approx(1.0)

    def test_corridor_risk_gbp_ng(self):
        """GBP→NG: corridor_risk=0.9."""
        rows = [_make_row(sender_country="GB", receiver_country="NG")]
        feats = _run_offline(rows)
        assert feats["corridor_risk"].iloc[0] == pytest.approx(0.9)

    def test_corridor_risk_default(self):
        """DE→FR: default corridor_risk=0.1."""
        rows = [_make_row(sender_country="DE", receiver_country="FR")]
        feats = _run_offline(rows)
        assert feats["corridor_risk"].iloc[0] == pytest.approx(0.1)

    def test_corridor_risk_usd_ngn(self):
        """US→NG: corridor_risk=0.9."""
        rows = [_make_row(sender_country="US", receiver_country="NG")]
        feats = _run_offline(rows)
        assert feats["corridor_risk"].iloc[0] == pytest.approx(0.9)


# ===========================================================================
# Rule feature tests
# ===========================================================================

class TestRuleFeatures:

    def test_rule_night_high_true(self):
        """is_night=1 AND amount=600,000 → rule_night_high=1."""
        ts = _BASE_TS.replace(hour=0, minute=30)
        rows = [_make_row(amount=600_000.0, ts=ts)]
        feats = _run_offline(rows)
        assert feats["rule_night_high"].iloc[0] == 1

    def test_rule_night_high_false_day(self):
        """is_night=0 → rule_night_high=0, even if amount is high."""
        ts = _BASE_TS.replace(hour=12)
        rows = [_make_row(amount=600_000.0, ts=ts)]
        feats = _run_offline(rows)
        assert feats["rule_night_high"].iloc[0] == 0

    def test_rule_night_high_false_small_amount(self):
        """is_night=1, but amount=100,000 (≤500,000) → rule_night_high=0."""
        ts = _BASE_TS.replace(hour=1)
        rows = [_make_row(amount=100_000.0, ts=ts)]
        feats = _run_offline(rows)
        assert feats["rule_night_high"].iloc[0] == 0

    def test_rule_corridor_high_true(self):
        """corridor_risk=1.0 > 0.7 → rule_corridor_high=1."""
        rows = [_make_row(sender_country="DE", receiver_country="KE")]
        feats = _run_offline(rows)
        assert feats["rule_corridor_high"].iloc[0] == 1

    def test_rule_corridor_high_false(self):
        """corridor_risk=0.1 → rule_corridor_high=0."""
        rows = [_make_row(sender_country="DE", receiver_country="FR")]
        feats = _run_offline(rows)
        assert feats["rule_corridor_high"].iloc[0] == 0


# ===========================================================================
# MCC risk test
# ===========================================================================

class TestMCCRisk:

    def test_mcc_risk_6012(self):
        rows = [_make_row(mcc="6012")]
        feats = _run_offline(rows)
        assert feats["mcc_risk"].iloc[0] == pytest.approx(1.0)

    def test_mcc_risk_unknown(self):
        rows = [_make_row(mcc="9999")]
        feats = _run_offline(rows)
        assert feats["mcc_risk"].iloc[0] == pytest.approx(0.1)


# ===========================================================================
# Device changed test
# ===========================================================================

class TestDeviceChanged:

    def test_device_changed_within_24h(self):
        """Same sender, different device, within 24h → device_changed_24h=1."""
        rows = [
            _make_row(sender_id="S1", device_id="dev_S1_0", ts=_BASE_TS),
            _make_row(sender_id="S1", device_id="dev_S1_1", ts=_BASE_TS + timedelta(hours=2)),
        ]
        feats = _run_offline(rows)
        assert feats["device_changed_24h"].iloc[1] == 1

    def test_device_not_changed(self):
        """Same sender, same device → device_changed_24h=0."""
        rows = [
            _make_row(sender_id="S1", device_id="dev_S1_0", ts=_BASE_TS),
            _make_row(sender_id="S1", device_id="dev_S1_0", ts=_BASE_TS + timedelta(hours=2)),
        ]
        feats = _run_offline(rows)
        assert feats["device_changed_24h"].iloc[1] == 0


# ===========================================================================
# Offline / Online parity
# ===========================================================================

class TestOfflineOnlineParity:

    def test_offline_online_parity(self):
        """
        Process 100 transactions via src/features.py (offline) and
        app/feature_state.py (online FeatureState).
        All 5 stateful features must match to within 1e-6 for every row.
        """
        rng = np.random.default_rng(99)
        n = 100

        sender_pool   = [f"S{i}" for i in range(1, 11)]   # 10 senders → repeated txns
        receiver_pool = [f"R{i}" for i in range(1, 6)]
        device_pool   = ["dev_A", "dev_B", "dev_C"]
        countries     = ["DE", "FR", "GB", "IE", "US"]

        rows = []
        for i in range(n):
            sid = rng.choice(sender_pool)
            rid = rng.choice(receiver_pool)
            sc  = rng.choice(countries)
            rc  = rng.choice(countries)
            dev = rng.choice(device_pool)
            ipc = rng.choice(countries)
            amt = float(rng.lognormal(9.0, 1.5))
            ts  = _BASE_TS + timedelta(minutes=int(rng.integers(0, 5_000)))
            rows.append(_make_row(
                sender_id=sid,
                receiver_id=rid,
                amount=amt,
                ts=ts,
                sender_country=sc,
                receiver_country=rc,
                device_id=dev,
                ip_country=ipc,
                mcc=rng.choice(["4829", "6012", "5999", "6211"]),
            ))

        # Sort rows by timestamp (both pipelines must process in same order)
        rows_sorted = sorted(rows, key=lambda r: r["timestamp"])

        # --- Offline ---
        df_offline = _run_offline(rows_sorted)

        # --- Online ---
        online_results = _run_online(rows_sorted)

        STATEFUL = [
            "velocity_1h",
            "velocity_24h",
            "amount_zscore_30d",
            "device_changed_24h",
            "receiver_fan_in_24h",
        ]

        for i, (off_row, on_dict) in enumerate(zip(
            df_offline.itertuples(index=False), online_results
        )):
            for feat in STATEFUL:
                off_val = float(getattr(off_row, feat))
                on_val  = float(on_dict[feat])
                assert abs(off_val - on_val) < 1e-6, (
                    f"Row {i}, feature '{feat}': "
                    f"offline={off_val:.8f}, online={on_val:.8f}, "
                    f"delta={abs(off_val - on_val):.2e}"
                )
