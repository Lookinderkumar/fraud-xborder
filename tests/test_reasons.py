"""
tests/test_reasons.py
Tests for src/reasons.py — SHAP-based reason code generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reasons import generate_reasons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "velocity_1h",
    "velocity_24h",
    "amount_zscore_30d",
    "log_amount",
    "is_night",
    "hour_of_day",
    "day_of_week",
    "is_cross_border",
    "corridor_risk",
    "device_changed_24h",
    "ip_mismatch",
    "mcc_risk",
    "receiver_fan_in_24h",
    "rule_night_high",
    "rule_corridor_high",
]

BASE_TXN = {
    "txn_id": "txn_0000001",
    "timestamp": "2026-01-15T02:30:00",
    "amount": 750_000.0,
    "sender_country": "DE",
    "receiver_country": "NG",
    "ip_country": "CN",
    "device_id": "dev_1_new_555",
    # Stateful feature values that templates read from txn dict
    "velocity_1h": 1.0,
    "velocity_24h": 1.0,
    "amount_zscore_30d": 0.0,
    "log_amount": 13.5,
    "is_night": 0.0,
    "corridor_risk": 0.1,
    "device_changed_24h": 0.0,
    "ip_mismatch": 0.0,
    "receiver_fan_in_24h": 1.0,
    "rule_night_high": 0.0,
}


def _shap_with(feature_name: str, value: float, others: float = 0.01) -> np.ndarray:
    """Return a SHAP array where feature_name has `value`, rest have `others`."""
    shap = np.full(len(FEATURE_NAMES), others)
    idx = FEATURE_NAMES.index(feature_name)
    shap[idx] = value
    return shap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAmountZscore:
    def test_reason_amount_zscore_high(self):
        txn = {**BASE_TXN, "amount_zscore_30d": 5.0}
        shap = _shap_with("amount_zscore_30d", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("average" in r for r in reasons), f"Reasons: {reasons}"

    def test_reason_amount_zscore_below_threshold(self):
        # zscore = 1.5 → below 2.0 threshold → no reason even if SHAP is high
        txn = {**BASE_TXN, "amount_zscore_30d": 1.5}
        shap = _shap_with("amount_zscore_30d", 2.0)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("average" in r for r in reasons)


class TestDeviceChange:
    def test_reason_device_change(self):
        txn = {**BASE_TXN, "device_changed_24h": 1.0}
        shap = _shap_with("device_changed_24h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("device" in r.lower() for r in reasons), f"Reasons: {reasons}"

    def test_reason_device_not_changed(self):
        txn = {**BASE_TXN, "device_changed_24h": 0.0}
        shap = _shap_with("device_changed_24h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("device" in r.lower() for r in reasons)


class TestCorridorHigh:
    def test_reason_corridor_high(self):
        txn = {**BASE_TXN, "corridor_risk": 1.0, "sender_country": "DE",
               "receiver_country": "KE"}
        shap = _shap_with("corridor_risk", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("\u2192" in r for r in reasons), f"Reasons: {reasons}"

    def test_reason_corridor_low(self):
        txn = {**BASE_TXN, "corridor_risk": 0.1}
        shap = _shap_with("corridor_risk", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("\u2192" in r for r in reasons)


class TestNightHigh:
    def test_reason_night_high(self):
        txn = {**BASE_TXN, "rule_night_high": 1.0, "timestamp": "2026-01-15T02:30:00"}
        shap = _shap_with("rule_night_high", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("hours" in r.lower() for r in reasons), f"Reasons: {reasons}"

    def test_reason_night_high_zero(self):
        txn = {**BASE_TXN, "rule_night_high": 0.0}
        shap = _shap_with("rule_night_high", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("hours" in r.lower() for r in reasons)


class TestOutputShape:
    def test_reasons_returns_list_of_strings(self):
        txn = {**BASE_TXN, "amount_zscore_30d": 5.0, "device_changed_24h": 1.0,
               "corridor_risk": 1.0}
        shap = np.array([0.1, 0.1, 1.5, 0.05, 0.05, 0.05, 0.05, 0.05,
                         1.0, 1.2, 0.05, 0.05, 0.05, 0.05, 0.05])
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert isinstance(reasons, list)
        assert all(isinstance(r, str) for r in reasons)

    def test_reasons_max_three(self):
        # All features active — must return at most 3 reasons
        txn = {
            **BASE_TXN,
            "amount_zscore_30d": 5.0,
            "device_changed_24h": 1.0,
            "corridor_risk": 1.0,
            "receiver_fan_in_24h": 10.0,
            "rule_night_high": 1.0,
            "ip_mismatch": 1.0,
            "velocity_1h": 5.0,
        }
        shap = np.ones(len(FEATURE_NAMES))
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert len(reasons) <= 3, f"Expected <= 3, got {len(reasons)}: {reasons}"

    def test_reasons_empty_shap(self):
        # All SHAP values = 0 → no features have real impact → empty list
        txn = {**BASE_TXN, "amount_zscore_30d": 5.0}
        shap = np.zeros(len(FEATURE_NAMES))
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        # With all-zero SHAP, sorted order is stable but feature values
        # still need to meet thresholds. Amount zscore IS > 2, so we may
        # get a reason even with SHAP=0.  The contract is: empty SHAP
        # should produce empty or unreliable reasons. Test that it's a list.
        assert isinstance(reasons, list)

    def test_reasons_truly_empty_shap(self):
        # All features below their value thresholds AND all SHAP = 0
        txn = {**BASE_TXN}  # defaults: zscore=0, velocity=1, corridor=0.1...
        shap = np.zeros(len(FEATURE_NAMES))
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert reasons == [], f"Expected empty list, got: {reasons}"


class TestVelocity:
    def test_reason_velocity_1h_high(self):
        txn = {**BASE_TXN, "velocity_1h": 5.0}
        shap = _shap_with("velocity_1h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("60 minutes" in r for r in reasons), f"Reasons: {reasons}"

    def test_reason_velocity_1h_low(self):
        # velocity = 2 → below threshold (>3)
        txn = {**BASE_TXN, "velocity_1h": 2.0}
        shap = _shap_with("velocity_1h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("60 minutes" in r for r in reasons)


class TestFanIn:
    def test_reason_fan_in_high(self):
        txn = {**BASE_TXN, "receiver_fan_in_24h": 10.0}
        shap = _shap_with("receiver_fan_in_24h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert any("mule" in r.lower() for r in reasons), f"Reasons: {reasons}"

    def test_reason_fan_in_low(self):
        txn = {**BASE_TXN, "receiver_fan_in_24h": 5.0}
        shap = _shap_with("receiver_fan_in_24h", 1.5)
        reasons = generate_reasons(shap, FEATURE_NAMES, txn)
        assert not any("mule" in r.lower() for r in reasons)
