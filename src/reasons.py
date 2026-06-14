"""
src/reasons.py
SHAP-based reason code generator for fraud alerts.

Usage:
    from src.reasons import generate_reasons

    reasons = generate_reasons(shap_row, feature_names, txn_dict)
    # Returns up to 3 human-readable strings explaining why the
    # transaction was flagged.
"""

from __future__ import annotations

import numpy as np


# Mapping from feature name to template function.
# Each template receives (shap_value, feature_value, txn_dict) and returns
# a string or None (None means the feature value doesn't meet its threshold
# and should not generate a reason even if SHAP is high).

def _reason_amount_zscore(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val > 2.0:
        return f"Amount {feat_val:.0f}x customer 30-day average"
    return None


def _reason_velocity_1h(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val > 3:
        return f"{int(feat_val)} transactions in last 60 minutes"
    return None


def _reason_device_changed(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val == 1:
        return "New device seen within 24 hours"
    return None


def _reason_ip_mismatch(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val == 1:
        ip  = txn.get("ip_country", "?")
        sc  = txn.get("sender_country", "?")
        return f"IP country ({ip}) differs from sender country ({sc})"
    return None


def _reason_corridor_risk(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val > 0.7:
        sc = txn.get("sender_country", "?")
        rc = txn.get("receiver_country", "?")
        return f"High-risk corridor {sc}\u2192{rc}"
    return None


def _reason_rule_night_high(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val == 1:
        ts = txn.get("timestamp")
        try:
            from datetime import datetime
            if hasattr(ts, "hour"):
                hour, minute = ts.hour, ts.minute
            elif isinstance(ts, str):
                dt = datetime.fromisoformat(ts)
                hour, minute = dt.hour, dt.minute
            else:
                hour, minute = 0, 0
        except Exception:
            hour, minute = 0, 0
        return f"High-value payment at {hour:02d}:{minute:02d} (outside business hours)"
    return None


def _reason_receiver_fan_in(shap_val: float, feat_val: float, txn: dict) -> str | None:
    if feat_val > 7:
        return f"Receiver collecting from {int(feat_val)} senders today (mule pattern)"
    return None


# Feature name → template function
_TEMPLATES: dict[str, callable] = {
    "amount_zscore_30d":   _reason_amount_zscore,
    "velocity_1h":         _reason_velocity_1h,
    "device_changed_24h":  _reason_device_changed,
    "ip_mismatch":         _reason_ip_mismatch,
    "corridor_risk":       _reason_corridor_risk,
    "rule_night_high":     _reason_rule_night_high,
    "receiver_fan_in_24h": _reason_receiver_fan_in,
}


def generate_reasons(
    shap_values_row: np.ndarray,
    feature_names: list[str],
    txn: dict,
) -> list[str]:
    """
    Generate up to 3 human-readable fraud reason codes from SHAP values.

    Algorithm:
      1. Pair each feature with its SHAP value.
      2. Sort by |SHAP value| descending (most impactful first).
      3. For the top features that have a template, apply the template.
      4. Skip features whose value doesn't meet the template threshold
         (avoids misleading reasons when SHAP is high but value is low).
      5. Return up to 3 non-None reasons.

    Args:
        shap_values_row:  1-D array of SHAP values for one transaction.
        feature_names:    List of feature names matching shap_values_row.
        txn:              Transaction dict (must contain timestamp,
                          ip_country, sender_country, receiver_country).

    Returns:
        List of up to 3 reason strings (empty list if all SHAP = 0).
    """
    if len(shap_values_row) == 0 or len(feature_names) == 0:
        return []

    # Sort features by |SHAP| descending
    paired = sorted(
        zip(feature_names, shap_values_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Build feature value lookup from txn (for non-stateful inline features)
    feat_lookup: dict[str, float] = {}
    for fname, fval in paired:
        # Feature value comes from txn dict if available; otherwise 0
        feat_lookup[fname] = float(txn.get(fname, 0.0))

    reasons: list[str] = []
    for fname, shap_val in paired:
        if len(reasons) >= 3:
            break
        template_fn = _TEMPLATES.get(fname)
        if template_fn is None:
            continue
        feat_val = feat_lookup[fname]
        reason = template_fn(shap_val, feat_val, txn)
        if reason is not None:
            reasons.append(reason)

    return reasons
