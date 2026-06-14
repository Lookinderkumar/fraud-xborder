"""
data_gen/synth_payments.py
Synthetic SWIFT/SEPA payment generator for fraud detection research.
Dissertation project — synthetic data only, no real PII.

Output: data/payments.parquet (100,000 rows, 15 columns)

Fraud patterns injected (~2.5% total):
  1. NIGHT HIGH-VALUE      — 30% of fraud
  2. HIGH-RISK CORRIDOR    — 25% of fraud
  3. DEVICE + IP CHANGE    — 20% of fraud
  4. STRUCTURING BURST     — 15% of fraud
  5. MULE ACCOUNT          — 10% of fraud

Run:
    python -m data_gen.synth_payments
    python -m data_gen.synth_payments --rows 100000 --seed 42 --out data/payments.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SENDERS = 1_000
N_RECEIVERS = 2_000
TOP_SENDER_FRAC = 0.80   # 80% of txns from top 200 senders
TOP_SENDER_N = 200

FRAUD_RATE = 0.025       # 2.5%
WINDOW_DAYS = 90

# Fraud pattern proportions (must sum to 1.0)
PATTERN_FRACS = {
    "night_high":   0.30,
    "corridor":     0.25,
    "device_ip":    0.20,
    "structuring":  0.15,
    "mule":         0.10,
}

# Currency weights: EUR 60%, USD 25%, GBP 10%, other 5%
CURRENCY_WEIGHTS = {
    "EUR": 0.60,
    "USD": 0.25,
    "GBP": 0.10,
    "CHF": 0.02,
    "JPY": 0.02,
    "AUD": 0.01,
}
CURRENCIES = list(CURRENCY_WEIGHTS.keys())
CURRENCY_PROBS = np.array(list(CURRENCY_WEIGHTS.values()))

# Sender country weights (SWIFT traffic-weighted)
SENDER_COUNTRIES = ["DE", "FR", "GB", "IE", "NL", "US", "BE", "LU", "CH", "IT"]
SENDER_COUNTRY_PROBS = np.array([0.20, 0.18, 0.16, 0.10, 0.10, 0.10, 0.06, 0.04, 0.04, 0.02])

# Receiver countries (normal + high-risk subset)
NORMAL_RECEIVER_COUNTRIES = ["DE", "FR", "GB", "IE", "NL", "US", "BE", "LU", "CH", "IT",
                              "ES", "PL", "SE", "DK", "NO", "AT", "FI", "PT"]
HIGH_RISK_COUNTRIES = ["KE", "NG", "PK", "VN", "CM"]

ALL_RECEIVER_COUNTRIES = NORMAL_RECEIVER_COUNTRIES + HIGH_RISK_COUNTRIES

# Receiver country probabilities: high-risk countries are rare in legit traffic.
# 98% of legit receiver txns go to normal countries, 2% to high-risk.
# This keeps corridor_risk as a strong discriminating feature.
_normal_p = 0.98 / len(NORMAL_RECEIVER_COUNTRIES)
_hr_p     = 0.02 / len(HIGH_RISK_COUNTRIES)
RECEIVER_COUNTRY_PROBS = np.array(
    [_normal_p] * len(NORMAL_RECEIVER_COUNTRIES) +
    [_hr_p]     * len(HIGH_RISK_COUNTRIES)
)

# Channel weights: SWIFT 70%, SEPA 25%, CHAPS 5%
CHANNELS = ["SWIFT", "SEPA", "CHAPS"]
CHANNEL_PROBS = np.array([0.70, 0.25, 0.05])

# MCC distribution (weighted)
MCC_LIST = ["6012", "6051", "4829", "6211", "5411", "7011", "4111", "5999"]
MCC_PROBS = np.array([0.10, 0.10, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10])

# High-risk corridor pairs for pattern 2
HIGH_RISK_CORRIDORS = [
    ("DE", "KE"),   # EUR→KES
    ("FR", "KE"),   # EUR→KES
    ("IE", "NG"),   # IE→NG
    ("GB", "NG"),   # GBP→NG
    ("US", "NG"),   # USD→NGN
    ("DE", "PK"),   # EUR→PK
    ("FR", "PK"),   # EUR→PK
    ("NL", "VN"),   # USD→VN
    ("US", "VN"),   # USD→VN
    ("DE", "CM"),   # EUR→CM
    ("FR", "CM"),   # EUR→CM
]

# Night hours (outside business hours)
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sender_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    1,000 unique senders; 80% of txns come from top 200.
    Implements power-law concentration via weighted sampling.
    """
    top = np.arange(1, TOP_SENDER_N + 1)            # senders 1..200
    rest = np.arange(TOP_SENDER_N + 1, N_SENDERS + 1)  # senders 201..1000

    # Weight top 200 so they generate 80% of traffic
    top_weight = TOP_SENDER_FRAC / len(top)
    rest_weight = (1.0 - TOP_SENDER_FRAC) / len(rest)

    all_ids = np.concatenate([top, rest])
    all_weights = np.concatenate([
        np.full(len(top), top_weight),
        np.full(len(rest), rest_weight),
    ])
    all_weights /= all_weights.sum()  # normalise

    return rng.choice(all_ids, size=n, p=all_weights).astype(str).astype(object)


def _make_receiver_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    2,000 unique receivers with power-law distribution (Zipf-like).
    Exponent 0.5 keeps a power-law shape while avoiding extreme concentration —
    this ensures that legitimate per-receiver fan_in stays well below the mule
    detection threshold (~8+ senders/24h), so receiver_fan_in_24h remains a
    strong discriminating feature.
    """
    ids = np.arange(1, N_RECEIVERS + 1)
    # Zipf weights: weight ∝ 1/rank^0.5  (less extreme than 1/rank)
    weights = 1.0 / (ids.astype(float) ** 0.5)
    weights /= weights.sum()
    return rng.choice(ids, size=n, p=weights).astype(str).astype(object)


def _make_timestamps(rng: np.random.Generator, n: int, window_days: int = WINDOW_DAYS) -> pd.DatetimeIndex:
    """
    90-day window with log-normal inter-arrival times (realistic SWIFT traffic).
    Start from 90 days ago and advance by lognormal inter-arrivals.
    """
    start = datetime(2026, 3, 20, tzinfo=timezone.utc) - timedelta(days=window_days)
    # Log-normal inter-arrivals in seconds; mean ~78s → ~1,100 txns/day for 100k in 90d
    # ln(78) ≈ 4.36, sigma=1.2 gives realistic burstiness
    inter_arrivals = rng.lognormal(mean=4.36, sigma=1.2, size=n)
    cum_seconds = np.cumsum(inter_arrivals)
    # Rescale so the series fits within the window
    max_seconds = window_days * 86_400
    cum_seconds = cum_seconds * (max_seconds * 0.95 / cum_seconds[-1])
    ts = [start + timedelta(seconds=float(s)) for s in cum_seconds]
    return pd.DatetimeIndex(ts)


def _make_amounts(rng: np.random.Generator, n: int) -> np.ndarray:
    """log-normal(mu=10, sigma=2.5), clipped €100 – €5M."""
    raw = rng.lognormal(mean=10.0, sigma=2.5, size=n)
    return np.clip(raw, 100.0, 5_000_000.0).round(2)


def _make_device_ids(rng: np.random.Generator, sender_ids: np.ndarray) -> np.ndarray:
    """3 devices per sender normally; each sender has a stable primary device.
    Returns an object-dtype array so later injection can write longer strings
    without numpy silent truncation."""
    unique = np.unique(sender_ids)
    # assign 3 device pool per sender
    sender_to_devices = {s: [f"dev_{s}_{i}" for i in range(3)] for s in unique}
    # pick device 0 (primary) for each transaction
    # Use object dtype to avoid fixed-width string truncation on injection
    primary = np.empty(len(sender_ids), dtype=object)
    for j, s in enumerate(sender_ids):
        primary[j] = sender_to_devices[s][0]
    return primary


# ---------------------------------------------------------------------------
# Fraud pattern injection
# ---------------------------------------------------------------------------

def _inject_night_high_value(
    idx: np.ndarray,
    timestamps: list,
    amounts: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Pattern 1: night hours + amount > 500,000."""
    for i in idx:
        ts = timestamps[i]
        # Force into night hour (22,23,0,1,2,3)
        night_hour = rng.choice([22, 23, 0, 1, 2, 3])
        timestamps[i] = ts.replace(hour=int(night_hour), minute=int(rng.integers(0, 60)))
        amounts[i] = round(float(rng.uniform(500_001, 5_000_000)), 2)


def _inject_corridor(
    idx: np.ndarray,
    sender_country: np.ndarray,
    receiver_country: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Pattern 2: EU/US sender → high-risk receiver country."""
    corridors = rng.choice(len(HIGH_RISK_CORRIDORS), size=len(idx))
    for j, i in enumerate(idx):
        sc, rc = HIGH_RISK_CORRIDORS[corridors[j]]
        sender_country[i] = sc
        receiver_country[i] = rc


def _inject_device_ip(
    idx: np.ndarray,
    sender_ids: np.ndarray,
    device_ids: np.ndarray,
    ip_country: np.ndarray,
    sender_country: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Pattern 3: new device AND ip_country != sender_country (ATO indicator)."""
    for i in idx:
        sid = sender_ids[i]
        # Assign a device that is NOT the sender's primary device
        device_ids[i] = f"dev_{sid}_new_{rng.integers(100, 999)}"
        # ip_country must differ from sender_country
        sc = sender_country[i]
        choices = [c for c in ALL_RECEIVER_COUNTRIES if c != sc]
        ip_country[i] = rng.choice(choices)


def _inject_structuring(
    idx: np.ndarray,
    sender_ids: np.ndarray,
    timestamps: list,
    amounts: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Pattern 4: structuring burst — 5-10 txns from SAME sender in 40-min window
    with amounts just below reporting thresholds (€9k-€9,999 or €99k-€99,999).

    Groups the provided indices into bursts of 5-10, assigns them the same
    sender, clusters their timestamps within a 40-minute window, and sets
    amounts to the structuring range.  This creates the velocity burst signal
    that makes the pattern detectable via velocity_1h.
    """
    n = len(idx)
    rng.shuffle(idx)
    i = 0
    while i < n:
        burst_size = int(rng.integers(5, 11))          # 5..10
        burst_idx  = idx[i: i + burst_size]
        i += burst_size

        # Pick one sender for the whole burst
        burst_sender = str(rng.integers(1, N_SENDERS + 1))

        # Use the earliest timestamp in this batch as anchor
        anchor_ts = min(timestamps[k] for k in burst_idx)

        # Spread txns uniformly within a 40-minute window from anchor
        offsets_s = rng.uniform(0, 40 * 60, size=len(burst_idx))
        offsets_s.sort()

        threshold = 99_000 if rng.random() < 0.5 else 9_000   # which range to use
        for rank, k in enumerate(burst_idx):
            sender_ids[k] = burst_sender
            timestamps[k] = anchor_ts + timedelta(seconds=float(offsets_s[rank]))
            if threshold == 9_000:
                amounts[k] = round(float(rng.uniform(9_000, 9_999)), 2)
            else:
                amounts[k] = round(float(rng.uniform(99_000, 99_999)), 2)


def _inject_mule_accounts(
    mule_receivers: list[str],
    sender_ids: np.ndarray,
    receiver_ids: np.ndarray,
    timestamps: list,
    fraud_mask: np.ndarray,
    rng: np.random.Generator,
    n_mule_fraud: int,
) -> None:
    """
    Pattern 5: receiver collects from 20-25 distinct senders within a 2-hour
    burst window (satisfies spec's "8+ distinct senders in 24h" threshold and
    makes receiver_fan_in_24h a strong discriminator against legitimate
    power-law receivers whose natural per-day fan_in is < 10).

    Using 20-25 senders instead of 8-12 ensures that the MAJORITY of rows in
    each cluster have fan_in >= 8 by the time they're processed, not just the
    last 2-3 rows.  The 2-hour window creates a tight burst that looks nothing
    like organic traffic patterns.
    """
    non_fraud_idx = np.where(~fraud_mask)[0]
    rng.shuffle(non_fraud_idx)
    cursor = 0

    for mule_rcv in mule_receivers:
        # 20–25 distinct senders per mule cluster — all within a 2-hour window
        n_senders_in = int(rng.integers(20, 26))
        senders_in = rng.choice(np.arange(1, N_SENDERS + 1), size=n_senders_in, replace=False).astype(str)

        # Anchor to the timestamp of the first picked row
        anchor_idx = non_fraud_idx[cursor] if cursor < len(non_fraud_idx) else 0
        anchor_ts = timestamps[anchor_idx]
        window_start = anchor_ts.replace(minute=0, second=0, microsecond=0)

        for k in range(n_senders_in):
            if cursor >= len(non_fraud_idx):
                break
            pick = non_fraud_idx[cursor]
            cursor += 1
            receiver_ids[pick] = mule_rcv
            sender_ids[pick] = senders_in[k % len(senders_in)]
            # Compress all senders into a 2-hour window (makes burst detectable)
            offset_s = float(rng.uniform(0, 2 * 3600))
            timestamps[pick] = window_start + timedelta(seconds=offset_s)
            fraud_mask[pick] = True

        if cursor >= n_mule_fraud:
            break


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic SWIFT/SEPA payment dataset with injected fraud patterns.
    Returns a sorted DataFrame — does NOT write to disk.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Base population
    # ------------------------------------------------------------------
    sender_ids = _make_sender_ids(rng, n_rows)
    receiver_ids = _make_receiver_ids(rng, n_rows)
    txn_ids = np.array([f"txn_{i:07d}" for i in range(n_rows)])

    timestamps_idx = _make_timestamps(rng, n_rows)
    # Work with a mutable list for fraud injection that modifies individual timestamps
    timestamps = list(timestamps_idx)

    amounts = _make_amounts(rng, n_rows)

    currencies = rng.choice(CURRENCIES, size=n_rows, p=CURRENCY_PROBS)

    sender_country = rng.choice(SENDER_COUNTRIES, size=n_rows, p=SENDER_COUNTRY_PROBS).astype(object)
    receiver_country = rng.choice(
        ALL_RECEIVER_COUNTRIES, size=n_rows, p=RECEIVER_COUNTRY_PROBS
    ).astype(object)

    device_ids = _make_device_ids(rng, sender_ids)
    # ip_country normally equals sender_country; ~3% natural mismatch (VPN/travel)
    ip_country = sender_country.copy().astype(object)
    natural_mismatch = rng.random(n_rows) < 0.03
    if natural_mismatch.any():
        replacements = rng.choice(SENDER_COUNTRIES, size=natural_mismatch.sum())
        for k, idx_k in enumerate(np.where(natural_mismatch)[0]):
            ip_country[idx_k] = replacements[k]

    channel = rng.choice(CHANNELS, size=n_rows, p=CHANNEL_PROBS)
    mcc = rng.choice(MCC_LIST, size=n_rows, p=MCC_PROBS)
    is_cross_border = (sender_country != receiver_country).astype(int)

    # ------------------------------------------------------------------
    # Fraud injection
    # ------------------------------------------------------------------
    fraud_mask = np.zeros(n_rows, dtype=bool)
    n_fraud_target = int(n_rows * FRAUD_RATE)

    n_night     = int(n_fraud_target * PATTERN_FRACS["night_high"])
    n_corridor  = int(n_fraud_target * PATTERN_FRACS["corridor"])
    n_device_ip = int(n_fraud_target * PATTERN_FRACS["device_ip"])
    n_struct    = int(n_fraud_target * PATTERN_FRACS["structuring"])
    n_mule      = int(n_fraud_target * PATTERN_FRACS["mule"])

    def _pick_fresh(k: int) -> np.ndarray:
        pool = np.where(~fraud_mask)[0]
        k = min(k, len(pool))
        return rng.choice(pool, size=k, replace=False)

    # Pattern 1 — Night high-value
    idx_night = _pick_fresh(n_night)
    _inject_night_high_value(idx_night, timestamps, amounts, rng)
    fraud_mask[idx_night] = True

    # Pattern 2 — High-risk corridor
    idx_corridor = _pick_fresh(n_corridor)
    _inject_corridor(idx_corridor, sender_country, receiver_country, rng)
    fraud_mask[idx_corridor] = True

    # Pattern 3 — Device + IP change
    idx_device = _pick_fresh(n_device_ip)
    _inject_device_ip(idx_device, sender_ids, device_ids, ip_country, sender_country, rng)
    fraud_mask[idx_device] = True

    # Pattern 4 — Structuring burst (velocity burst required)
    idx_struct = _pick_fresh(n_struct)
    _inject_structuring(idx_struct, sender_ids, timestamps, amounts, rng)
    fraud_mask[idx_struct] = True

    # Pattern 5 — Mule accounts (20-25 senders per cluster → ~10 clusters for n_mule≈250)
    n_mule_receivers = max(1, n_mule // 25)
    mule_receiver_ids = [f"mule_{i:04d}" for i in range(n_mule_receivers)]
    _inject_mule_accounts(
        mule_receivers=mule_receiver_ids,
        sender_ids=sender_ids,
        receiver_ids=receiver_ids,
        timestamps=timestamps,
        fraud_mask=fraud_mask,
        rng=rng,
        n_mule_fraud=n_mule,
    )

    # ------------------------------------------------------------------
    # Recompute derived fields after injection
    # ------------------------------------------------------------------
    is_cross_border = (sender_country != receiver_country).astype(int)
    label_fraud = fraud_mask.astype(int)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        "txn_id":           txn_ids,
        "timestamp":        pd.DatetimeIndex(timestamps),
        "amount":           amounts,
        "currency":         currencies,
        "sender_id":        sender_ids,
        "receiver_id":      receiver_ids,
        "sender_country":   sender_country,
        "receiver_country": receiver_country,
        "device_id":        device_ids,
        "ip_country":       ip_country,
        "channel":          channel,
        "mcc":              mcc,
        "is_cross_border":  is_cross_border,
        "label_fraud":      label_fraud,
    })

    # Sort by timestamp (required by spec)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Post-generation assertions
    # ------------------------------------------------------------------
    fraud_rate = df["label_fraud"].mean()
    assert 0.02 <= fraud_rate <= 0.03, (
        f"Fraud rate {fraud_rate:.4f} outside [0.02, 0.03]"
    )

    # Verify all 5 patterns are present
    fraud_df = df[df["label_fraud"] == 1].copy()
    fraud_df["hour"] = fraud_df["timestamp"].dt.hour

    night_mask = fraud_df["hour"].isin(NIGHT_HOURS) & (fraud_df["amount"] > 500_000)
    assert night_mask.sum() > 0, "Pattern 1 (night high-value) not found in fraud rows"

    corridor_mask = fraud_df["receiver_country"].isin(["KE", "NG", "PK", "VN", "CM"])
    assert corridor_mask.sum() > 0, "Pattern 2 (high-risk corridor) not found in fraud rows"

    # Pattern 3: rows where device changed (new device format = dev_<sid>_new_*)
    device_changed_mask = fraud_df["device_id"].str.contains("_new_", na=False)
    assert device_changed_mask.sum() > 0, "Pattern 3 (device+IP change) not found"

    # Pattern 4: structuring amounts (9k-10k or 99k-100k)
    struct_mask = (
        ((fraud_df["amount"] >= 9_000) & (fraud_df["amount"] <= 9_999)) |
        ((fraud_df["amount"] >= 99_000) & (fraud_df["amount"] <= 99_999))
    )
    assert struct_mask.sum() > 0, "Pattern 4 (structuring) not found in fraud rows"

    # Pattern 5: mule receiver IDs
    mule_mask = fraud_df["receiver_id"].str.startswith("mule_", na=False)
    assert mule_mask.sum() > 0, "Pattern 5 (mule account) not found in fraud rows"

    assert df["timestamp"].is_monotonic_increasing, "Timestamps not monotonically increasing"
    assert df.isnull().sum().sum() == 0, "DataFrame contains null values"

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SWIFT/SEPA payments with injected fraud patterns."
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="data/payments.parquet")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} rows (seed={args.seed}) ...")
    df = generate(n_rows=args.rows, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    fraud_rate = df["label_fraud"].mean() * 100
    n_fraud = int(df["label_fraud"].sum())
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Fraud rate: {fraud_rate:.2f}%  ({n_fraud:,} fraud rows)")
    print(f"Columns: {list(df.columns)}")

    # Pattern breakdown
    fraud_df = df[df["label_fraud"] == 1].copy()
    fraud_df["hour"] = fraud_df["timestamp"].dt.hour
    p1 = (fraud_df["hour"].isin(NIGHT_HOURS) & (fraud_df["amount"] > 500_000)).sum()
    p2 = fraud_df["receiver_country"].isin(["KE", "NG", "PK", "VN", "CM"]).sum()
    p3 = fraud_df["device_id"].str.contains("_new_", na=False).sum()
    p4 = (((fraud_df["amount"] >= 9_000) & (fraud_df["amount"] <= 9_999)) |
          ((fraud_df["amount"] >= 99_000) & (fraud_df["amount"] <= 99_999))).sum()
    p5 = fraud_df["receiver_id"].str.startswith("mule_", na=False).sum()
    print(f"\nPattern breakdown (of {n_fraud} fraud rows):")
    print(f"  P1 Night+High-Value:  {p1:4d}  ({p1/n_fraud*100:.1f}%)")
    print(f"  P2 High-Risk Corridor:{p2:4d}  ({p2/n_fraud*100:.1f}%)")
    print(f"  P3 Device+IP Change:  {p3:4d}  ({p3/n_fraud*100:.1f}%)")
    print(f"  P4 Structuring:       {p4:4d}  ({p4/n_fraud*100:.1f}%)")
    print(f"  P5 Mule Account:      {p5:4d}  ({p5/n_fraud*100:.1f}%)")


if __name__ == "__main__":
    main()
