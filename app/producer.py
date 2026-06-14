"""
app/producer.py
Reads data/payments.parquet and yields one transaction dict every 0.5 s
from a random start position, cycling indefinitely.

Usage (standalone):
    python app/producer.py

Usage (importable):
    from app.producer import transaction_stream
    for txn in transaction_stream():
        ...
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Generator

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARQUET_FILE = Path("data/payments.parquet")
EMIT_INTERVAL = 0.5  # seconds between transactions

# Columns expected by POST /score (excludes label_fraud)
_API_COLS = [
    "txn_id", "timestamp", "amount", "currency",
    "sender_id", "receiver_id", "sender_country", "receiver_country",
    "device_id", "ip_country", "channel", "mcc", "is_cross_border",
]


def _load_records() -> list[dict]:
    """Load parquet and return list of API-ready dicts."""
    df = pd.read_parquet(PARQUET_FILE, columns=_API_COLS + ["label_fraud"])

    # Convert is_cross_border int -> bool (API expects bool)
    df["is_cross_border"] = df["is_cross_border"].astype(bool)

    # Convert timestamp to ISO string (JSON-serialisable)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Drop label so it is never sent to the API
    df = df.drop(columns=["label_fraud"])

    return df.to_dict("records")


def transaction_stream(
    interval: float = EMIT_INTERVAL,
    start_index: int | None = None,
) -> Generator[dict, None, None]:
    """
    Yield transaction dicts from a random position in the dataset,
    cycling indefinitely.

    Args:
        interval:    Seconds to sleep between yields. Default 0.5 s.
        start_index: Starting row index. Randomised if None.
    """
    records = _load_records()
    n = len(records)

    idx = start_index if start_index is not None else random.randint(0, n - 1)
    print(f"[producer] Loaded {n:,} transactions. Starting at index {idx}.")

    while True:
        yield records[idx % n]
        idx += 1
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Standalone entry point — prints transactions to stdout
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[producer] Streaming transactions (Ctrl+C to stop) ...\n")
    for i, txn in enumerate(transaction_stream(), start=1):
        print(
            f"[{i:>6}] {txn['txn_id']}  "
            f"amount={txn['amount']:>12,.2f}  "
            f"corridor={txn['sender_country']}->{txn['receiver_country']}"
        )
