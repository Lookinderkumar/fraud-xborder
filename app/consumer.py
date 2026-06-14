"""
app/consumer.py
Reads from producer, calls POST /score on http://localhost:8000,
appends ALL scored transactions to data/alerts.jsonl, and prints a
summary line for transactions scoring above 0.5.

Usage:
    python app/consumer.py

Requires the API to be running:
    uvicorn app.model_api:app --reload

Environment (.env):
    API_KEY=changeme        # X-API-Key header value
    API_URL=http://localhost:8000  # optional override
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from app.producer import transaction_stream

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY  = os.getenv("API_KEY", "changeme")
API_URL  = os.getenv("API_URL", "http://localhost:8000")
SCORE_URL = f"{API_URL}/score"
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

ALERTS_FILE   = Path("data/alerts.jsonl")
STATS_EVERY   = 50          # print latency stats every N transactions
HIGH_SCORE_THRESHOLD = 0.5  # print to console when score exceeds this


def _ensure_data_dir() -> None:
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _fmt_reasons(reasons: list[str]) -> str:
    return " | ".join(reasons) if reasons else "(none)"


def run(interval: float = 0.5) -> None:
    """
    Main consumer loop. Streams transactions from producer, scores each
    via the API, and writes all results to data/alerts.jsonl.

    Args:
        interval: Seconds between transactions (passed to producer).
    """
    _ensure_data_dir()

    latencies: deque[float] = deque(maxlen=1_000)
    total = 0
    errors = 0

    print(f"[consumer] Connecting to {SCORE_URL}")
    print(f"[consumer] Writing all scored transactions to {ALERTS_FILE}")
    print(f"[consumer] Printing alerts with score > {HIGH_SCORE_THRESHOLD}")
    print("[consumer] Press Ctrl+C to stop.\n")

    # Quick connectivity check before starting stream
    try:
        resp = requests.get(f"{API_URL}/health", headers={"X-API-Key": API_KEY}, timeout=5)
        resp.raise_for_status()
        health = resp.json()
        print(
            f"[consumer] API healthy — version={health.get('version')}  "
            f"threshold={health.get('threshold')}\n"
        )
    except requests.exceptions.ConnectionError:
        print(
            f"[consumer] ERROR: Cannot reach {API_URL}. "
            "Start the API first:\n"
            "  uvicorn app.model_api:app --reload\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"[consumer] WARNING: Health check failed ({exc}). Continuing anyway.\n")

    with open(ALERTS_FILE, "a", encoding="utf-8") as alert_fh:
        for txn in transaction_stream(interval=interval):
            total += 1

            # --- POST /score ---
            t0 = time.perf_counter()
            try:
                resp = requests.post(SCORE_URL, json=txn, headers=HEADERS, timeout=10)
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                errors += 1
                print(f"[consumer] Request error on {txn['txn_id']}: {exc}", file=sys.stderr)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1_000
            latencies.append(elapsed_ms)

            body = resp.json()
            score     = body["score"]
            is_fraud  = body["is_fraud"]
            reasons   = body.get("reasons", [])

            # --- Append ALL scored transactions to alerts.jsonl ---
            record = {
                "txn_id":        txn["txn_id"],
                "timestamp":     txn["timestamp"],
                "score":         score,
                "is_fraud":      is_fraud,
                "latency_ms":    elapsed_ms,
                "reasons":       reasons,
                "model_version": body.get("model_version", ""),
            }
            alert_fh.write(json.dumps(record) + "\n")
            alert_fh.flush()

            # --- Console output: only high-scoring transactions ---
            if score > HIGH_SCORE_THRESHOLD:
                flag = "FRAUD" if is_fraud else "HIGH "
                print(
                    f"[{flag}] #{total:>6}  {txn['txn_id']:<22}"
                    f"  score={score:.4f}"
                    f"  {txn['sender_country']}->{txn['receiver_country']}"
                    f"  EUR {txn['amount']:>12,.0f}"
                    f"  {_fmt_reasons(reasons)}"
                )

            # --- Rolling latency stats every STATS_EVERY transactions ---
            if total % STATS_EVERY == 0:
                lats = list(latencies)
                p50 = np.percentile(lats, 50)
                p95 = np.percentile(lats, 95)
                fraud_pct = (
                    sum(1 for _ in [record] if record["is_fraud"]) / min(total, len(lats)) * 100
                )
                print(
                    f"--- stats @ {total:,} txns  "
                    f"p50={p50:.0f}ms  p95={p95:.0f}ms  "
                    f"errors={errors}  "
                    f"file={ALERTS_FILE} ---"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[consumer] Stopped.")
