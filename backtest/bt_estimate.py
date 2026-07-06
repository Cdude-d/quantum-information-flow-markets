"""Estimate the Databento cost of a historical backtest BEFORE downloading.

FREE — metadata/cost queries never spend credits. Prices each symbol's chain
snapshot ONCE (at a representative recent week) and multiplies by the number of
weeks, instead of querying every week separately. ~26 calls, not ~676.

Backtest design (end-of-week): one entry per week at ~10:00 ET on the week's
first trading day, targeting that week's Friday expiry, held to the Friday close
(settlement uses free yfinance closes, so it is not priced here).

Run:  python backtest/bt_estimate.py
"""
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402

import databento as db  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DATASET = "OPRA.PILLAR"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA",
            "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX"]
QUOTE_SCHEMA = "cbbo-1m"        # consolidated 1-min BBO (cmbp-1 504s; avoid)
ENTRY_ET = dtime(10, 0)
WINDOW_SECONDS = 60
N_WEEKS = 26                    # ~6 months
BUDGET = 30.0
# A representative recent week, used as the per-snapshot unit cost (chains are
# largest now, so this is a slight over-estimate -> conservative/safe).
SAMPLE_DAY = date(2026, 5, 4)


def window_utc(d: date):
    start = datetime.combine(d, ENTRY_ET, tzinfo=ET).astimezone(UTC)
    return start, start + timedelta(seconds=WINDOW_SECONDS)


def main():
    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY not set (.env not loaded).")
        return
    client = db.Historical(key)
    start, end = window_utc(SAMPLE_DAY)
    print(f"Pricing one {QUOTE_SCHEMA} snapshot + definition per symbol "
          f"(sample week {SAMPLE_DAY}), x{N_WEEKS} weeks.\n", flush=True)
    print(f"{'sym':6}{'quote $/snap':>14}{'def $/snap':>13}{'MB/snap':>10}", flush=True)
    print("-" * 43, flush=True)

    rows = []
    for sym in UNIVERSE:
        rec = {"sym": sym, "q": 0.0, "d": 0.0, "mb": 0.0, "err": None}
        for schema, fld in ((QUOTE_SCHEMA, "q"), ("definition", "d")):
            try:
                cost = float(client.metadata.get_cost(
                    dataset=DATASET, schema=schema, symbols=[f"{sym}.OPT"],
                    stype_in="parent", start=start, end=end))
                size = int(client.metadata.get_billable_size(
                    dataset=DATASET, schema=schema, symbols=[f"{sym}.OPT"],
                    stype_in="parent", start=start, end=end))
                rec[fld] = cost
                rec["mb"] += size / 1e6
            except Exception as e:  # noqa: BLE001
                rec["err"] = f"{type(e).__name__}"
        rows.append(rec)
        tag = f"  !{rec['err']}" if rec["err"] else ""
        print(f"{sym:6}{rec['q']:>14.4f}{rec['d']:>13.4f}{rec['mb']:>10.1f}{tag}",
              flush=True)

    q_week = sum(r["q"] for r in rows)
    d_week = sum(r["d"] for r in rows)
    mb_week = sum(r["mb"] for r in rows)
    total = (q_week + d_week) * N_WEEKS
    gb = mb_week * N_WEEKS / 1e3

    print("\n" + "=" * 50, flush=True)
    print(f"Per-week (all {len(UNIVERSE)} symbols): quotes ${q_week:.3f} + "
          f"defs ${d_week:.3f} = ${q_week + d_week:.3f}", flush=True)
    print(f"x {N_WEEKS} weeks  ->  TOTAL ${total:,.2f}   ({gb:.2f} GB)", flush=True)
    print("=" * 50, flush=True)
    if total <= BUDGET:
        print(f"WITHIN your ${BUDGET:.0f} budget ({total/BUDGET*100:.0f}% of it). "
              f"Safe to download.", flush=True)
    else:
        print(f"OVER by ${total-BUDGET:,.2f}. Trim scope and re-estimate.", flush=True)
    print("No credits spent (metadata only).", flush=True)


if __name__ == "__main__":
    main()
