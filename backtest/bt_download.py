"""Download end-of-week entry-snapshot option chains from Databento.

Spends credits. For each of the last N completed weeks, pulls a one-minute
consolidated BBO (cbbo-1m) snapshot of all 13 underlyings' option chains at the
~10:00 ET entry. Idempotent: existing files are skipped, so a re-run resumes.

Files: backtest/data/chains/<entry>_<expiry>.dbn.zst

Usage:
  python backtest/bt_download.py --limit 1     # just the most recent week (test)
  python backtest/bt_download.py               # all N_WEEKS (skips existing)
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402
from options_scout.signals import is_trading_day  # noqa: E402

import databento as db  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1m"
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA",
            "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX"]
ENTRY_ET = dtime(10, 0)
WINDOW_SECONDS = 60
N_WEEKS = 26
OUT = Path(__file__).resolve().parent / "data" / "chains"


def gen_weeks(n: int, today: date) -> list[tuple[date, date]]:
    """(entry, expiry) for the last n completed weeks. Entry = week's first
    trading day; expiry = week's last trading day (Fri, or earlier if holiday)."""
    pairs = []
    monday = today - timedelta(days=today.weekday())
    w = 1
    while len(pairs) < n and w < n + 12:
        m = monday - timedelta(weeks=w)
        fri = m + timedelta(days=4)
        if fri < today:
            entry = m
            while not is_trading_day(entry):
                entry += timedelta(days=1)
            exp = fri
            while not is_trading_day(exp):
                exp -= timedelta(days=1)
            if entry < exp:
                pairs.append((entry, exp))
        w += 1
    return sorted(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="download only the most recent N weeks")
    ap.add_argument("--weeks", type=int, default=N_WEEKS)
    ap.add_argument("--end", default=None,
                    help="reference date (YYYY-MM-DD); weeks counted back from here")
    ap.add_argument("--subdir", default="chains",
                    help="output folder under backtest/data/ (use a new name for OOS)")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated universe override (e.g. RRC,JEF,EXEL)")
    ap.add_argument("--estimate", action="store_true",
                    help="print Databento cost and exit (spends nothing)")
    args = ap.parse_args()
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    out_dir = Path(__file__).resolve().parent / "data" / args.subdir
    universe = args.symbols.split(",") if args.symbols else UNIVERSE

    load_env()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY not set.")
        return
    client = db.Historical(key)
    out_dir.mkdir(parents=True, exist_ok=True)

    weeks = gen_weeks(args.weeks, end_date)
    if args.limit:
        weeks = weeks[-args.limit:]
    syms = [f"{s}.OPT" for s in universe]
    print(f"{len(weeks)} week(s): {weeks[0][0]} .. {weeks[-1][0]} | {len(universe)} symbols",
          flush=True)

    if args.estimate:
        total = 0.0
        for entry, exp in weeks:
            start = datetime.combine(entry, ENTRY_ET, tzinfo=ET).astimezone(UTC)
            end = start + timedelta(seconds=WINDOW_SECONDS)
            try:
                total += float(client.metadata.get_cost(
                    dataset=DATASET, schema=SCHEMA, symbols=syms, stype_in="parent",
                    start=start, end=end))
            except Exception as e:  # noqa: BLE001
                print(f"  estimate err {entry}: {e}", flush=True)
        print(f"\nESTIMATED COST: ${total:,.2f} for {len(weeks)} weeks x {len(universe)} "
              f"symbols. (No credits spent.)", flush=True)
        return

    total_mb = 0.0
    failures = []
    for entry, exp in weeks:
        path = out_dir / f"{entry.isoformat()}_{exp.isoformat()}.dbn.zst"
        if path.exists():
            print(f"skip  {entry} (exists)", flush=True)
            continue
        start = datetime.combine(entry, ENTRY_ET, tzinfo=ET).astimezone(UTC)
        end = start + timedelta(seconds=WINDOW_SECONDS)
        print(f"pull  {entry} -> exp {exp} ...", end="", flush=True)
        ok = False
        for attempt in range(4):
            try:
                client.timeseries.get_range(
                    dataset=DATASET, schema=SCHEMA, symbols=syms, stype_in="parent",
                    start=start, end=end, path=str(path))
                ok = True
                break
            except Exception as e:  # noqa: BLE001 - retry transient gateway errors
                path.unlink(missing_ok=True)  # don't leave a partial file
                if attempt < 3:
                    print(f" retry({type(e).__name__})", end="", flush=True)
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f" FAILED: {e}", flush=True)
        if not ok:
            failures.append(entry)
            continue
        mb = path.stat().st_size / 1e6
        total_mb += mb
        print(f" {mb:.1f} MB", flush=True)
    print(f"\nDone. {total_mb:.1f} MB this run.", flush=True)
    if failures:
        print(f"Failed weeks (re-run to retry): {[d.isoformat() for d in failures]}", flush=True)


if __name__ == "__main__":
    main()
