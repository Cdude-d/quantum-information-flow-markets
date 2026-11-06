"""Download 0DTE SPY at-the-money call+put quotes (9:49-10:51 ET) for every held-out
day (2023-01-03 .. 2026-10-30), for the range-forecast straddle backtest.

We know SPY's 9:50 price from the minute bars, so we request only the ATM strike's
call and put for the same-day expiry (raw OSI symbols) instead of whole chains.

  python research/odte_download.py --estimate   # free: verify symbols + price the data
  python research/odte_download.py              # download (spends credits), idempotent
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402

import databento as db  # noqa: E402

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "backtest" / "data" / "intraday" / "opening_features.csv"
OUT = ROOT / "backtest" / "data" / "odte"
DATASET, SCHEMA = "OPRA.PILLAR", "cbbo-1m"
SPLIT = "2023-01-01"
WORKERS = 16        # parallel requests; per-request latency, not data size, is the bottleneck


def osi(day, cp, strike):
    return f"SPY   {day:%y%m%d}{cp}{int(round(strike * 1000)):08d}"


def plan(start=SPLIT, end=None):
    f = pd.read_csv(FEAT, parse_dates=["date"])
    f = f[(f.symbol == "SPY") & (f.date >= start)].copy()
    if end:
        f = f[f.date <= end]
    f["spot950"] = f.open_930 * (1 + f.open_ret)
    f["strike"] = f.spot950.round()
    rows = []
    for _, r in f.iterrows():
        d = r.date.date()
        rows.append((d, r.strike, [osi(d, "C", r.strike), osi(d, "P", r.strike)]))
    return rows


def plan_from_bars(start=SPLIT, end=None):
    """Same plan, built straight from SPY minute bars (9:50 price -> ATM strike), so days
    without opening-window features (e.g. trading halts at the open) are still included."""
    from opening_screen import load_bars
    b = load_bars()
    b = b[b.symbol == "SPY"]
    rows = []
    for d, g in b.groupby("date"):
        if str(d) < start or (end and str(d) > end):
            continue
        pre = g[g.mins <= 19]
        if pre.empty:
            continue
        k = float(round(pre.close.iloc[-1]))
        rows.append((d, k, [osi(d, "C", k), osi(d, "P", k)]))
    return rows


def window(d):
    s = datetime.combine(d, dtime(9, 49), tzinfo=ET).astimezone(UTC)
    e = datetime.combine(d, dtime(10, 52), tzinfo=ET).astimezone(UTC)
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--start", default=SPLIT, help="first day (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last day (YYYY-MM-DD)")
    ap.add_argument("--from-bars", action="store_true",
                    help="build the day list from SPY minute bars instead of the feature table")
    args = ap.parse_args()
    load_env()
    c = db.Historical(os.environ["DATABENTO_API_KEY"])
    days = (plan_from_bars(args.start, args.end) if args.from_bars
            else plan(args.start, args.end))
    print(f"{len(days)} SPY days, {days[0][0]}..{days[-1][0]}", flush=True)

    if args.estimate:
        def resolves(item):
            d, _, syms = item
            res = c.symbology.resolve(dataset=DATASET, symbols=syms, stype_in="raw_symbol",
                                      stype_out="instrument_id", start_date=d,
                                      end_date=d + pd.Timedelta(days=1))
            return not res.get("not_found")
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            ok = list(pool.map(resolves, days))
        have = [x for x, good in zip(days, ok) if good]
        by_dow = pd.Series([pd.Timestamp(d).day_name()[:3] for d, _, _ in have]).value_counts()
        print(f"days with a same-day ATM straddle listed: {len(have)} of {len(days)}  "
              f"{dict(by_dow)}")
        if not have:
            return
        sample = have[:: max(1, len(have) // 15)]
        costs = []
        for d, k, syms in sample:
            s, e = window(d)
            costs.append(float(c.metadata.get_cost(dataset=DATASET, schema=SCHEMA, symbols=syms,
                                                   stype_in="raw_symbol", start=s, end=e)))
        avg = sum(costs) / len(costs)
        print(f"avg cost/day ${avg:.5f}  ->  ESTIMATED TOTAL ${avg * len(have):.2f} "
              f"(no credits spent)")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    todo = [x for x in days if not (OUT / f"{x[0]}.dbn.zst").exists()]
    print(f"{len(days) - len(todo)} already downloaded, {len(todo)} to fetch "
          f"({WORKERS} in parallel)", flush=True)

    def fetch(item):
        d, _, syms = item
        path, part = OUT / f"{d}.dbn.zst", OUT / f"{d}.dbn.zst.part"
        s, e = window(d)
        for attempt in range(4):
            try:
                c.timeseries.get_range(dataset=DATASET, schema=SCHEMA, symbols=syms,
                                       stype_in="raw_symbol", start=s, end=e, path=str(part))
                part.replace(path)          # only a complete file ever gets the real name
                return None
            except Exception as ex:  # noqa: BLE001
                part.unlink(missing_ok=True)
                msg = str(ex).splitlines()[0][:80]
                if attempt == 3 or "symbology" in msg:     # no contract that day: don't retry
                    return (d, msg)
                time.sleep(3 * (attempt + 1))

    fails, done = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for res in pool.map(fetch, todo):
            done += 1
            if res:
                fails.append(res)
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} fetched", flush=True)
    print(f"done. failures: {len(fails)}", flush=True)
    for d, msg in fails[:10]:
        print(f"  {d}: {msg}")


if __name__ == "__main__":
    main()
