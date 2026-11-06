"""Download 1-minute bars for SPY/QQQ/IWM from Databento XNAS.ITCH (Jun 2018 - Oct 2026).

Spends credits (~$2.82 total, estimated). One file per calendar year, idempotent and
retried, saved under backtest/data/intraday/ (git-ignored: licensed data).

Run:  python research/intraday_download.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402

import databento as db  # noqa: E402

DATASET, SCHEMA = "XNAS.ITCH", "ohlcv-1m"
SYMBOLS = ["SPY", "QQQ", "IWM"]
START, END = "2018-06-01", "2026-10-31"
OUT = Path(__file__).resolve().parent.parent / "backtest" / "data" / "intraday"


def main():
    load_env()
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    OUT.mkdir(parents=True, exist_ok=True)
    for year in range(int(START[:4]), int(END[:4]) + 1):
        s = max(START, f"{year}-01-01")
        e = min(END, f"{year}-12-31")
        path = OUT / f"xnas_ohlcv1m_{year}.dbn.zst"
        if path.exists():
            print(f"skip {year} (exists)", flush=True)
            continue
        print(f"pull {year} {s}..{e} ...", end="", flush=True)
        for attempt in range(4):
            try:
                client.timeseries.get_range(dataset=DATASET, schema=SCHEMA,
                                            symbols=SYMBOLS, start=s, end=e,
                                            path=str(path))
                print(f" {path.stat().st_size / 1e6:.1f} MB", flush=True)
                break
            except Exception as ex:  # noqa: BLE001 - retry transient gateway errors
                path.unlink(missing_ok=True)
                if attempt == 3:
                    print(f" FAILED: {str(ex).splitlines()[0]}", flush=True)
                else:
                    print(f" retry({type(ex).__name__})", end="", flush=True)
                    time.sleep(5 * (attempt + 1))


if __name__ == "__main__":
    main()
