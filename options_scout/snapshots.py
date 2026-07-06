"""Persist the raw option chain captured during each scan.

Pure data hygiene — this records exactly what the scanner already fetched so a
past morning can be replayed later with REAL entry quotes (not a model guess).
It does not influence any pick. One CSV per (date, symbol, mode, expiry):

    data/chains/2026-06-16/SPY_eod_2026-06-16.csv

Each row is one contract (call or put) with the quote as captured, plus the
spot price and capture timestamp so the moment is fully reconstructable.
"""

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger("options_scout")

# Columns kept from the yfinance chain (others are dropped to keep files small).
_KEEP = ["contractSymbol", "strike", "bid", "ask", "lastPrice",
         "volume", "openInterest", "impliedVolatility"]


def _prep(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    cols = [c for c in _KEEP if c in df.columns]
    out = df[cols].copy()
    out.insert(0, "type", kind)
    return out


def save_chain(cfg: dict, symbol: str, mode: str, expiry: date, spot: float,
               calls: pd.DataFrame, puts: pd.DataFrame, now_et: datetime) -> Path | None:
    """Write one snapshot file. Best-effort: never raises into the scan."""
    try:
        day_dir = Path(cfg["data_dir"]) / "chains" / f"{now_et:%Y-%m-%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.concat([_prep(calls, "call"), _prep(puts, "put")],
                          ignore_index=True)
        frame.insert(0, "snapshot_ts", now_et.isoformat(timespec="seconds"))
        frame.insert(1, "symbol", symbol)
        frame.insert(2, "mode", mode)
        frame.insert(3, "expiry", expiry.isoformat())
        frame.insert(4, "spot", round(spot, 4))
        path = day_dir / f"{symbol}_{mode}_{expiry.isoformat()}.csv"
        frame.to_csv(path, index=False)
        log.info("Saved chain snapshot %s (%d contracts)", path.name, len(frame))
        return path
    except Exception as e:  # noqa: BLE001 - snapshotting must never break a scan
        log.warning("snapshot save failed for %s %s: %s", symbol, expiry, e)
        return None
