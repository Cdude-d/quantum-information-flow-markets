"""Market data access via yfinance, with retries and light caching.

Note: Yahoo options quotes are delayed ~15 minutes (OPRA). A scan at 9:45 ET
sees option quotes from roughly 9:30, i.e. the first post-open prints.
"""

import logging
import time
from datetime import date, datetime

import pandas as pd
import yfinance as yf

log = logging.getLogger("options_scout")

_TICKER_CACHE: dict[str, yf.Ticker] = {}


def _retry(fn, tries: int = 3, delay: float = 2.0, label: str = ""):
    last_err = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network layer, retry anything
            last_err = e
            if attempt < tries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"data fetch failed for {label}: {last_err}") from last_err


def get_ticker(symbol: str) -> yf.Ticker:
    if symbol not in _TICKER_CACHE:
        _TICKER_CACHE[symbol] = yf.Ticker(symbol)
    return _TICKER_CACHE[symbol]


def get_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    def fetch():
        df = get_ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("empty history")
        return df
    return _retry(fetch, label=f"{symbol} history")


def get_spot(symbol: str, hist: pd.DataFrame | None = None) -> float:
    t = get_ticker(symbol)
    try:
        px = t.fast_info["last_price"]
        if px and px > 0:
            return float(px)
    except Exception:  # noqa: BLE001
        pass
    if hist is not None and not hist.empty:
        return float(hist["Close"].iloc[-1])
    return float(get_history(symbol, "5d")["Close"].iloc[-1])


def list_expiries(symbol: str) -> list[date]:
    def fetch():
        opts = get_ticker(symbol).options
        if not opts:
            raise ValueError("no listed expiries")
        return [datetime.strptime(e, "%Y-%m-%d").date() for e in opts]
    return _retry(fetch, label=f"{symbol} expiries")


def get_chain(symbol: str, expiry: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (calls, puts) DataFrames for the expiry."""
    def fetch():
        ch = get_ticker(symbol).option_chain(expiry.strftime("%Y-%m-%d"))
        if ch.calls.empty and ch.puts.empty:
            raise ValueError("empty chain")
        return ch.calls.copy(), ch.puts.copy()
    return _retry(fetch, label=f"{symbol} {expiry} chain")


def get_vix() -> dict:
    """VIX level and 1-day change; empty dict on failure (non-critical)."""
    try:
        h = get_history("^VIX", "10d")
        level = float(h["Close"].iloc[-1])
        prev = float(h["Close"].iloc[-2]) if len(h) > 1 else level
        return {"level": level, "change_pct": (level / prev - 1.0) * 100.0}
    except Exception as e:  # noqa: BLE001
        log.warning("VIX fetch failed: %s", e)
        return {}


def next_earnings_date(symbol: str) -> date | None:
    """Best-effort next earnings date; None if unavailable."""
    t = get_ticker(symbol)
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            dates = [d for d in dates if isinstance(d, date)]
            future = [d for d in dates if d >= date.today()]
            if future:
                return min(future)
    except Exception:  # noqa: BLE001
        pass
    try:
        df = t.get_earnings_dates(limit=8)
        if df is not None and not df.empty:
            future = [ts.date() for ts in df.index if ts.date() >= date.today()]
            if future:
                return min(future)
    except Exception:  # noqa: BLE001
        pass
    return None
