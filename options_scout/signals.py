"""Technical and volatility signals, plus the market calendar (events/holidays)."""

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# --- Calendar -------------------------------------------------------------

NYSE_HOLIDAYS = {
    date(2024, 1, 1): "New Year's Day",
    date(2024, 1, 15): "MLK Day",
    date(2024, 2, 19): "Presidents' Day",
    date(2024, 3, 29): "Good Friday",
    date(2024, 5, 27): "Memorial Day",
    date(2024, 6, 19): "Juneteenth",
    date(2024, 7, 4): "Independence Day",
    date(2024, 9, 2): "Labor Day",
    date(2024, 11, 28): "Thanksgiving",
    date(2024, 12, 25): "Christmas",
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "MLK Day",
    date(2025, 2, 17): "Presidents' Day",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 6, 19): "Juneteenth",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving",
    date(2025, 12, 25): "Christmas",
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "MLK Day",
    date(2026, 2, 16): "Presidents' Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (observed)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving",
    date(2026, 12, 25): "Christmas",
    date(2027, 1, 1): "New Year's Day",
    date(2027, 1, 18): "MLK Day",
}

# FOMC rate-decision days (2 p.m. ET statement), 2026 published schedule.
FOMC_DECISIONS = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    nxt = d
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def _nfp_dates(year: int) -> list[date]:
    """Nonfarm payrolls: first Friday of each month (8:30 a.m. ET)."""
    out = []
    for month in range(1, 13):
        d = date(year, month, 1)
        d += timedelta(days=(4 - d.weekday()) % 7)
        out.append(d)
    return out


def events_between(start: date, end: date, custom: list[str] | None = None) -> list[tuple[date, str]]:
    """Scheduled macro events in [start, end], inclusive."""
    events = [(d, "FOMC rate decision (2pm ET)") for d in FOMC_DECISIONS]
    for y in {start.year, end.year}:
        events += [(d, "Nonfarm payrolls (8:30am ET)") for d in _nfp_dates(y)]
    for item in custom or []:
        try:
            ds, label = item.split(":", 1)
            events.append((date.fromisoformat(ds.strip()), label.strip()))
        except ValueError:
            continue
    hits = [(d, lbl) for d, lbl in events if start <= d <= end]
    return sorted(set(hits))


# --- Technicals -----------------------------------------------------------

def rsi(closes: pd.Series, period: int = 2) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    val = out.iloc[-1]
    return 50.0 if pd.isna(val) else float(val)


def ewma_vol(closes: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility, annualized."""
    rets = np.log(closes / closes.shift(1)).dropna().to_numpy()
    if len(rets) < 30:
        return float(np.std(rets) * math.sqrt(252)) if len(rets) > 2 else 0.2
    var = rets[:30].var()
    for r in rets[30:]:
        var = lam * var + (1 - lam) * r * r
    return float(math.sqrt(var * 252))


def gap_fill_stats(hist: pd.DataFrame, min_gap: float = 0.002) -> dict:
    """How often opening gaps beyond `min_gap` filled (touched prior close) same day."""
    df = hist.copy()
    df["prev_close"] = df["Close"].shift(1)
    df["gap"] = df["Open"] / df["prev_close"] - 1.0
    df = df.dropna()
    up = df[df["gap"] >= min_gap]
    down = df[df["gap"] <= -min_gap]
    up_fill = (up["Low"] <= up["prev_close"]).mean() if len(up) else float("nan")
    down_fill = (down["High"] >= down["prev_close"]).mean() if len(down) else float("nan")
    return {
        "min_gap": min_gap,
        "n_up": int(len(up)), "up_fill_rate": float(up_fill) if len(up) else None,
        "up_o2c_mean": float((up["Close"] / up["Open"] - 1).mean()) if len(up) else None,
        "n_down": int(len(down)), "down_fill_rate": float(down_fill) if len(down) else None,
        "down_o2c_mean": float((down["Close"] / down["Open"] - 1).mean()) if len(down) else None,
    }


def compute_signals(hist: pd.DataFrame, spot: float, session_date: date) -> dict:
    closes = hist["Close"]
    last_bar_date = hist.index[-1].date()
    prev_close = float(closes.iloc[-2]) if last_bar_date == session_date and len(closes) > 1 else float(closes.iloc[-1])

    gap_pct = since_open_pct = None
    if last_bar_date == session_date:
        today_open = float(hist["Open"].iloc[-1])
        gap_pct = (today_open / prev_close - 1.0) * 100.0
        since_open_pct = (spot / today_open - 1.0) * 100.0
    else:
        gap_pct = (spot / prev_close - 1.0) * 100.0  # pre-market / overnight move

    ema20 = float(closes.ewm(span=20).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50).mean().iloc[-1])
    ema20_slope = float(closes.ewm(span=20).mean().diff(5).iloc[-1])
    if spot > ema20 > ema50 and ema20_slope > 0:
        trend = "up"
    elif spot < ema20 < ema50 and ema20_slope < 0:
        trend = "down"
    else:
        trend = "mixed"

    ret_5d = float(spot / closes.iloc[-6] - 1.0) if len(closes) > 6 else 0.0
    daily_range = ((hist["High"] - hist["Low"]) / hist["Close"]).tail(20)

    return {
        "spot": spot,
        "prev_close": prev_close,
        "gap_pct": gap_pct,
        "since_open_pct": since_open_pct,
        "trend": trend,
        "ema20": ema20,
        "ema50": ema50,
        "rsi2": rsi(closes, 2),
        "rsi14": rsi(closes, 14),
        "ret_5d_pct": ret_5d * 100.0,
        "realized_vol": ewma_vol(closes),
        "avg_daily_range_pct": float(daily_range.mean() * 100.0),
        "gap_stats": gap_fill_stats(hist),
    }


def bias_score(sig: dict, weights: dict) -> dict:
    """Directional bias in [-1, +1] from trend, mean-reversion, momentum."""
    trend_comp = {"up": 1.0, "down": -1.0, "mixed": 0.0}[sig["trend"]]
    meanrev_comp = float(np.clip((50.0 - sig["rsi2"]) / 40.0, -1.0, 1.0))
    momentum_comp = float(np.clip(sig["ret_5d_pct"] / 3.0, -1.0, 1.0))
    total = (weights["trend"] * trend_comp
             + weights["meanrev"] * meanrev_comp
             + weights["momentum"] * momentum_comp)
    return {
        "trend": trend_comp,
        "meanrev": meanrev_comp,
        "momentum": momentum_comp,
        "total": float(np.clip(total, -1.0, 1.0)),
    }
