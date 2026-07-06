"""Probe two documented, free-to-test anomalies — held to honest standards.

Pre-specified hypotheses (no data-mining):
  1. Overnight effect: equity ETF returns concentrate in the close->open session,
     with open->close (intraday) flat-to-negative.
  2. Time-series momentum (trend following): go long an asset when its trailing
     12-month return is positive, else flat; diversified across asset classes.

For each, reports the effect AND its stability across the first vs second half of
history (a poor-man's out-of-sample). Free yfinance daily data only.

Run:  python research/alpha_probes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

EQUITY = ["SPY", "QQQ", "IWM", "DIA"]
# Cross-asset basket for trend following (trend works best diversified).
TREND = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP"]
PERIOD = "12y"


def ann_stats(daily: pd.Series) -> dict:
    d = daily.dropna()
    if len(d) < 60:
        return {"ret": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    ann_ret = (1 + d).prod() ** (252 / len(d)) - 1
    vol = d.std() * np.sqrt(252)
    sharpe = (d.mean() * 252) / vol if vol > 0 else np.nan
    cum = (1 + d).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()
    return {"ret": ann_ret, "vol": vol, "sharpe": sharpe, "maxdd": maxdd}


def fetch(symbols):
    data = {}
    for s in symbols:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)
            if not h.empty and len(h) > 250:
                data[s] = h
        except Exception:  # noqa: BLE001
            pass
    return data


def probe_overnight(data):
    print("\n" + "=" * 68)
    print("PROBE 1 — Overnight (close->open) vs Intraday (open->close)")
    print("=" * 68)
    print(f"{'sym':5}{'period':>10}{'overnight':>20}{'intraday':>18}{'buy&hold':>14}")
    print(f"{'':5}{'':>10}{'ann%  Sharpe':>20}{'ann%  Sharpe':>18}{'ann%':>14}")
    for s in EQUITY:
        h = data.get(s)
        if h is None:
            continue
        on = h["Open"] / h["Close"].shift(1) - 1
        intr = h["Close"] / h["Open"] - 1
        bh = h["Close"].pct_change()
        half = len(h) // 2
        for tag, sl in (("full", slice(None)), ("1st-half", slice(0, half)),
                        ("2nd-half", slice(half, None))):
            o, i, b = ann_stats(on.iloc[sl]), ann_stats(intr.iloc[sl]), ann_stats(bh.iloc[sl])
            print(f"{s if tag=='full' else '':5}{tag:>10}"
                  f"{o['ret']*100:>11.1f}{o['sharpe']:>7.2f}"
                  f"{i['ret']*100:>11.1f}{i['sharpe']:>7.2f}"
                  f"{b['ret']*100:>14.1f}")
    print("\nIf overnight ann% >> intraday across BOTH halves -> the effect is real & stable.")
    print("Caveat: capturing it needs a daily round-trip (open & close) -> heavy costs.")


def probe_trend(data):
    print("\n" + "=" * 68)
    print("PROBE 2 — Time-series momentum (long if trailing 12m > 0, else flat)")
    print("=" * 68)
    rets, strat = {}, {}
    for s, h in data.items():
        c = h["Close"]
        r = c.pct_change()
        signal = (c.shift(1) / c.shift(253) - 1) > 0   # trailing ~12m, as of yesterday
        rets[s] = r
        strat[s] = r.where(signal, 0.0)
    R = pd.DataFrame(rets).dropna(how="all")
    S = pd.DataFrame(strat).reindex(R.index)
    port_bh = R.mean(axis=1)          # equal-weight buy & hold
    port_tr = S.mean(axis=1)          # equal-weight trend
    half = len(R) // 2
    print(f"{'window':>10}{'   trend: ann%  Sharpe  maxDD':>34}"
          f"{'   buy&hold: ann%  Sharpe  maxDD':>36}")
    for tag, sl in (("full", slice(None)), ("1st-half", slice(0, half)),
                    ("2nd-half", slice(half, None))):
        t, b = ann_stats(port_tr.iloc[sl]), ann_stats(port_bh.iloc[sl])
        print(f"{tag:>10}{t['ret']*100:>14.1f}{t['sharpe']:>8.2f}{t['maxdd']*100:>8.0f}%"
              f"{b['ret']*100:>16.1f}{b['sharpe']:>8.2f}{b['maxdd']*100:>8.0f}%")
    print(f"\nBasket: {', '.join(R.columns)}")
    print("Trend wins if it MATCHES buy&hold return at a HIGHER Sharpe / smaller drawdown,")
    print("stably across both halves. Turnover is low (sign flips) -> costs are minor.")


def main():
    print("Fetching daily data (free)...", flush=True)
    data = fetch(sorted(set(EQUITY + TREND)))
    print(f"Got {len(data)} symbols.", flush=True)
    probe_overnight(data)
    probe_trend(data)
    print("\nBoth are documented anomalies; this is a first read, not proof. Survivors")
    print("get a proper cost + true out-of-sample test before being believed.")


if __name__ == "__main__":
    main()
