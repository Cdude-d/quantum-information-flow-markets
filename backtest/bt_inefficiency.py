"""Measure option-market inefficiency in the snapshots we already have.

For each name/week: compare what the market PRICED IN at entry (ATM implied vol
& the ATM straddle = implied weekly move) against what actually HAPPENED
(realized vol that week, and the actual move to expiry). Aggregates the variance
risk premium and "straddle coverage" (how often price finished inside the
implied move) across both periods and per name.

This is measurement, not a strategy. Edges suggested here are hypotheses that
would still need their own out-of-sample test. No new data is pulled.

Usage:  python backtest/bt_inefficiency.py
"""
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout import strategies  # noqa: E402
from options_scout.config import load_config  # noqa: E402
from options_scout.pricing import year_fraction  # noqa: E402

from bt_run import (  # noqa: E402
    full_hist, load_week, settle_close, side_frame, spot_from_parity)

ET = ZoneInfo("America/New_York")
DIRS = {"in-sample": "chains", "OOS": "chains_oos"}


def atm(cp, pp, spot):
    cm = cp[cp["mid"] > 0][["strike", "mid", "iv"]].rename(columns={"mid": "c", "iv": "civ"})
    pm = pp[pp["mid"] > 0][["strike", "mid", "iv"]].rename(columns={"mid": "p", "iv": "piv"})
    m = cm.merge(pm, on="strike")
    if m.empty:
        return None
    row = m.iloc[(m["strike"] - spot).abs().argmin()]
    iv = np.nanmean([row["civ"], row["piv"]])
    return float(iv), float(row["c"] + row["p"])  # annualized IV, ATM straddle $


def realized_vol_week(symbol, entry, expiry):
    h = full_hist(symbol)
    seg = h[(h.index.date >= entry) & (h.index.date <= expiry)]["Close"]
    if len(seg) < 3:
        return None
    r = np.log(seg / seg.shift(1)).dropna()
    return float(r.std() * np.sqrt(252))


def main():
    cfg = load_config()
    r = cfg["risk_free_rate"]
    base = Path(__file__).resolve().parent / "data"
    rows = []
    for period, sub in DIRS.items():
        for path in sorted((base / sub).glob("*.dbn.zst")):
            entry, exp, df = load_week(path)
            t = year_fraction(datetime.combine(entry, dtime(10, 0), tzinfo=ET), exp)
            for sym, g in df.groupby("root"):
                calls, puts = side_frame(g[g.otype == "call"]), side_frame(g[g.otype == "put"])
                if len(calls) < 5 or len(puts) < 5:
                    continue
                spot = spot_from_parity(calls, puts)
                if not spot or spot <= 0:
                    continue
                cp = strategies.prep_side(calls, "call", spot, t, r)
                pp = strategies.prep_side(puts, "put", spot, t, r)
                a = atm(cp, pp, spot)
                if a is None:
                    continue
                iv, straddle = a
                settle = settle_close(sym, exp)
                rv = realized_vol_week(sym, entry, exp)
                if settle is None or rv is None or iv <= 0:
                    continue
                implied_move = straddle / spot          # fractional, this week
                actual_move = abs(settle - spot) / spot
                rows.append({
                    "period": period, "symbol": sym, "iv": iv, "rv": rv,
                    "iv_rv": iv / rv if rv > 0 else np.nan,
                    "implied_move": implied_move, "actual_move": actual_move,
                    "inside": int(actual_move < implied_move),
                })

    d = pd.DataFrame(rows)
    print(f"\n{len(d)} name-weeks measured "
          f"({(d.period=='in-sample').sum()} in-sample, {(d.period=='OOS').sum()} OOS)\n")

    def block(g, name):
        print(f"--- {name} (n={len(g)}) ---")
        print(f"  ATM implied vol (annualized):  {g['iv'].mean()*100:5.1f}%")
        print(f"  Realized vol that week:        {g['rv'].mean()*100:5.1f}%")
        print(f"  Variance risk premium IV/RV:   {g['iv_rv'].median():.2f}x median "
              f"({(g['iv']>g['rv']).mean()*100:.0f}% of weeks IV>RV)")
        print(f"  Implied weekly move priced:    {g['implied_move'].mean()*100:5.2f}%")
        print(f"  Actual weekly move realized:   {g['actual_move'].mean()*100:5.2f}%")
        print(f"  Finished INSIDE implied move:  {g['inside'].mean()*100:.0f}% of weeks "
              f"(seller-favorable if >~57%)")
        print()

    block(d, "ALL name-weeks")
    for p in ("in-sample", "OOS"):
        block(d[d.period == p], p)

    print("Richest names by IV/RV (most overpriced vol, both periods pooled):")
    by = d.groupby("symbol").agg(n=("iv_rv", "size"), iv_rv=("iv_rv", "median"),
                                 inside=("inside", "mean"),
                                 move_ratio=("implied_move", "mean")).sort_values("iv_rv", ascending=False)
    by["actual_mv"] = d.groupby("symbol")["actual_move"].mean()
    for sym, rr in by.iterrows():
        print(f"  {sym:5} IV/RV {rr['iv_rv']:.2f}x  inside {rr['inside']*100:3.0f}%  "
              f"impliedMove {rr['move_ratio']*100:.2f}% vs actual {rr['actual_mv']*100:.2f}%")
    d.to_csv(base / "inefficiency.csv", index=False)
    print(f"\nSaved per-name-week detail to {base / 'inefficiency.csv'}")


if __name__ == "__main__":
    main()
