"""Trade the next-hour range forecast with 0DTE SPY ATM straddles, held 9:50 -> 10:50 ET,
over the held-out period (2023-01-03 .. 2026-10-30).

Rules fixed before looking at option data:
  - Forecast: the opening_screen 'all features + baseline' model for next-hour range,
    fit on 2018-2022 only (no refitting).
  - Thresholds from 2018-2022 SPY fitted values: SELL the straddle when the forecast is
    below the 20th percentile, BUY when above the 80th.
  - Benchmarks: always-sell and always-buy. The forecast has value only if it picks
    better days than random -- tested with a permutation test on same-size random days.
  - Fills: mid, and real (cross the bid/ask on both legs, open and close) + $0.65/contract.

Run:  python research/odte_straddle_backtest.py
"""
from datetime import time as dtime
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

from odte_download import OUT, osi
from opening_screen import BASELINE, FEATURES, SPLIT, TARGETS, build_features, prep

COMM = 4 * 0.65                 # 2 legs open + 2 legs close, per straddle
ENTRY, EXIT = dtime(9, 50), dtime(10, 50)
rng = np.random.default_rng(20261103)


def forecasts():
    feat = build_features()
    feat["date_d"] = feat["date"].dt.date
    disc = feat[feat.date_d < SPLIT].reset_index(drop=True)
    conf = feat[feat.date_d >= SPLIT].reset_index(drop=True)
    cols = FEATURES + list(TARGETS)
    zd, zc = prep(disc, cols, disc), prep(conf, cols, disc)
    Xd = np.column_stack([np.ones(len(zd))] + [zd[c] for c in FEATURES])
    Xc = np.column_stack([np.ones(len(zc))] + [zc[c] for c in FEATURES])
    beta = np.linalg.lstsq(Xd, zd["t_range"].to_numpy(), rcond=None)[0]
    lo, hi = np.quantile((Xd @ beta)[disc.symbol.to_numpy() == "SPY"], [0.2, 0.8])
    conf["pred"] = Xc @ beta
    spy = conf[conf.symbol == "SPY"].copy()
    spy["spot950"] = spy.open_930 * (1 + spy.open_ret)
    spy["strike"] = spy.spot950.round()
    return spy, lo, hi


def straddle_quotes(day, strike):
    path = OUT / f"{day}.dbn.zst"
    if not path.exists():
        return None
    q = db.DBNStore.from_file(path).to_df()
    q.index = q.index.tz_convert("America/New_York")
    out = {}
    for tag, t in (("in", ENTRY), ("out", EXIT)):
        snap = q[q.index.time == t]
        legs = {}
        for cp in ("C", "P"):
            r = snap[snap.symbol == osi(day, cp, strike)]
            if r.empty:
                return None
            b, a = float(r.bid_px_00.iloc[-1]), float(r.ask_px_00.iloc[-1])
            if not (np.isfinite(b) and np.isfinite(a)) or a <= 0 or b < 0 or a < b:
                return None
            legs[cp] = (b, a)
        out[f"{tag}_bid"] = legs["C"][0] + legs["P"][0]
        out[f"{tag}_ask"] = legs["C"][1] + legs["P"][1]
    if out["in_bid"] <= 0:
        return None
    return out


def stats(daily, traded):
    d = np.asarray(daily, float)
    t = d[traded]
    cum = np.cumsum(d)
    sh = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan
    return dict(n=int(traded.sum()), avg=t.mean() if len(t) else np.nan,
                win=(t > 0).mean() if len(t) else np.nan, total=d.sum(), sharpe=sh,
                maxdd=(cum - np.maximum.accumulate(cum)).min())


def perm_p(all_pnl, chosen_mean, n, iters=10000):
    """P(random same-size subset of days has mean >= the model's chosen days)."""
    draws = np.array([rng.choice(all_pnl, n, replace=False).mean() for _ in range(iters)])
    return (np.sum(draws >= chosen_mean) + 1) / (iters + 1)


def main():
    spy, lo, hi = forecasts()
    rows = []
    for _, r in spy.iterrows():
        qd = straddle_quotes(r.date.date(), r.strike)
        if qd:
            qd.update(date=r.date, pred=r.pred, spot=r.spot950, t_range=r.t_range)
            rows.append(qd)
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"{len(d)} of {len(spy)} held-out days with clean quotes at 9:50 and 10:50 "
          f"({d.date.min().date()}..{d.date.max().date()})\n")

    mid_in = (d.in_bid + d.in_ask) / 2
    mid_out = (d.out_bid + d.out_ask) / 2
    pnl = {
        "long_mid": (mid_out - mid_in) * 100,
        "long_real": (d.out_bid - d.in_ask) * 100 - COMM,
        "short_mid": (mid_in - mid_out) * 100,
        "short_real": (d.in_bid - d.out_ask) * 100 - COMM,
    }
    is_lo, is_hi = (d.pred < lo).to_numpy(), (d.pred > hi).to_numpy()
    allday = np.ones(len(d), bool)

    print(f"Model picks: SELL on {is_lo.sum()} days (forecast < 2018-22 20th pct), "
          f"BUY on {is_hi.sum()} days (> 80th pct)\n")
    for fill in ("mid", "real"):
        lp, sp = pnl[f"long_{fill}"].to_numpy(), pnl[f"short_{fill}"].to_numpy()
        print("=" * 82)
        print(f"FILLS: {fill.upper()}" + ("  (cross bid/ask both legs, open+close, + $2.60 comm)"
                                           if fill == "real" else "  (optimistic)"))
        print("=" * 82)
        print(f"  {'strategy':28}{'trades':>7}{'$/trade':>9}{'win%':>7}{'total $':>10}"
              f"{'Sharpe':>8}{'maxDD $':>10}")
        strategies = [
            ("always SELL straddle", np.where(allday, sp, 0), allday),
            ("always BUY straddle", np.where(allday, lp, 0), allday),
            ("model SELL (quiet days)", np.where(is_lo, sp, 0), is_lo),
            ("model BUY (wild days)", np.where(is_hi, lp, 0), is_hi),
            ("model combined", np.where(is_lo, sp, 0) + np.where(is_hi, lp, 0), is_lo | is_hi),
        ]
        for name, daily, traded in strategies:
            s = stats(daily, traded)
            print(f"  {name:28}{s['n']:>7}{s['avg']:>9.1f}{s['win']*100:>6.0f}%{s['total']:>10,.0f}"
                  f"{s['sharpe']:>8.2f}{s['maxdd']:>10,.0f}")
        p_sell = perm_p(sp, sp[is_lo].mean(), int(is_lo.sum()))
        p_buy = perm_p(lp, lp[is_hi].mean(), int(is_hi.sum()))
        print(f"\n  Does the forecast pick better days than random? (permutation test)")
        print(f"    SELL on model-quiet days: ${sp[is_lo].mean():+.1f}/trade vs ${sp.mean():+.1f} "
              f"on all days -> p = {p_sell:.3f}")
        print(f"    BUY on model-wild days:   ${lp[is_hi].mean():+.1f}/trade vs ${lp.mean():+.1f} "
              f"on all days -> p = {p_buy:.3f}\n")

    # Is the forecast already priced? (descriptive, OOS quintiles of the forecast)
    d["implied_move"] = mid_in / d.spot * 100
    d["realized_range"] = d.t_range * 100
    d["long_mid"] = pnl["long_mid"]
    d["q"] = pd.qcut(d.pred, 5, labels=["Q1 quiet", "Q2", "Q3", "Q4", "Q5 wild"])
    print("=" * 82)
    print("IS IT PRICED IN?  by forecast quintile (held-out, descriptive)")
    print("=" * 82)
    print(f"  {'forecast':10}{'straddle price (% spot)':>25}{'realized 1h range %':>22}"
          f"{'ratio':>8}{'long $/trade (mid)':>20}")
    for q, g in d.groupby("q", observed=True):
        ratio = g.realized_range.mean() / g.implied_move.mean()
        print(f"  {str(q):10}{g.implied_move.mean():>25.2f}{g.realized_range.mean():>22.2f}"
              f"{ratio:>8.2f}{g.long_mid.mean():>20.1f}")
    print("  (ratio flat across quintiles => the straddle price already moves with the forecast)")

    combo = np.where(is_lo, pnl["short_real"], 0) + np.where(is_hi, pnl["long_real"], 0)
    d["combo_real"] = combo
    print("\nModel combined, REAL fills, by year:")
    for y, g in d.groupby(d.date.dt.year):
        tr = (is_lo | is_hi)[g.index]
        print(f"  {y}: {int(tr.sum()):>3} trades, total ${g.combo_real.sum():>8,.0f}, "
              f"$/trade {g.combo_real[tr].mean() if tr.any() else float('nan'):>6.1f}")


if __name__ == "__main__":
    main()
