"""Direct test: does the 9:50 range forecast know anything the 0DTE straddle price doesn't?

Uses the same held-out days and quotes as odte_straddle_backtest.py (no new data).
  1. Information test: explain realized next-hour range (log) with the straddle-implied
     move alone vs implied + our forecast. Incremental R^2 / t-stat of the forecast.
  2. Gap trade: gap = log(our forecast range) - log(straddle-implied move). Buy the
     straddle when the gap is above its TRAILING 80th percentile, sell when below the
     trailing 20th (trailing 120 trading days only -> no look-ahead). Held 9:50 -> 10:50,
     mid and real fills, permutation test vs random days.

Run:  python research/odte_gap_test.py
"""
import numpy as np
import pandas as pd

from odte_straddle_backtest import COMM, forecasts, perm_p, stats, straddle_quotes
from opening_screen import SPLIT, build_features

TRAIL = 120


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    cov = (e @ e / (len(y) - X.shape[1])) * np.linalg.inv(X.T @ X)
    r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    return b, b / np.sqrt(np.diag(cov)), r2


def spy_forecasts():
    """Held-out SPY days with the range forecast converted back to log % range
    (inverting prep()'s per-symbol z-score with SPY discovery stats)."""
    spy, _, _ = forecasts()
    feat = build_features()
    disc = feat[(feat.symbol == "SPY") & (feat.date.dt.date < SPLIT)]
    x = np.log(disc.t_range.clip(lower=1e-9))
    xr = x.clip(x.quantile(0.01), x.quantile(0.99))
    spy["log_fcst"] = xr.mean() + xr.std() * spy.pred
    return spy


def main():
    spy = spy_forecasts()
    rows = []
    for _, r in spy.iterrows():
        qd = straddle_quotes(r.date.date(), r.strike)
        if qd:
            qd.update(date=r.date, log_fcst=r.log_fcst, spot=r.spot950, t_range=r.t_range)
            rows.append(qd)
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    mid_in = (d.in_bid + d.in_ask) / 2
    mid_out = (d.out_bid + d.out_ask) / 2
    d["log_implied"] = np.log(mid_in / d.spot)
    d["log_real"] = np.log(d.t_range.clip(lower=1e-9))
    d["gap"] = d.log_fcst - d.log_implied
    long_mid = ((mid_out - mid_in) * 100).to_numpy()
    long_real = ((d.out_bid - d.in_ask) * 100 - COMM).to_numpy()
    short_mid = -long_mid
    short_real = ((d.in_bid - d.out_ask) * 100 - COMM).to_numpy()

    print(f"{len(d)} held-out days ({d.date.min().date()}..{d.date.max().date()})\n")
    print("=" * 74)
    print("1) INFORMATION TEST: explaining realized next-hour range (log)")
    print("=" * 74)
    _, _, r2_imp = ols(d.log_real.to_numpy(), [d.log_implied.to_numpy()])
    _, _, r2_fc = ols(d.log_real.to_numpy(), [d.log_fcst.to_numpy()])
    b, t, r2_both = ols(d.log_real.to_numpy(), [d.log_implied.to_numpy(), d.log_fcst.to_numpy()])
    print(f"  straddle price alone        R^2 {r2_imp*100:5.1f}%")
    print(f"  our forecast alone          R^2 {r2_fc*100:5.1f}%")
    print(f"  both                        R^2 {r2_both*100:5.1f}%   "
          f"(forecast adds {(r2_both - r2_imp)*100:+.1f} pts)")
    print(f"  coefficients: straddle {b[1]:+.2f} (t {t[1]:.1f}), forecast {b[2]:+.2f} "
          f"(t {t[2]:.1f})")
    rho = pd.Series(d.gap).corr(pd.Series(long_mid), method="spearman")
    print(f"  rank corr(gap, long-straddle P&L) = {rho:+.3f}  "
          f"(>0 means 'forecast above price' days pay buyers)\n")

    hi_t = d.gap.shift(1).rolling(TRAIL).quantile(0.8)
    lo_t = d.gap.shift(1).rolling(TRAIL).quantile(0.2)
    live = hi_t.notna().to_numpy()
    buy = (d.gap > hi_t).to_numpy() & live
    sell = (d.gap < lo_t).to_numpy() & live
    print("=" * 74)
    print(f"2) GAP TRADE (trailing-{TRAIL}-day thresholds, {live.sum()} tradable days from "
          f"{d.date[live].min().date()})")
    print("=" * 74)
    for fill, lp, sp in (("MID", long_mid, short_mid), ("REAL", long_real, short_real)):
        lp_l, sp_l = lp[live], sp[live]
        b_l, s_l = buy[live], sell[live]
        print(f"  {fill} fills{'  (bid/ask both legs + $2.60 comm)' if fill == 'REAL' else ''}")
        print(f"    {'strategy':26}{'trades':>7}{'$/trade':>9}{'win%':>7}{'total $':>10}{'Sharpe':>8}")
        for name, daily, tr in (
                ("always sell", sp_l, np.ones(len(sp_l), bool)),
                ("BUY when gap high", np.where(b_l, lp_l, 0), b_l),
                ("SELL when gap low", np.where(s_l, sp_l, 0), s_l),
                ("gap combined", np.where(b_l, lp_l, 0) + np.where(s_l, sp_l, 0), b_l | s_l)):
            s = stats(daily, tr)
            print(f"    {name:26}{s['n']:>7}{s['avg']:>9.1f}{s['win']*100:>6.0f}%"
                  f"{s['total']:>10,.0f}{s['sharpe']:>8.2f}")
        print(f"    vs random days: BUY p = {perm_p(lp_l, lp_l[b_l].mean(), int(b_l.sum())):.3f}, "
              f"SELL p = {perm_p(sp_l, sp_l[s_l].mean(), int(s_l.sum())):.3f}\n")


if __name__ == "__main__":
    main()
