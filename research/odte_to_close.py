"""Forecast REST-OF-DAY variance at 9:50 and trade the 0DTE SPY straddle held to expiry.

Fixes the two reasons the 1-hour version wasn't tradable:
  horizon  forecast 9:50 -> 16:00 realized variance, the quantity the 0DTE straddle prices;
  costs    hold to expiry: options settle at intrinsic, so the bid/ask is crossed once, not twice.

  1. Forecast: same opening-window features, target log realized variance (sum of squared
     1-minute log returns, 9:50 -> close), OLS fit on 2018-2022, scored on 2023-2026.
  2. Information test (SPY held-out): realized variance explained by the 9:50 straddle-implied
     variance alone vs + our forecast.
  3. Trades: sell / buy the 9:50 ATM straddle, settle at intrinsic on the 4 PM close.
     Unhedged, and delta-hedged every REHEDGE min (BS delta at the entry implied vol, SPY at
     HALF_SPREAD $/share). REAL fills: entry crosses the bid/ask (+ OPT_SLIP per leg), $0.65 x 2
     contracts. Rules: always-sell, sell when gap = forecast - implied is below its trailing
     120-day 20th pct, buy when above the 80th. Half-days excluded.

Run:  python research/odte_to_close.py      (audit: research/odte_to_close_audit.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.pricing import bs_delta, implied_vol  # noqa: E402

from odte_delta_hedged import HALF_SPREAD, R, YEAR_MIN, minute_quotes  # noqa: E402
from odte_download import OUT as ODTE  # noqa: E402
from odte_gap_test import TRAIL, ols  # noqa: E402
from odte_straddle_backtest import perm_p, stats  # noqa: E402
from opening_screen import BASELINE, FEATURES, SPLIT, build_features, load_bars, prep  # noqa: E402

ENTRY_M, CLOSE_M = 20, 390
COMM_OPEN = 2 * 0.65
REHEDGE = 5


def rest_of_day(bars):
    """Per symbol-day: realized variance 9:50 -> close, bar count, and SPY minute closes."""
    # SPY/QQQ/IWM share timestamps: a positional index avoids misaligning on duplicates
    b = bars.sort_values(["symbol", "date", "mins"]).reset_index(drop=True)
    lr = np.log(b.close).groupby([b.symbol, b.date]).diff()
    b = b.assign(lr2=lr ** 2)
    after = b[b.mins >= ENTRY_M]
    rv = after.groupby(["symbol", "date"]).agg(rv=("lr2", "sum"), nbars=("mins", "size"))
    rv = rv.reset_index()
    spy = b[b.symbol == "SPY"]
    paths = {d: g.set_index(g.mins + 1).close for d, g in spy.groupby("date")}
    return rv, paths


def prepare(verbose=True):
    """Fit the rest-of-day variance forecast; return held-out SPY days and SPY minute paths."""
    rv, paths = rest_of_day(load_bars())
    feat = build_features()
    feat["date_d"] = feat.date.dt.date
    feat = feat.merge(rv.rename(columns={"date": "date_d"}), on=["symbol", "date_d"])
    feat = feat[feat.nbars >= 360].copy()                     # drop half-days
    feat["lrv"] = np.log(feat.rv.clip(lower=1e-12))

    disc = feat[feat.date_d < SPLIT].reset_index(drop=True)
    conf = feat[feat.date_d >= SPLIT].reset_index(drop=True)
    cols = FEATURES + ["lrv"]
    zd, zc = prep(disc, cols, disc), prep(conf, cols, disc)

    def fit_predict(xs):
        Xd = np.column_stack([np.ones(len(zd))] + [zd[c] for c in xs])
        Xc = np.column_stack([np.ones(len(zc))] + [zc[c] for c in xs])
        beta = np.linalg.lstsq(Xd, zd.lrv.to_numpy(), rcond=None)[0]
        pred = Xc @ beta
        y = zc.lrv.to_numpy()
        return pred, 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    _, r2_base = fit_predict(BASELINE)
    pred, r2_all = fit_predict(FEATURES)
    if verbose:
        print("=" * 76)
        print("1) FORECASTING REST-OF-DAY VARIANCE (9:50 -> close), held-out 2023-2026, SPY/QQQ/IWM")
        print("=" * 76)
        print(f"  pre-open info only (VIX, 20d vol, prior range): OOS R^2 {r2_base*100:5.1f}%")
        print(f"  + first 20 minutes:                              OOS R^2 {r2_all*100:5.1f}%\n")

    x = disc[disc.symbol == "SPY"].lrv       # back to log-variance units (invert prep for SPY)
    xr = x.clip(x.quantile(0.01), x.quantile(0.99))
    conf["log_fcst"] = xr.mean() + xr.std() * pred
    spy = conf[conf.symbol == "SPY"].copy()
    spy["spot950"] = spy.open_930 * (1 + spy.open_ret)
    spy["strike"] = spy.spot950.round()
    return spy, paths


def simulate(spy, paths, rehedge=REHEDGE, half_spread=HALF_SPREAD, opt_slip=0.0):
    """One row per held-out day: straddle P&L held to expiry, unhedged and delta-hedged."""
    rows = []
    for _, r in spy.iterrows():
        d = r.date.date()
        if d not in paths or not (ODTE / f"{d}.dbn.zst").exists():
            continue
        legs = minute_quotes(d, r.strike)
        c0, p0 = legs["C"].loc[ENTRY_M], legs["P"].loc[ENTRY_M]
        if c0.isna().any() or p0.isna().any() or c0.ask_px_00 <= 0 or p0.ask_px_00 <= 0:
            continue
        S = paths[d].reindex(range(ENTRY_M, CLOSE_M + 1)).ffill().to_numpy()
        if np.isnan(S).any():
            continue
        K = r.strike
        cmid, pmid = (c0.bid_px_00 + c0.ask_px_00) / 2, (p0.bid_px_00 + p0.ask_px_00) / 2
        t0 = (CLOSE_M - ENTRY_M) / YEAR_MIN
        ivc = implied_vol("call", cmid, S[0], K, t0, R)
        ivp = implied_vol("put", pmid, S[0], K, t0, R)
        if ivc is None and ivp is None:
            continue
        ivc, ivp = ivc or ivp, ivp or ivc
        shares, hedge, traded = 0.0, 0.0, 0.0           # hedge for a LONG straddle
        for i, m in enumerate(range(ENTRY_M, CLOSE_M)):
            if i % rehedge == 0:
                t = max(CLOSE_M - m, 1) / YEAR_MIN
                delta = bs_delta("call", S[i], K, t, ivc, R) + bs_delta("put", S[i], K, t, ivp, R)
                target = -100 * delta
                traded += abs(target - shares)
                shares = target
            hedge += shares * (S[i + 1] - S[i])
        traded += abs(shares)
        payoff = abs(S[-1] - K) * 100
        mid = (cmid + pmid) * 100
        hc = traded * half_spread
        sell_px = (c0.bid_px_00 + p0.bid_px_00 - 2 * opt_slip) * 100
        buy_px = (c0.ask_px_00 + p0.ask_px_00 + 2 * opt_slip) * 100
        rows.append(dict(
            date=r.date, log_fcst=r.log_fcst, lrv=r.lrv,
            log_implied=2 * np.log((cmid + pmid) / S[0]),
            short_mid=mid - payoff, short_real=sell_px - payoff - COMM_OPEN,
            long_mid=payoff - mid, long_real=payoff - buy_px - COMM_OPEN,
            short_mid_h=mid - payoff - hedge, short_real_h=sell_px - payoff - COMM_OPEN - hedge - hc,
            long_mid_h=payoff - mid + hedge, long_real_h=payoff - buy_px - COMM_OPEN + hedge - hc,
            premium=mid, spot=S[0],
            spread=(c0.ask_px_00 - c0.bid_px_00 + p0.ask_px_00 - p0.bid_px_00) * 100,
            iv=(ivc + ivp) / 2, rv_dollar=float(np.sum(np.diff(S) ** 2)),
            net_move_pct=(S[-1] / S[0] - 1) * 100,
            range_pct=(S.max() - S.min()) / S[0] * 100, shares_traded=traded))
    t = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    t["gap"] = t.log_fcst - t.log_implied
    return t


def report(t):
    print("=" * 76)
    print(f"2) INFORMATION TEST — SPY, {len(t)} held-out full days")
    print("=" * 76)
    _, _, r2_imp = ols(t.lrv.to_numpy(), [t.log_implied.to_numpy()])
    _, _, r2_fc = ols(t.lrv.to_numpy(), [t.log_fcst.to_numpy()])
    b, tt, r2_both = ols(t.lrv.to_numpy(), [t.log_implied.to_numpy(), t.log_fcst.to_numpy()])
    print(f"  straddle-implied variance alone  R^2 {r2_imp*100:5.1f}%")
    print(f"  our forecast alone               R^2 {r2_fc*100:5.1f}%")
    print(f"  both                             R^2 {r2_both*100:5.1f}%  "
          f"(forecast adds {(r2_both - r2_imp)*100:+.1f} pts, t {tt[2]:.1f})")
    for lbl, col in (("unhedged", "long_mid"), ("hedged", "long_mid_h")):
        print(f"  rank corr(gap, {lbl} long P&L) = "
              f"{t.gap.corr(t[col], method='spearman'):+.3f}")

    hi_t = t.gap.shift(1).rolling(TRAIL).quantile(0.8)
    lo_t = t.gap.shift(1).rolling(TRAIL).quantile(0.2)
    live = hi_t.notna().to_numpy()
    buy = ((t.gap > hi_t).to_numpy() & live)[live]
    sell = ((t.gap < lo_t).to_numpy() & live)[live]
    print("\n" + "=" * 76)
    print(f"3) TRADES held to expiry — {live.sum()} tradable days from {t.date[live].min().date()}")
    print("=" * 76)
    for hedged in ("", "_h"):
        for fill in ("mid", "real"):
            lp = t[f"long_{fill}{hedged}"].to_numpy()[live]
            sp = t[f"short_{fill}{hedged}"].to_numpy()[live]
            print(f"  {'DELTA-HEDGED' if hedged else 'UNHEDGED'}, {fill.upper()} fills")
            print(f"    {'strategy':22}{'trades':>7}{'$/trade':>9}{'win%':>7}{'total $':>10}"
                  f"{'Sharpe':>8}{'maxDD $':>10}")
            for name, daily, tr in (
                    ("always SELL", sp, np.ones(len(sp), bool)),
                    ("always BUY", lp, np.ones(len(lp), bool)),
                    ("SELL when gap low", np.where(sell, sp, 0), sell),
                    ("BUY when gap high", np.where(buy, lp, 0), buy),
                    ("gap combined", np.where(sell, sp, 0) + np.where(buy, lp, 0), sell | buy)):
                s = stats(daily, tr)
                print(f"    {name:22}{s['n']:>7}{s['avg']:>9.1f}{s['win']*100:>6.0f}%"
                      f"{s['total']:>10,.0f}{s['sharpe']:>8.2f}{s['maxdd']:>10,.0f}")
            print(f"    vs random days: SELL p = {perm_p(sp, sp[sell].mean(), int(sell.sum())):.3f}, "
                  f"BUY p = {perm_p(lp, lp[buy].mean(), int(buy.sum())):.3f}\n")


def main():
    print("Loading minute bars...", flush=True)
    spy, paths = prepare()
    report(simulate(spy, paths))


if __name__ == "__main__":
    main()
