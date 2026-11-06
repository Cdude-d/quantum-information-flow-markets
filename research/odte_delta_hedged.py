"""Delta-hedged 0DTE SPY straddle (gamma scalping), 9:50 -> 10:50 ET, held-out 2023-2026.

An unhedged straddle closed after an hour pays on the NET move; a delta-hedged one pays
on how much price wiggles along the way (realized variance) minus time decay -- the thing
the 9:50 range forecast actually predicts.

  Each minute m from 9:50 to 10:49: implied vol from each leg's mid (Black-Scholes, time
  to the 4 PM expiry), straddle delta = delta_call + delta_put, hold -100*delta SPY shares
  (for a long straddle). Rehedge every REHEDGE minutes; flatten the hedge at 10:50.
  Costs (REAL): straddle bid/ask on entry + exit, $0.65 x 4 contracts, $0.005/share per
  hedge trade (half of SPY's 1-cent spread). MID: option mids, no costs.
  Rule: same gap as odte_gap_test (log forecast range - log straddle-implied move), trailing
  120-day 20th/80th percentile thresholds -> sell / buy; plus always-buy / always-sell.

Run:  python research/odte_delta_hedged.py
"""
import sys
from datetime import date
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.pricing import bs_delta, implied_vol  # noqa: E402

from odte_download import OUT as ODTE, osi  # noqa: E402
from odte_gap_test import TRAIL, spy_forecasts  # noqa: E402
from odte_straddle_backtest import COMM, perm_p, stats  # noqa: E402
from opening_screen import DATA as BARS  # noqa: E402

ENTRY_M, EXIT_M = 20, 80                    # minutes after 9:30
HALF_SPREAD = 0.005                         # $/share per hedge trade
R = 0.045
YEAR_MIN = 365 * 24 * 60


def spy_minutes(days):
    """Spot at each minute boundary 9:30+m = close of the 1-minute bar starting at m-1."""
    out = {}
    for y in sorted({d.year for d in days}):
        df = db.DBNStore.from_file(BARS / f"xnas_ohlcv1m_{y}.dbn.zst").to_df()
        df = df[df.symbol == "SPY"]
        df.index = df.index.tz_convert("America/New_York")
        m = df.index.hour * 60 + df.index.minute - 570
        df = df[(m >= ENTRY_M - 1) & (m < EXIT_M)].assign(m=m[(m >= ENTRY_M - 1) & (m < EXIT_M)] + 1)
        for d, g in df.groupby(df.index.date):
            if d in days:
                out[d] = g.set_index("m").close
    return out


def minute_quotes(day, strike):
    q = db.DBNStore.from_file(ODTE / f"{day}.dbn.zst").to_df()
    q.index = q.index.tz_convert("America/New_York")
    q["m"] = q.index.hour * 60 + q.index.minute - 570
    legs = {}
    for cp in ("C", "P"):
        x = q[q.symbol == osi(day, cp, strike)].groupby("m")[["bid_px_00", "ask_px_00"]].last()
        legs[cp] = x.reindex(range(ENTRY_M, EXIT_M + 1)).ffill()
    return legs


def straddle_delta(s, k, mins_left, c_mid, p_mid, prev):
    t = max(mins_left, 1) / YEAR_MIN
    ivc = implied_vol("call", c_mid, s, k, t, R)
    ivp = implied_vol("put", p_mid, s, k, t, R)
    if ivc is None and ivp is None:
        return prev
    ivc, ivp = ivc or ivp, ivp or ivc
    return bs_delta("call", s, k, t, ivc, R) + bs_delta("put", s, k, t, ivp, R)


def hedged_day(day, strike, spot, rehedge):
    legs = minute_quotes(day, strike)
    c, p = legs["C"], legs["P"]
    if c.isna().any().any() or p.isna().any().any() or spot.reindex(
            range(ENTRY_M, EXIT_M + 1)).isna().any():
        return None
    S = spot.reindex(range(ENTRY_M, EXIT_M + 1)).to_numpy()
    cmid = ((c.bid_px_00 + c.ask_px_00) / 2).to_numpy()
    pmid = ((p.bid_px_00 + p.ask_px_00) / 2).to_numpy()
    shares, hedge_pnl, traded, delta = 0.0, 0.0, 0.0, 0.0
    for i, m in enumerate(range(ENTRY_M, EXIT_M)):
        if i % rehedge == 0:
            delta = straddle_delta(S[i], strike, 390 - m, cmid[i], pmid[i], delta)
            target = -100 * delta                      # shares to hold against a LONG straddle
            traded += abs(target - shares)
            shares = target
        hedge_pnl += shares * (S[i + 1] - S[i])
    traded += abs(shares)                               # flatten at 10:50
    e, x = 0, EXIT_M - ENTRY_M
    opt_mid = (cmid[x] + pmid[x] - cmid[e] - pmid[e]) * 100
    long_real_opt = (c.bid_px_00.iloc[x] + p.bid_px_00.iloc[x]
                     - c.ask_px_00.iloc[e] - p.ask_px_00.iloc[e]) * 100
    short_real_opt = (c.bid_px_00.iloc[e] + p.bid_px_00.iloc[e]
                      - c.ask_px_00.iloc[x] - p.ask_px_00.iloc[x]) * 100
    hcost = traded * HALF_SPREAD
    return dict(
        long_mid=opt_mid + hedge_pnl, short_mid=-(opt_mid + hedge_pnl),
        long_real=long_real_opt + hedge_pnl - hcost - COMM,
        short_real=short_real_opt - hedge_pnl - hcost - COMM,
        unhedged_long_mid=opt_mid, rv=float(np.sum(np.diff(S) ** 2)),
        net_move=float(S[-1] - S[0]), shares_traded=traded,
        implied=(cmid[e] + pmid[e]) / S[0])


def main():
    spy = spy_forecasts()
    days = {d.date() for d in spy.date}
    print("Loading SPY minute bars...", flush=True)
    spots = spy_minutes(days)

    for rehedge in (1, 5):
        rows = []
        for _, r in spy.iterrows():
            d = r.date.date()
            if d not in spots or not (ODTE / f"{d}.dbn.zst").exists():
                continue
            res = hedged_day(d, r.strike, spots[d], rehedge)
            if res:
                res.update(date=r.date, log_fcst=r.log_fcst)
                rows.append(res)
        t = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        t["gap"] = t.log_fcst - np.log(t.implied)

        print("\n" + "=" * 78)
        print(f"DELTA-HEDGED STRADDLE, rehedge every {rehedge} min — {len(t)} held-out days")
        print("=" * 78)
        print("  Sanity checks (is the hedge doing its job?):")
        print(f"    corr(hedged long P&L, realized variance)  {t.long_mid.corr(t.rv):+.2f}  "
              f"(should be strongly +)")
        print(f"    corr(hedged long P&L, net move)           {t.long_mid.corr(t.net_move):+.2f}  "
              f"(should be ~0)")
        print(f"    corr(UNhedged long P&L, |net move|)       "
              f"{t.unhedged_long_mid.corr(t.net_move.abs()):+.2f}  (why unhedged pays on net move)")
        print(f"    avg SPY shares traded per straddle        {t.shares_traded.mean():.0f}")
        rho = t.gap.corr(t.long_mid, method="spearman")
        print(f"  rank corr(gap, HEDGED long P&L) = {rho:+.3f}   (was -0.06 unhedged)")

        hi_t = t.gap.shift(1).rolling(TRAIL).quantile(0.8)
        lo_t = t.gap.shift(1).rolling(TRAIL).quantile(0.2)
        live = hi_t.notna().to_numpy()
        buy = ((t.gap > hi_t).to_numpy() & live)[live]
        sell = ((t.gap < lo_t).to_numpy() & live)[live]
        for fill in ("mid", "real"):
            lp = t[f"long_{fill}"].to_numpy()[live]
            sp = t[f"short_{fill}"].to_numpy()[live]
            print(f"\n  {fill.upper()} fills{'  (option bid/ask + $2.60 + hedge $0.005/sh)' if fill == 'real' else ''}"
                  f" — {live.sum()} tradable days")
            print(f"    {'strategy':24}{'trades':>7}{'$/trade':>9}{'win%':>7}{'total $':>10}{'Sharpe':>8}")
            for name, daily, tr in (
                    ("always BUY (hedged)", lp, np.ones(len(lp), bool)),
                    ("always SELL (hedged)", sp, np.ones(len(sp), bool)),
                    ("BUY when gap high", np.where(buy, lp, 0), buy),
                    ("SELL when gap low", np.where(sell, sp, 0), sell),
                    ("gap combined", np.where(buy, lp, 0) + np.where(sell, sp, 0), buy | sell)):
                s = stats(daily, tr)
                print(f"    {name:24}{s['n']:>7}{s['avg']:>9.1f}{s['win']*100:>6.0f}%"
                      f"{s['total']:>10,.0f}{s['sharpe']:>8.2f}")
            print(f"    vs random days: BUY p = {perm_p(lp, lp[buy].mean(), int(buy.sum())):.3f}, "
                  f"SELL p = {perm_p(sp, sp[sell].mean(), int(sell.sum())):.3f}")


if __name__ == "__main__":
    main()
