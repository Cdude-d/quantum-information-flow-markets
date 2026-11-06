"""Crash test: always-sell delta-hedged 0DTE SPY straddle held to expiry on 2018-07..2022-12
(the Mon/Wed/Fri expiry days before daily expiries), covering Dec 2018, the March 2020 crash
with market-wide trading halts, and the 2022 bear market. Same simulation and costs as
odte_to_close.py; the forecast is not used (always-sell), so nothing here was fit on these days.

Run:  python research/odte_history.py
"""
import numpy as np
import pandas as pd

from odte_delta_hedged import HALF_SPREAD
from odte_to_close import rest_of_day, simulate
from odte_to_close_audit import summ
from opening_screen import load_bars

START, END = "2018-06-01", "2022-12-30"
STRESS = ["2018-10-10", "2018-12-24", "2019-08-05", "2020-02-24", "2020-02-27", "2020-03-09",
          "2020-03-11", "2020-03-13", "2020-03-16", "2020-03-18", "2020-03-20", "2020-03-23",
          "2020-03-24", "2020-06-11", "2021-01-27", "2022-05-18", "2022-06-13", "2022-09-13"]


def day_list(bars, rv):
    """SPY days straight from minute bars (no dependence on opening-window features, which
    are missing on halt days): ATM strike from the 9:50 price; half-days excluded by an
    early last bar, NOT by bar count (a mid-day trading halt also lowers the bar count)."""
    spy = bars[bars.symbol == "SPY"]
    rows = []
    for d, g in spy.groupby("date"):
        if not (START <= str(d) <= END):
            continue
        pre = g[g.mins <= 19]
        if pre.empty or g.mins.max() < 385:      # no 9:50 price, or early close
            continue
        rows.append(dict(date=pd.Timestamp(d), strike=float(round(pre.close.iloc[-1]))))
    f = pd.DataFrame(rows)
    f["date_d"] = f.date.dt.date
    r = rv[rv.symbol == "SPY"].rename(columns={"date": "date_d"})
    f = f.merge(r[["date_d", "rv"]], on="date_d", how="left")
    f["lrv"] = np.log(f.rv.clip(lower=1e-12))
    f["log_fcst"] = np.nan                       # always-sell: no forecast needed
    return f


def main():
    print("Loading minute bars...", flush=True)
    bars = load_bars()
    rv, paths = rest_of_day(bars)
    f = day_list(bars, rv)

    t = simulate(f, paths)
    print(f"\n{len(t)} 0DTE days simulated, {t.date.min().date()}..{t.date.max().date()}\n")
    print("ALWAYS SELL, real fills (sell at bid + $1.30, hedge $0.005/sh, 5-min rehedge)")
    print(f"  hedged    {summ(t.short_real_h)}")
    print(f"  unhedged  {summ(t.short_real)}")
    print(f"  avg premium ${t.premium.mean():.0f}, entry spread ${t.spread.mean():.1f} "
          f"({(t.spread / t.premium).median()*100:.1f}% of premium)")
    print("\n  by year (hedged | unhedged):")
    for y, g in t.groupby(t.date.dt.year):
        print(f"    {y}: {len(g):>3} days  hedged {summ(g.short_real_h)}")
        print(f"    {'':4}{'':>10}  unhedg {summ(g.short_real)}")

    x = t.short_real_h
    print(f"\n  tail (hedged): std ${x.std():.0f}  skew {x.skew():+.2f}  "
          f"1st pct ${x.quantile(0.01):+.0f}  worst ${x.min():+.0f}")
    cols = ["date", "short_real_h", "short_real", "net_move_pct", "range_pct", "iv", "premium"]
    w = t.nsmallest(10, "short_real_h")[cols].copy()
    w["date"] = w.date.dt.date
    print("  10 worst days:")
    print(w.to_string(index=False, formatters={
        "short_real_h": "{:+.0f}".format, "short_real": "{:+.0f}".format,
        "net_move_pct": "{:+.2f}".format, "range_pct": "{:.2f}".format,
        "iv": "{:.0%}".format, "premium": "{:.0f}".format}))
    print("\n  stress days (hedged / unhedged):")
    have = {d.date().isoformat(): r for d, r in zip(t.date, t.itertuples())}
    for d in STRESS:
        r = have.get(d)
        if r is None:
            print(f"    {d}: no 0DTE expiry that day (or no data)")
        else:
            print(f"    {d}: hedged {r.short_real_h:+7.0f}  unhedged {r.short_real:+7.0f}  "
                  f"net {r.net_move_pct:+.2f}%  range {r.range_pct:.2f}%  IV {r.iv:.0%}  "
                  f"premium ${r.premium:.0f}")

    covid = t[(t.date >= "2020-02-20") & (t.date <= "2020-04-30")]
    print(f"\n  Feb 20 - Apr 30 2020 ({len(covid)} 0DTE days): hedged {summ(covid.short_real_h)}")

    print("\n  slippage stress (hedged):")
    for hs, os_ in ((0.005, 0.0), (0.02, 0.0), (0.02, 0.02), (0.05, 0.0)):
        adj = t.short_real_h - t.shares_traded * (hs - HALF_SPREAD) - 200 * os_
        print(f"    hedge ${hs:.3f}/sh, entry {os_*100:.0f}c/leg below bid: {summ(adj)}")


if __name__ == "__main__":
    main()
