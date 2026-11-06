"""Audit the delta-hedged short 0DTE SPY straddle held to expiry (odte_to_close.py), which
showed Sharpe ~4 after costs -- too good to accept without trying to break it.

  1. Hedge sanity: hedged P&L should track realized variance and ignore the net move.
  2. Always-sell over ALL held-out days (Jan 2023 onward, incl. March 2023 SVB), by year.
  3. Where the money comes from: implied vs realized volatility.
  4. Tail: distribution, 10 worst days, and named stress days.
  5. Hedge frequency 1 / 5 / 15 / 60 minutes (re-simulated).
  6. Slippage stress: hedge fills $0.005-$0.05/share, option entry $0-$0.05/leg below the bid.
  7. Return on Reg-T margin (~20% of notional + premium for a naked straddle).

Run:  python research/odte_to_close_audit.py
"""
import numpy as np
import pandas as pd

from odte_delta_hedged import HALF_SPREAD
from odte_to_close import CLOSE_M, ENTRY_M, prepare, simulate

STRESS_DAYS = ["2023-03-10", "2023-03-13", "2023-03-15", "2024-08-05", "2024-12-18",
               "2025-03-10", "2025-04-03", "2025-04-04", "2025-04-07", "2025-04-08",
               "2025-04-09", "2025-04-10", "2025-04-21"]
YEAR_MIN = 365 * 24 * 60


def summ(x):
    x = np.asarray(x, float)
    cum = np.cumsum(x)
    return (f"$/trade {x.mean():+7.1f} | win {np.mean(x > 0)*100:3.0f}% | Sharpe "
            f"{x.mean() / x.std() * np.sqrt(252):5.2f} | maxDD ${(cum - np.maximum.accumulate(cum)).min():>8,.0f}"
            f" | total ${x.sum():>9,.0f}")


def main():
    print("Preparing forecast + minute paths...", flush=True)
    spy, paths = prepare(verbose=False)
    t = simulate(spy, paths)
    print(f"\n{len(t)} held-out full days, {t.date.min().date()}..{t.date.max().date()}\n")

    print("1) HEDGE SANITY")
    print(f"   corr(hedged long P&L, realized $ variance) {t.long_mid_h.corr(t.rv_dollar):+.2f}  (want +)")
    print(f"   corr(hedged long P&L, net move)            {t.long_mid_h.corr(t.net_move_pct):+.2f}  (want ~0)")
    print(f"   corr(unhedged long P&L, |net move|)        {t.long_mid.corr(t.net_move_pct.abs()):+.2f}")
    print(f"   avg SPY shares traded per straddle         {t.shares_traded.mean():.0f}\n")

    print("2) ALWAYS SELL, ALL held-out days (no warm-up)")
    print(f"   hedged, real   {summ(t.short_real_h)}")
    print(f"   unhedged, real {summ(t.short_real)}")
    print("   by year (hedged, real):")
    for y, g in t.groupby(t.date.dt.year):
        print(f"     {y}: {len(g):>3} days  {summ(g.short_real_h)}")

    implied_var = t.iv ** 2 * (CLOSE_M - ENTRY_M) / YEAR_MIN
    realized_var = np.exp(t.lrv)
    print("\n3) WHERE THE MONEY COMES FROM")
    print(f"   mean implied variance / mean realized variance (9:50->close): "
          f"{implied_var.mean() / realized_var.mean():.2f}  "
          f"(vol ratio {np.sqrt(implied_var.mean() / realized_var.mean()):.2f})")
    print(f"   days realized < implied: {(realized_var < implied_var).mean()*100:.0f}%")
    print(f"   avg premium ${t.premium.mean():.0f} | avg entry spread ${t.spread.mean():.1f} "
          f"({(t.spread / t.premium).median()*100:.1f}% of premium) | hedged P&L "
          f"{(t.short_real_h / t.premium).mean()*100:+.1f}% of premium")

    x = t.short_real_h
    print("\n4) TAIL (hedged, real)")
    print(f"   std ${x.std():.0f}  skew {x.skew():+.2f}  1st pct ${x.quantile(0.01):+.0f}  "
          f"worst ${x.min():+.0f}  best ${x.max():+.0f}")
    cols = ["date", "short_real_h", "short_real", "net_move_pct", "range_pct", "iv", "premium"]
    w = t.nsmallest(10, "short_real_h")[cols].copy()
    w["date"] = w.date.dt.date
    print("   10 worst days:")
    print(w.to_string(index=False, formatters={
        "short_real_h": "{:+.0f}".format, "short_real": "{:+.0f}".format,
        "net_move_pct": "{:+.2f}".format, "range_pct": "{:.2f}".format,
        "iv": "{:.0%}".format, "premium": "{:.0f}".format}))
    print("   named stress days (present? hedged / unhedged P&L):")
    have = {d.date().isoformat(): r for d, r in zip(t.date, t.itertuples())}
    for d in STRESS_DAYS:
        if d in have:
            r = have[d]
            print(f"     {d}: hedged {r.short_real_h:+7.0f}  unhedged {r.short_real:+7.0f}  "
                  f"net move {r.net_move_pct:+.2f}%  range {r.range_pct:.2f}%  IV {r.iv:.0%}")
        else:
            print(f"     {d}: NOT IN SAMPLE")

    print("\n5) HEDGE FREQUENCY (always sell, hedged, real)")
    for rh in (1, 5, 15, 60):
        tt = t if rh == 5 else simulate(spy, paths, rehedge=rh)
        print(f"   every {rh:>2} min: {summ(tt.short_real_h)}  "
              f"(shares/day {tt.shares_traded.mean():.0f})", flush=True)

    print("\n6) SLIPPAGE STRESS (always sell, hedged, rehedge 5 min)")
    for hs in (0.005, 0.02, 0.05):
        for os_ in (0.0, 0.02, 0.05):
            adj = t.short_real_h - t.shares_traded * (hs - HALF_SPREAD) - 200 * os_
            print(f"   hedge ${hs:.3f}/sh, entry {os_*100:.0f}c/leg below bid: {summ(adj)}")

    margin = 0.20 * t.spot * 100 + t.premium                         # Reg-T naked ATM straddle
    ann = t.short_real_h.mean() * 252 / margin.mean()
    print(f"\n7) RETURN ON REG-T MARGIN: avg margin ~${margin.mean():,.0f}/straddle -> "
          f"{ann*100:.0f}%/yr at the backtest's average P&L (before taxes; ignores capacity)")


if __name__ == "__main__":
    main()
