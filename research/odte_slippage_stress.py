"""Regime-dependent cost stress for the always-sell delta-hedged 0DTE SPY straddle, over both
periods (2018-07..2022-12 and 2023-01..2026-10, ~1,670 days).

Uniform slippage understates the real risk: spreads blow out exactly on the panic days that
earn the biggest premiums. Here hedge cost ($/share) and entry slippage ($/leg below the bid)
depend on the day's 9:50 implied vol.

Run:  python research/odte_slippage_stress.py
"""
import numpy as np
import pandas as pd

from odte_delta_hedged import HALF_SPREAD
from odte_history import day_list
from odte_to_close import prepare, rest_of_day, simulate
from odte_to_close_audit import summ
from opening_screen import load_bars


def tiers(iv, cuts):
    """cuts: list of (iv_threshold, hedge $/sh, entry $/leg), ascending; default = base."""
    hs = np.full(len(iv), HALF_SPREAD)
    os_ = np.zeros(len(iv))
    for thr, h, o in cuts:
        hs = np.where(iv > thr, h, hs)
        os_ = np.where(iv > thr, o, os_)
    return hs, os_


SCENARIOS = {
    "base: $0.005/sh, fill at bid, every day": lambda iv: tiers(iv, []),
    "crisis A: IV>60% -> $0.05/sh + 5c/leg": lambda iv: tiers(iv, [(0.60, 0.05, 0.05)]),
    "crisis B: IV>40% -> $0.05+5c; IV>80% -> $0.10+10c":
        lambda iv: tiers(iv, [(0.40, 0.05, 0.05), (0.80, 0.10, 0.10)]),
    "proportional: costs scale with IV / 15%":
        lambda iv: (np.maximum(HALF_SPREAD, 0.005 * iv / 0.15), 0.01 * iv / 0.15),
    "proportional x2":
        lambda iv: (np.maximum(HALF_SPREAD, 0.010 * iv / 0.15), 0.02 * iv / 0.15),
    "proportional x3":
        lambda iv: (np.maximum(HALF_SPREAD, 0.015 * iv / 0.15), 0.03 * iv / 0.15),
}
BUCKETS = [0, 0.20, 0.40, 0.80, np.inf]
LABELS = ["IV <20%", "20-40%", "40-80%", ">80%"]


def main():
    print("Simulating 2018-2022 (day list from bars)...", flush=True)
    bars = load_bars()
    rv, paths = rest_of_day(bars)
    t1 = simulate(day_list(bars, rv), paths).assign(period="2018-22")
    del bars
    print("Simulating 2023-2026 (held-out)...", flush=True)
    spy, paths2 = prepare(verbose=False)
    t2 = simulate(spy, paths2).assign(period="2023-26")
    t = pd.concat([t1, t2], ignore_index=True).sort_values("date").reset_index(drop=True)
    t["bucket"] = pd.cut(t.iv, BUCKETS, labels=LABELS)
    print(f"\n{len(t)} days, {t.date.min().date()}..{t.date.max().date()}  "
          f"({len(t1)} + {len(t2)})")
    print("days per 9:50 IV regime:", t.bucket.value_counts().reindex(LABELS).to_dict(), "\n")

    print("=" * 100)
    print("ALWAYS SELL, delta-hedged, real fills — regime-dependent cost scenarios")
    print("=" * 100)
    adj = {}
    for name, fn in SCENARIOS.items():
        hs, os_ = fn(t.iv.to_numpy())
        a = t.short_real_h - t.shares_traded * (hs - HALF_SPREAD) - 200 * os_
        adj[name] = a
        hi = a[t.iv > 0.40].sum()
        print(f"  {name}")
        print(f"    all     {summ(a)}")
        for p in ("2018-22", "2023-26"):
            print(f"    {p} {summ(a[t.period == p])}")
        print(f"    share of total profit from IV>40% days: "
              f"{hi / a.sum() * 100 if a.sum() else float('nan'):.0f}%\n")

    print("=" * 100)
    print("$/trade by IV regime under each scenario (how dependent is it on panic days?)")
    print("=" * 100)
    print(f"  {'scenario':52}" + "".join(f"{l:>11}" for l in LABELS))
    for name, a in adj.items():
        row = [a[t.bucket == l].mean() for l in LABELS]
        print(f"  {name[:52]:52}" + "".join(f"{v:>+11.1f}" for v in row))
    calm = t.iv <= 0.40
    base = adj["base: $0.005/sh, fill at bid, every day"]
    print(f"\n  excluding every IV>40% day entirely (base costs): {summ(base[calm])}")


if __name__ == "__main__":
    main()
