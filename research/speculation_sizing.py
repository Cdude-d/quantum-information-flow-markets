"""Is speculating with bought options 'valid'? Reframe: it's valid only if SIZED
to survive. This separates the two things people conflate and Monte-Carlos them.

  SHAPE  = the payoff convexity of long options (lose-small-often, win-big-rarely).
           Taken from our ACTUAL backtested long-option trades (real, stable).
  EDGE   = the mean return on premium. This is what you can't reliably get; we
           make it a DIAL and de-mean the empirical shape, so EDGE and SHAPE are
           independent. Buying options long-run has EDGE <= 0 (proven over 36y).

For each (edge, bet-fraction f), simulate many bankroll paths over N trades
(bankroll *= 1 + f*return each trade) and report where you land. This shows:
  - sizing CANNOT fix a non-positive edge (you just lose slower),
  - how much edge you'd need for speculation to be +EV,
  - that even WITH an edge, over-betting ruins you (Kelly).

Run:  python research/speculation_sizing.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "backtest" / "data"
START = 10_000.0
N_TRADES = 100          # ~2 years of weekly speculation
N_PATHS = 20_000
RUIN = 0.25             # 'ruined' = ended below 25% of starting bankroll
rng = np.random.default_rng(20260620)


def empirical_shape():
    """Return-on-premium for each real long-option trade, de-meaned to isolate SHAPE."""
    rs = []
    for f in ["bt_long_0.35.csv", "bt_long_0.50.csv"]:
        p = DATA / f
        if p.exists():
            d = pd.read_csv(p)
            r = d["pl"] / (d["debit"] * 100.0)   # P/L as fraction of premium paid
            rs.append(r[np.isfinite(r)])
    allr = pd.concat(rs).to_numpy()
    return allr - allr.mean(), allr.mean()       # (zero-mean shape, original mean)


def simulate(shape, edge, f):
    """N_PATHS bankroll paths; each trade draws a shape return + edge, bets fraction f."""
    draws = rng.choice(shape, size=(N_PATHS, N_TRADES), replace=True) + edge
    growth = 1.0 + f * draws
    growth = np.maximum(growth, 0.0)             # can't go below 0 on one bet
    final = START * growth.prod(axis=1)
    return final


def main():
    shape, emp_mean = empirical_shape()
    print(f"Empirical long-option SHAPE from {len(shape)} real backtested trades:")
    print(f"  win rate {np.mean(shape + emp_mean > 0)*100:.0f}%  "
          f"median return {np.median(shape + emp_mean)*100:+.0f}%  "
          f"best {(shape+emp_mean).max()*100:+.0f}%  worst {(shape+emp_mean).min()*100:+.0f}%")
    print(f"  (in-sample mean was {emp_mean*100:+.0f}% — regime-inflated; true long-run edge <= 0)\n")

    edges = {"-10% (realistic, the VRP drag)": -0.10,
             "  0% (break-even after premium)": 0.0,
             "+10% (if you had real skill)": 0.10,
             "+25% (elite, unlikely)": 0.25}
    fracs = [0.02, 0.05, 0.10, 0.20, 0.50]

    print(f"After {N_TRADES} trades, starting ${START:,.0f}. Each cell: median outcome | "
          f"P(profit) | P(ruin<25%)")
    print(f"\n{'edge \\ bet size':32}" + "".join(f"{f'{int(f*100)}%':>16}" for f in fracs))
    print("-" * (32 + 16 * len(fracs)))
    for label, edge in edges.items():
        cells = []
        for f in fracs:
            final = simulate(shape, edge, f)
            med = np.median(final) / START
            p_profit = np.mean(final > START) * 100
            p_ruin = np.mean(final < START * RUIN) * 100
            cells.append(f"{med:4.1f}x {p_profit:2.0f}/{p_ruin:2.0f}")
        print(f"{label:32}" + "".join(f"{c:>16}" for c in cells))

    print("\n(cell = median ending multiple | % of paths profitable | % ruined)")

    # Kelly-style: best bet fraction by median log-growth, per edge.
    print("\nGrowth-optimal bet size (maximizes typical compounding):")
    grid = np.linspace(0.0, 1.0, 51)
    for label, edge in edges.items():
        best_f, best_g = 0.0, -1e9
        for f in grid:
            final = simulate(shape, edge, f)
            g = np.median(np.log(np.maximum(final / START, 1e-9)))
            if g > best_g:
                best_g, best_f = g, f
        verdict = "DON'T BET (no f wins)" if best_f < 1e-6 else f"bet ~{best_f*100:.0f}% per trade"
        print(f"  edge {label:32} -> {verdict}")

    print("\nReadout: negative/zero edge -> optimal bet is ZERO; any sizing just sets how")
    print("fast you bleed. A real positive edge -> a small optimal size; over-betting still")
    print("ruins you. Speculation is 'valid' only as bounded-cost convex fun, OR if you")
    print("genuinely have edge AND size small. Sizing controls survival, not the sign.")


if __name__ == "__main__":
    main()
