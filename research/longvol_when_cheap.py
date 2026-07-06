"""Test the one legitimate 'buy undervalued options' idea over 30 years (free).

Hypothesis: options are underpriced (profitable to BUY) when implied vol is
unusually cheap, because vol mean-reverts up. We test it directly:

  For each day, a ~1-month ATM straddle costs ~0.8*IV*sqrt(t) of spot (standard
  approx). A naked long-straddle buyer's edge over the next month is:
        edge = |actual move|  -  implied move
  Positive edge = options were UNDERpriced (buyer wins). We bucket this by IV
  RANK (where VIX sits vs its own trailing year) to see if buying when vol is
  CHEAP actually pays.

Uses free ^VIX (implied) and ^GSPC (realized) daily history. ATM straddle is a
clean proxy for 'buy vol naked'.

Run:  python research/longvol_when_cheap.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

H = 21            # ~1-month horizon (trading days)
STRADDLE_K = 0.8  # ATM straddle price ~ 0.8 * sigma * sqrt(T) * S


def main():
    print("Fetching ^VIX and ^GSPC (free, ~30y)...", flush=True)
    vix = yf.Ticker("^VIX").history(period="max", interval="1d")["Close"]
    spx = yf.Ticker("^GSPC").history(period="max", interval="1d")["Close"]
    for s in (vix, spx):
        s.index = s.index.tz_localize(None).normalize()
    df = pd.concat([vix.rename("vix"), spx.rename("spx")], axis=1).dropna()
    print(f"  vix={len(vix)} spx={len(spx)} aligned={len(df)}", flush=True)
    if df.empty:
        print("No overlapping data.")
        return
    print(f"Sample: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)\n")

    iv = df["vix"] / 100.0
    implied_move = STRADDLE_K * iv * np.sqrt(H / 252.0)          # fraction of spot
    fwd = df["spx"].shift(-H) / df["spx"] - 1.0
    actual_move = fwd.abs()
    edge = actual_move - implied_move                            # long-straddle P/L / spot
    iv_rank = df["vix"].rolling(252).apply(
        lambda w: (w.iloc[-1] >= w).mean(), raw=False)           # 0..1 percentile

    d = pd.DataFrame({"iv": iv, "implied_move": implied_move, "actual_move": actual_move,
                      "edge": edge, "iv_rank": iv_rank}).dropna()

    print("=" * 64)
    print("LONG-STRADDLE BUYER'S EDGE  (edge>0 => options were underpriced)")
    print("=" * 64)
    print(f"  Overall mean edge: {d['edge'].mean()*100:+.2f}% of spot per month "
          f"({(d['edge']>0).mean()*100:.0f}% of months positive)")
    print(f"  (Negative = the variance risk premium: buyers lose on average.)\n")

    print("By IV RANK bucket (is buying when vol is CHEAP better?):")
    print(f"  {'IV-rank bucket':18}{'n':>6}{'mean edge':>12}{'% positive':>12}"
          f"{'avg IV':>9}")
    d["bucket"] = pd.cut(d["iv_rank"], [0, .2, .4, .6, .8, 1.0],
                         labels=["cheapest 0-20%", "20-40%", "40-60%", "60-80%",
                                 "richest 80-100%"])
    for b, g in d.groupby("bucket", observed=True):
        print(f"  {str(b):18}{len(g):>6}{g['edge'].mean()*100:>11.2f}%"
              f"{(g['edge']>0).mean()*100:>11.0f}%{g['iv'].mean()*100:>8.0f}%")

    cheap = d[d["iv_rank"] <= 0.2]["edge"]
    rich = d[d["iv_rank"] >= 0.8]["edge"]
    print(f"\n  Buy-when-CHEAP avg edge: {cheap.mean()*100:+.2f}%  | "
          f"Buy-when-RICH avg edge: {rich.mean()*100:+.2f}%")
    print("\nVerdict: if 'cheapest' bucket edge is solidly POSITIVE, buying undervalued")
    print("vol has a real edge. If it's <= 0, cheap IV is cheap for a reason.")
    print("Caveat: ignores bid/ask, slippage, and is held-to-expiry undelta-hedged.")


if __name__ == "__main__":
    main()
