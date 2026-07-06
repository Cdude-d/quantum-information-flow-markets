"""Survivorship-bias-FREE cross-sectional momentum: rank the 11 sector ETFs by
trailing-J return, long top / short bottom, hold 21d. Sectors never delist, so
zero survivorship bias -- the honest read on whether the momentum we saw in the
hand-picked stock universe is real or inflated. Drawdown/crash stats included so
the tail under the Sharpe is visible. Free yfinance data.

Run:  python research/sector_momentum.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
FORMATIONS = [5, 21, 63, 126]
HOLD = 21
PERIOD = "12y"
COST = 0.0002          # ~2 bps (very liquid sector ETFs)
N_SIDE_FRAC = 0.3      # ~3 names each side for 11 sectors


def stats(daily):
    d = np.asarray(daily, float)
    d = d[~np.isnan(d)]
    if len(d) < 60:
        return dict(ann=np.nan, sharpe=np.nan, maxdd=np.nan, worst21=np.nan)
    vol = d.std() * np.sqrt(252)
    cum = np.cumprod(1 + d)
    maxdd = (cum / np.maximum.accumulate(cum) - 1).min()
    worst21 = pd.Series(d).rolling(21).sum().min()
    return dict(ann=(1 + d).prod() ** (252 / len(d)) - 1,
                sharpe=(d.mean() * 252) / vol if vol > 0 else np.nan,
                maxdd=maxdd, worst21=worst21)


def run(P, R, J, K):
    n = len(R)
    gross = np.zeros(n)
    cost = np.zeros(n)
    cur_long, cur_short = [], []
    lp, sp = set(), set()
    k = max(J, 1)
    while k < n:
        if (k - J) % K == 0 or not cur_long:
            past = (P.iloc[k] / P.iloc[k - J] - 1.0).dropna()
            if len(past) >= 6:
                ranked = past.sort_values()
                m = max(2, round(len(ranked) * N_SIDE_FRAC))
                cur_short = list(ranked.index[:m])
                cur_long = list(ranked.index[-m:])
                turn = len(set(cur_long) ^ lp) + len(set(cur_short) ^ sp)
                cost[k] = turn / (2 * m) * COST
                lp, sp = set(cur_long), set(cur_short)
        if cur_long:
            gross[k] = R.iloc[k][cur_long].mean() - R.iloc[k][cur_short].mean()
        k += 1
    return gross[J:] - cost[J:]


def main():
    print("Fetching sector ETFs (free)...", flush=True)
    cl = {}
    for s in UNIVERSE:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
            if len(h) > 250:
                h.index = h.index.tz_localize(None).normalize()
                cl[s] = h
        except Exception:  # noqa: BLE001
            pass
    P = pd.concat(cl, axis=1)
    R = P.pct_change()
    half = len(R) // 2
    print(f"{len(P.columns)} sector ETFs, {len(P)} days, ~{round(len(P.columns)*N_SIDE_FRAC)} "
          f"long & short.\n", flush=True)

    print(f"{'formation':>10}{'net ann':>9}{'Sharpe':>8}{'1st':>6}{'2nd':>6}"
          f"{'maxDD':>8}{'worst21d':>10}")
    print("-" * 57)
    for J in FORMATIONS:
        net = run(P, R, J, HOLD)
        s = stats(net)
        s1 = stats(net[:half])["sharpe"]
        s2 = stats(net[half:])["sharpe"]
        lbl = {5: "1 week", 21: "1 month", 63: "1 quarter", 126: "6 months"}[J]
        print(f"{lbl:>10}{s['ann']*100:>8.1f}%{s['sharpe']:>8.2f}{s1:>6.2f}{s2:>6.2f}"
              f"{s['maxdd']*100:>7.0f}%{s['worst21']*100:>9.0f}%")

    # benchmark: equal-weight sector buy & hold
    bh = stats(R.mean(axis=1).dropna().to_numpy())
    print(f"\n  benchmark equal-weight sectors (long-only): Sharpe {bh['sharpe']:.2f}, "
          f"maxDD {bh['maxdd']*100:.0f}%")
    print("\nIf sector momentum Sharpe collapses vs the stock test (0.5-0.8), that result")
    print("was mostly survivorship. If it holds AND maxDD is tolerable, it's a real edge.")


if __name__ == "__main__":
    main()
