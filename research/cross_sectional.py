"""Cross-sectional (market-neutral) test: rank a stock universe by trailing-J
return, go long top quintile / short bottom quintile, hold K days, roll. This is
structurally different from everything prior -- it's RELATIVE, so it doesn't care
where the whole market goes, only relative performance.

Tests several formation windows J (week / month / quarter / half-year) per the
user's idea. Reports MOMENTUM (long winners) and REVERSAL (long losers) net of
costs, with sub-period stability. Free yfinance data.

Run:  python research/cross_sectional.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AMD", "CRM", "ORCL",
            "CSCO", "ADBE", "JPM", "BAC", "GS", "MS", "WFC", "XOM", "CVX", "COP",
            "JNJ", "PFE", "UNH", "MRK", "ABBV", "WMT", "HD", "KO", "PG", "MCD",
            "NKE", "DIS", "CAT", "BA", "HON"]
FORMATIONS = [5, 21, 63, 126]
HOLD = 21
PERIOD = "12y"
COST = 0.0003          # ~3 bps per name per side (liquid large-caps)
QUINTILE = 0.2


def ann(d):
    d = np.asarray(d)
    d = d[~np.isnan(d)]
    if len(d) < 30:
        return np.nan, np.nan
    vol = d.std() * np.sqrt(252)
    return ((1 + d).prod() ** (252 / len(d)) - 1, (d.mean() * 252) / vol if vol > 0 else np.nan)


def run(prices, R, J, K):
    days = R.index
    n = len(days)
    syms = list(R.columns)
    gross = np.zeros(n)
    cost = np.zeros(n)
    long_prev, short_prev = set(), set()
    cur_long, cur_short = [], []
    k = max(J, 1)
    while k < n:
        if (k - J) % K == 0 or not cur_long:    # rebalance
            past = prices.iloc[k] / prices.iloc[k - J] - 1.0
            past = past.dropna()
            if len(past) >= 10:
                ranked = past.sort_values()
                m = max(1, int(len(ranked) * QUINTILE))
                cur_short = list(ranked.index[:m])     # losers
                cur_long = list(ranked.index[-m:])     # winners
                turn = len(set(cur_long) ^ long_prev) + len(set(cur_short) ^ short_prev)
                cost[k] = turn / (2 * m) * COST
                long_prev, short_prev = set(cur_long), set(cur_short)
        if cur_long:
            rl = R.iloc[k][cur_long].mean()
            rs = R.iloc[k][cur_short].mean()
            gross[k] = rl - rs                          # momentum: long winners - short losers
        k += 1
    return gross[J:], cost[J:]


def main():
    print("Fetching daily data (free)...", flush=True)
    cl = {}
    for s in UNIVERSE:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
            if len(h) > 500:
                h.index = h.index.tz_localize(None).normalize()
                cl[s] = h
        except Exception:  # noqa: BLE001
            pass
    P = pd.concat(cl, axis=1).dropna()
    R = P.pct_change().fillna(0)
    half = len(R) // 2
    print(f"{len(P.columns)} stocks, {len(P)} days. Quintile long-short, hold {HOLD}d.\n",
          flush=True)

    print(f"{'formation':>10}{'  MOMENTUM: net ann / Sharpe / 1st / 2nd':>42}"
          f"{'  REVERSAL: Sharpe':>20}")
    print("-" * 72)
    for J in FORMATIONS:
        g, c = run(P, R, J, HOLD)
        mom = g - c
        rev = -g - c
        a_m, s_m = ann(mom)
        _, s1 = ann(mom[:half])
        _, s2 = ann(mom[half:])
        _, s_r = ann(rev)
        label = {5: "1 week", 21: "1 month", 63: "1 quarter", 126: "6 months"}[J]
        print(f"{label:>10}{a_m*100:>14.1f}%{s_m:>9.2f}{s1:>8.2f}{s2:>8.2f}"
              f"{s_r:>20.2f}")

    print("\nMOMENTUM = long recent winners; REVERSAL = long recent losers (= -momentum gross,")
    print("but both pay cost). A real cross-sectional edge = net Sharpe solidly >0 AND stable")
    print("in both halves at some formation horizon. Short-horizon usually reversal, medium")
    print("usually momentum -- if any survives costs.")


if __name__ == "__main__":
    main()
