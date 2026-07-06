"""Test the 'patterns are LOCAL not year-stable' hypothesis via walk-forward.

At each day t, fit a small adaptive ridge model using ONLY the prior L days
(features = lag-1 returns of a few related assets), predict day t's return, take
position = sign(prediction), realize P/L net of cost. Roll forward. Every
prediction is genuinely out-of-sample vs its own training window, so a positive
net aggregate is real -- this is how adaptive/online systems actually work.

Tests several lookbacks L (a week, two weeks, a month). Reports net Sharpe and
sub-period stability. Honest danger: short L overfits noise; costs from daily
trading bite. Free yfinance data.

Run:  python research/adaptive_walkforward.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

FEATURES = ["SPY", "QQQ", "IWM", "HYG", "TLT"]   # lag-1 predictors
TARGETS = ["SPY", "QQQ", "IWM", "HYG"]
LOOKBACKS = [5, 10, 21]
PERIOD = "12y"
COST = 0.00015      # ~1.5 bps per position flip (liquid ETFs)
ALPHA = 1e-3        # ridge regularization


def ann(d):
    d = np.asarray(d)
    vol = d.std() * np.sqrt(252)
    return ((1 + d).prod() ** (252 / len(d)) - 1 if len(d) else np.nan,
            (d.mean() * 252) / vol if vol > 0 else np.nan)


def ridge_predict(Xtr, ytr, xnew, alpha=ALPHA):
    Xc = np.column_stack([np.ones(len(Xtr)), Xtr])
    A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    beta = np.linalg.solve(A, Xc.T @ ytr)
    return float(np.array([1.0, *xnew]) @ beta)


def main():
    print("Fetching daily data (free)...", flush=True)
    cl = {}
    for s in set(FEATURES + TARGETS):
        h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
        h.index = h.index.tz_localize(None).normalize()
        cl[s] = h
    R = pd.concat(cl, axis=1).pct_change().dropna()
    feat = R[FEATURES].to_numpy()
    n = len(R)
    half = n // 2
    print(f"{n} days. Walk-forward adaptive ridge, lag-1 features {FEATURES}.\n", flush=True)

    print(f"{'target':8}{'L':>4}{'net ann%':>10}{'net Sharpe':>12}"
          f"{'1st-half Sh':>13}{'2nd-half Sh':>13}{'trades/yr':>11}")
    print("-" * 71)
    pooled = {L: [] for L in LOOKBACKS}
    for tgt in TARGETS:
        y = R[tgt].to_numpy()
        ti = FEATURES.index(tgt) if tgt in FEATURES else None
        for L in LOOKBACKS:
            pos = np.zeros(n)
            net = np.zeros(n)
            for t in range(L + 1, n):
                Xtr = feat[t - 1 - L:t - 1]        # features at days predicting...
                ytr = y[t - L:t]                   # ...returns shifted +1
                pred = ridge_predict(Xtr, ytr, feat[t - 1])
                p = np.sign(pred)
                pos[t] = p
                flip = abs(p - pos[t - 1]) > 0
                net[t] = p * y[t] - (COST if flip else 0)
            seg = net[L + 1:]
            a, s = ann(seg)
            _, s1 = ann(net[L + 1:half])
            _, s2 = ann(net[half:])
            tpy = np.mean(np.abs(np.diff(pos[L + 1:])) > 0) * 252
            print(f"{tgt:8}{L:>4}{a*100:>9.1f}%{s:>12.2f}{s1:>13.2f}{s2:>13.2f}{tpy:>11.0f}")
            pooled[L].append(seg[-min(len(seg), n - 1 - 21):])
        print()

    print("Equal-weight portfolio across all targets (net):")
    print(f"  {'L':>4}{'net ann%':>10}{'net Sharpe':>12}")
    for L in LOOKBACKS:
        m = min(len(x) for x in pooled[L])
        port = np.mean([x[-m:] for x in pooled[L]], axis=0)
        a, s = ann(port)
        print(f"  {L:>4}{a*100:>9.1f}%{s:>12.2f}")
    print("\nVerdict: a short lookback wins only if NET Sharpe is solidly positive AND")
    print("holds in both halves. If net is <=0 / unstable, the 'local pattern' was noise")
    print("+ costs. (Adaptive retraining is genuinely OOS here -- no lookahead.)")


if __name__ == "__main__":
    main()
