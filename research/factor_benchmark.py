"""Classical benchmark the quantum model must beat. Same inputs (within-month factor
percentile ranks), same target (beat = next-month return above cross-sectional median),
same split: discovery 1963-07..2004-12, held-out 2005-01..2024-11.

  Part A: per-factor value-weighted top-minus-bottom decile portfolios (high-minus-low,
          so the sign shows direction), Newey-West t-stats, BH-FDR on discovery,
          confirmation on held-out. All stocks and large caps (>= NYSE median ME).
  Part B: logistic regression on all 9 ranks, fit on discovery. Held-out accuracy, AUC,
          and a VW decile long-short on predicted probability, gross and net of a
          turnover cost assumption, plus alpha vs FF5 + momentum.

Run:  python research/factor_benchmark.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from factor_panel import FACTORS, OUT
from wrds_lib import q

DISC_END, HOLD_START = 2004 * 12 + 12, 2005 * 12 + 1
COST = {"all": 0.0030, "large": 0.0010}      # one-way cost per unit of weight traded
FDR_Q = 0.05


def nw_t(x, lags=6):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, m = len(x), x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return m / np.sqrt(s / n)


def ols_alpha(y, X, lags=6):
    X = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None]) / n
    for k in range(1, lags + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / n
        S += (1 - k / (lags + 1)) * (G + G.T)
    V = n * XtX_inv @ S @ XtX_inv
    return b[0], b[0] / np.sqrt(V[0, 0])


def bh(p, q=FDR_Q):
    p = np.asarray(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, len(p) + 1) / len(p)
    return p <= (p[o][np.where(ok)[0].max()] if ok.any() else -1)


def long_short(d, score, cost):
    """VW decile long-short on `score` (top minus bottom), monthly. Returns gross, net."""
    d = d.assign(dec=np.minimum((d.groupby("ym")[score].rank(pct=True) * 10).astype(int), 9))
    gross, net, months = [], [], []
    prev = {}
    for ym, g in d.groupby("ym", sort=True):
        legs, ret, trade = {}, 0.0, 0.0
        for side, dec, sign in (("L", 9, 1), ("S", 0, -1)):
            x = g[g.dec == dec]
            if x.empty:
                break
            w = x.me / x.me.sum()
            legs[side] = dict(zip(x.permno, w))
            ret += sign * (w * x.ret_next).sum()
            old = prev.get(side, {})
            keys = set(old) | set(legs[side])
            trade += sum(abs(legs[side].get(k, 0) - old.get(k, 0)) for k in keys)
        else:
            gross.append(ret)
            net.append(ret - trade * cost)
            months.append(ym)
            prev = legs
    return pd.Series(gross, index=months), pd.Series(net, index=months)


def ann(s):
    return s.mean() * 12, s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else np.nan


def logistic_fit(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        step = np.linalg.solve((X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    npos = y.sum()
    nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    p = pd.read_pickle(OUT / "panel.pkl")
    p = p.dropna(subset=FACTORS)
    p = p[p.ym >= 1963 * 12 + 7]
    ff = q("SELECT date, mktrf, smb, hml, rmw, cma, umd FROM ff.fivefactors_monthly")
    ff["date"] = pd.to_datetime(ff["date"])
    ff["ym"] = ff.date.dt.year * 12 + ff.date.dt.month
    ff = ff.set_index("ym")
    print(f"{len(p):,} stock-months with all 9 factors "
          f"({p.date.min():%Y-%m}..{p.date.max():%Y-%m})\n", flush=True)

    for uni in ("all", "large"):
        d = p if uni == "all" else p[p.large]
        d = d.copy()
        for f in FACTORS:
            d[f"r_{f}"] = d.groupby("ym")[f].rank(pct=True)
        disc, hold = d[d.ym <= DISC_END], d[d.ym >= HOLD_START]
        label = "ALL STOCKS" if uni == "all" else "LARGE CAPS (>= NYSE median size)"

        print("=" * 84)
        print(f"PART A — single-factor high-minus-low decile portfolios, value-weighted: {label}")
        print("=" * 84)
        print(f"  {'factor':7}{'disc ann%':>10}{'disc t':>8}{'FDR':>5}"
              f"{'held ann%':>11}{'held t':>8}{'confirmed':>11}{'held Sharpe':>13}")
        res = {}
        for f in FACTORS:
            gd, _ = long_short(disc, f"r_{f}", 0)
            gh, _ = long_short(hold, f"r_{f}", 0)
            res[f] = (gd, gh)
        tvals = {f: nw_t(res[f][0]) for f in FACTORS}
        from scipy.stats import norm
        pv = [2 * (1 - norm.cdf(abs(tvals[f]))) for f in FACTORS]
        passed = dict(zip(FACTORS, bh(pv)))
        for f in FACTORS:
            gd, gh = res[f]
            th = nw_t(gh)
            conf = passed[f] and np.sign(th) == np.sign(tvals[f]) and abs(th) > 1.96
            print(f"  {f:7}{gd.mean()*1200:>9.1f}%{tvals[f]:>8.2f}{'Y' if passed[f] else '-':>5}"
                  f"{gh.mean()*1200:>10.1f}%{th:>8.2f}{'YES' if conf else 'no':>11}"
                  f"{ann(gh)[1]*np.sign(gh.mean()):>13.2f}")

        print(f"\nPART B — logistic regression on all 9 factor ranks, fit on 1963-2004: {label}")
        cols = [f"r_{f}" for f in FACTORS]
        Xd = np.column_stack([np.ones(len(disc)), disc[cols].to_numpy() - 0.5])
        b = logistic_fit(Xd, disc.beat.to_numpy().astype(float))
        Xh = np.column_stack([np.ones(len(hold)), hold[cols].to_numpy() - 0.5])
        prob = 1 / (1 + np.exp(-(Xh @ b)))
        yh = hold.beat.to_numpy()
        print("  coefficients (rank - 0.5):",
              {f: round(c, 3) for f, c in zip(FACTORS, b[1:])})
        print(f"  held-out accuracy {np.mean((prob > 0.5) == yh)*100:.2f}%  "
              f"(50% = coin flip)   AUC {auc(prob, yh):.4f}  (0.5 = no skill)")
        hold = hold.assign(prob=prob)
        gross, net = long_short(hold, "prob", COST[uni])
        ag, sg = ann(gross)
        an, sn = ann(net)
        fx = ff.loc[gross.index + 1, ["mktrf", "smb", "hml", "rmw", "cma", "umd"]].to_numpy()
        a, ta = ols_alpha(gross.to_numpy(), fx)
        print(f"  decile long-short on predicted prob: gross {ag*100:+.1f}%/yr (Sharpe {sg:.2f}, "
              f"t {nw_t(gross):.2f}) | net of {COST[uni]*1e4:.0f}bp/side: {an*100:+.1f}%/yr "
              f"(Sharpe {sn:.2f})")
        print(f"  alpha vs FF5+momentum: {a*1200:+.1f}%/yr (t {ta:.2f})\n")
    print("Quantum model target: beat these held-out AUC / long-short numbers with the SAME inputs.")


if __name__ == "__main__":
    main()
