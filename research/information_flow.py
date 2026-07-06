"""Directed information flow (Granger causality) across a cross-asset basket,
with the two safeguards that separate signal from noise:
  1. False-discovery-rate (Benjamini-Hochberg) correction across ALL pairs.
  2. Sub-period stability: a link must hold in the 1st AND 2nd half of history.

Granger 'X causes Y' = X's past return helps predict Y's next return beyond Y's
own past. Tests daily returns (the tradable horizon). A link that survives both
safeguards is a candidate lead-lag edge; everything else is noise.

Free yfinance daily data. Run:  python research/information_flow.py
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# Cross-asset basket with economic priors for plausible lead-lag.
ASSETS = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "USO", "UUP",
          "XLE", "XLF", "EEM", "BTC-USD"]
PERIOD = "12y"
LAG = 1          # 1-day lead-lag (the tradable horizon)
FDR_Q = 0.05


def granger(y, x, p=LAG):
    """p-value that x's past Granger-causes y; plus sign of the lag-1 coefficient
    and the extra R^2 it explains."""
    n = len(y)
    T = n - p
    Y = y[p:]
    cols_r = [np.ones(T)] + [y[p - k:n - k] for k in range(1, p + 1)]
    Xr = np.column_stack(cols_r)
    Xu = np.column_stack(cols_r + [x[p - k:n - k] for k in range(1, p + 1)])

    def rss(X):
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        return float(res @ res), beta
    rss_r, _ = rss(Xr)
    rss_u, beta_u = rss(Xu)
    df2 = T - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return 1.0, 0.0, 0.0
    F = ((rss_r - rss_u) / p) / (rss_u / df2)
    pval = float(stats.f.sf(F, p, df2))
    tss = float(((Y - Y.mean()) ** 2).sum())
    return pval, float(np.sign(beta_u[Xr.shape[1]])), (rss_r - rss_u) / tss if tss else 0.0


def bh_fdr(pvals, q=FDR_Q):
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(m, bool)
    cut = p[order][np.where(passed)[0].max()]
    return p <= cut


def main():
    print("Fetching daily data (free)...", flush=True)
    cl = {}
    for s in ASSETS:
        try:
            h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
            if len(h) > 500:
                h.index = h.index.tz_localize(None).normalize()
                cl[s] = h
        except Exception:  # noqa: BLE001
            pass
    R = pd.concat(cl, axis=1).pct_change().dropna()
    print(f"{len(R.columns)} assets, {len(R)} common days "
          f"({R.index[0].date()}..{R.index[-1].date()})\n", flush=True)

    syms = list(R.columns)
    half = len(R) // 2
    pairs, pfull = [], []
    detail = {}
    for xi in syms:
        for yi in syms:
            if xi == yi:
                continue
            pv, sg, dr2 = granger(R[yi].to_numpy(), R[xi].to_numpy())
            p1, s1, _ = granger(R[yi].to_numpy()[:half], R[xi].to_numpy()[:half])
            p2, s2, _ = granger(R[yi].to_numpy()[half:], R[xi].to_numpy()[half:])
            pairs.append((xi, yi))
            pfull.append(pv)
            detail[(xi, yi)] = dict(p=pv, sign=sg, dr2=dr2, p1=p1, p2=p2,
                                    stable=(p1 < 0.05 and p2 < 0.05 and s1 == s2 == sg))

    sig = bh_fdr(pfull)
    n_sig = int(sig.sum())
    print(f"{len(pairs)} ordered pairs tested. {n_sig} pass FDR<{FDR_Q} "
          f"(noise-corrected).", flush=True)

    survivors = []
    print("\nFDR-significant directed links (X -> Y):")
    print(f"  {'link':18}{'p(full)':>10}{'extraR2':>9}{'dir':>5}{'both halves?':>14}")
    for (xi, yi), is_sig in sorted(zip(pairs, sig), key=lambda z: detail[z[0]]["p"]):
        if not is_sig:
            continue
        d = detail[(xi, yi)]
        arrow = "same" if d["sign"] > 0 else "oppos"
        ok = "STABLE" if d["stable"] else "unstable"
        if d["stable"]:
            survivors.append((xi, yi, d))
        print(f"  {xi+' -> '+yi:18}{d['p']:>10.1e}{d['dr2']*100:>8.2f}%{arrow:>5}{ok:>14}")

    print("\n" + "=" * 60)
    if survivors:
        print(f"{len(survivors)} link(s) survived BOTH FDR and sub-period stability:")
        for xi, yi, d in survivors:
            direction = "same-direction" if d["sign"] > 0 else "reversal"
            print(f"  {xi} leads {yi} ({direction}), extra R^2 {d['dr2']*100:.2f}% "
                  f"-> next: test if it trades net of costs")
    else:
        print("NO link survived both noise-correction AND sub-period stability.")
        print("The apparent information flows are noise or unstable -> no daily edge here.")
    print(f"\nEffect sizes (extra R^2) are the key: even 'real' links explaining <<1% of")
    print("next-day variance rarely beat costs. Linear Granger only; nonlinear transfer")
    print("entropy is the next tool if a stable linear link appears.")


if __name__ == "__main__":
    main()
