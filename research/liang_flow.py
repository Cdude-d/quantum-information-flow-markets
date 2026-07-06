"""Classical Liang-Kleeman Information Flow (Liang 2014 bivariate estimator)
applied to cross-asset returns — a rigorous upgrade over Granger causality.

LKIF property: T_{2->1} = 0  iff  x1 is independent of x2 (necessary AND
sufficient for causality), asymmetric, with a SIGN (T>0: source adds uncertainty
to target; T<0: source makes target more certain). Liang 2014, Phys Rev E 90.

Estimator (per ordered pair x2 -> x1), with dx1 the Euler-forward difference:
  T_{2->1} = (C11*C12*C_2,d1 - C12^2 * C_1,d1) / (C11^2*C22 - C11*C12^2)

Significance: circular-shift surrogates (preserve each series' own autocorrelation,
destroy the cross-timing) -> p-value. Then FDR across pairs + both-halves stability,
exactly as the Granger run, for a like-for-like comparison.

Free yfinance daily data. Run:  python research/liang_flow.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "HYG", "GLD", "USO", "UUP",
          "XLE", "XLF", "EEM", "BTC-USD"]
PERIOD = "12y"
N_SURR = 300
FDR_Q = 0.05
rng = np.random.default_rng(20250718)


def liang_T(x1, x2):
    """Info flow rate x2 -> x1 (nats/step). dx1 = forward difference."""
    dx1 = x1[1:] - x1[:-1]
    a, b = x1[:-1], x2[:-1]
    C = np.cov(np.vstack([a, b]))
    C11, C22, C12 = C[0, 0], C[1, 1], C[0, 1]
    C_2d1 = np.cov(b, dx1)[0, 1]
    C_1d1 = np.cov(a, dx1)[0, 1]
    denom = C11 ** 2 * C22 - C11 * C12 ** 2
    if abs(denom) < 1e-300:
        return 0.0
    return (C11 * C12 * C_2d1 - C12 ** 2 * C_1d1) / denom


def pval(x1, x2):
    """Two-sided p via circular-shift surrogates of x2."""
    obs = liang_T(x1, x2)
    n = len(x2)
    null = np.empty(N_SURR)
    for i in range(N_SURR):
        s = rng.integers(1, n - 1)
        null[i] = liang_T(x1, np.roll(x2, s))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_SURR + 1)
    return obs, p


def bh_fdr(p, q=FDR_Q):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    if not passed.any():
        return np.zeros(m, bool)
    return p <= p[order][np.where(passed)[0].max()]


def main():
    print("Fetching daily data (free)...", flush=True)
    cl = {}
    for s in ASSETS:
        h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
        if len(h) > 500:
            h.index = h.index.tz_localize(None).normalize()
            cl[s] = h
    R = pd.concat(cl, axis=1).pct_change().dropna()
    syms = list(R.columns)
    half = len(R) // 2
    print(f"{len(syms)} assets, {len(R)} days. Liang-Kleeman IF, {N_SURR} surrogates/pair.\n",
          flush=True)

    pairs, ps, detail = [], [], {}
    for xi in syms:           # source (x2)
        for yi in syms:       # target (x1)
            if xi == yi:
                continue
            x1f, x2f = R[yi].to_numpy(), R[xi].to_numpy()
            T, p = pval(x1f, x2f)
            T1 = liang_T(x1f[:half], x2f[:half])
            T2 = liang_T(x1f[half:], x2f[half:])
            pairs.append((xi, yi))
            ps.append(p)
            detail[(xi, yi)] = dict(T=T, p=p, stable=(np.sign(T1) == np.sign(T2) == np.sign(T)
                                                      and abs(T1) > 1e-9 and abs(T2) > 1e-9))

    sig = bh_fdr(ps)
    print(f"{len(pairs)} ordered pairs. {int(sig.sum())} pass FDR<{FDR_Q}.\n")
    print(f"  {'link (src->tgt)':20}{'T (info flow)':>15}{'p':>9}{'sign':>7}{'stable':>9}")
    survivors = []
    for (xi, yi), is_sig in sorted(zip(pairs, sig), key=lambda z: detail[z[0]]["p"]):
        if not is_sig:
            continue
        d = detail[(xi, yi)]
        eff = "+unc" if d["T"] > 0 else "-unc"
        st = "STABLE" if d["stable"] else "unstable"
        if d["stable"]:
            survivors.append((xi, yi, d))
        print(f"  {xi+' -> '+yi:20}{d['T']:>15.2e}{d['p']:>9.3f}{eff:>7}{st:>9}")

    print("\n" + "=" * 60)
    if survivors:
        print(f"{len(survivors)} link(s) survived FDR + both-halves sign-stability:")
        for xi, yi, d in survivors:
            print(f"  {xi} -> {yi}: T={d['T']:.2e} "
                  f"({'adds' if d['T']>0 else 'removes'} uncertainty)")
    else:
        print("No link survived both. (LKIF is stricter than Granger — informative either way.)")
    print("\nCompare to Granger: did QQQ->HYG / SPY->HYG show up under a true causality test?")


if __name__ == "__main__":
    main()
