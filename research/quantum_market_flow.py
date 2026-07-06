"""Market-as-quantum-state: density matrix, von Neumann entropy dynamics, and a
first-pass collective (multipartite) information flow between asset blocs.

Construction (honest about the mapping):
  rho(t) = C(t)/Tr(C(t))  where C(t) = rolling CORRELATION matrix of returns.
  Correlation matrices are PSD with Tr = N, so rho is a valid density matrix.
  NOTE: rho lives in ASSET space, not a tensor-product Hilbert space, so a bloc's
  'reduced state' is the renormalized SUB-BLOCK (a marginal), NOT a tensor partial
  trace. This differs from the qubit setting in the thesis and is flagged as such.

Quantities:
  S(rho)            von Neumann entropy = market disorder (low = everything
                    correlated = systemic stress; high = diversified/calm).
  I(A:B)            quantum mutual information S(rhoA)+S(rhoB)-S(rhoAB) between
                    blocs = collective shared structure (needs no generator).
  T(S_A -> S_B)     classical Liang-Kleeman flow between bloc-entropy TIME SERIES
                    = directed 'disorder flow' (rigorous; sidesteps the Lindbladian
                    needed for the full IQLF/DIQLF, which is the next step).

Free yfinance data. Run:  python research/quantum_market_flow.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "EEM", "TLT", "HYG", "GLD", "USO", "UUP"]
BLOCS = {"EQ": ["SPY", "QQQ", "IWM", "XLF", "XLE", "EEM"],
         "CREDIT/RATES": ["TLT", "HYG"],
         "REAL/USD": ["GLD", "USO", "UUP"]}
WINDOW = 60
PERIOD = "12y"
rng = np.random.default_rng(20250718)


def vn_entropy(corr):
    """von Neumann entropy of rho = corr/trace (nats)."""
    rho = corr / np.trace(corr)
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)))


def liang_T(x1, x2):
    """Classical Liang-Kleeman info flow x2 -> x1 (Liang 2014)."""
    dx1 = x1[1:] - x1[:-1]
    a, b = x1[:-1], x2[:-1]
    C = np.cov(np.vstack([a, b]))
    C11, C22, C12 = C[0, 0], C[1, 1], C[0, 1]
    denom = C11 ** 2 * C22 - C11 * C12 ** 2
    if abs(denom) < 1e-300:
        return 0.0
    return (C11 * C12 * np.cov(b, dx1)[0, 1] - C12 ** 2 * np.cov(a, dx1)[0, 1]) / denom


def liang_p(x1, x2, n_surr=300):
    obs = liang_T(x1, x2)
    n = len(x2)
    null = [abs(liang_T(x1, np.roll(x2, rng.integers(1, n - 1)))) for _ in range(n_surr)]
    return obs, (np.sum(np.array(null) >= abs(obs)) + 1) / (n_surr + 1)


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
    idx = {s: i for i, s in enumerate(syms)}
    print(f"{len(syms)} assets, {len(R)} days, window {WINDOW}d.\n", flush=True)

    # --- rolling density matrix -> von Neumann entropy (whole market + blocs) ---
    dates, S_all, S_bloc = [], [], {b: [] for b in BLOCS}
    MI = {("EQ", "CREDIT/RATES"): [], ("EQ", "REAL/USD"): [], ("CREDIT/RATES", "REAL/USD"): []}
    for k in range(WINDOW, len(R)):
        win = R.iloc[k - WINDOW:k]
        Cfull = np.corrcoef(win.to_numpy().T)
        dates.append(R.index[k])
        S_all.append(vn_entropy(Cfull))
        sub = {}
        for b, names in BLOCS.items():
            ii = [idx[s] for s in names]
            sub[b] = vn_entropy(Cfull[np.ix_(ii, ii)])
            S_bloc[b].append(sub[b])
        for (a, c) in MI:
            ia = [idx[s] for s in BLOCS[a]]
            ic = [idx[s] for s in BLOCS[c]]
            iab = ia + ic
            mi = sub[a] + sub[c] - vn_entropy(Cfull[np.ix_(iab, iab)])
            MI[(a, c)].append(mi)

    S_all = np.array(S_all)
    dates = pd.DatetimeIndex(dates)
    nmax = np.log(len(syms))
    print("=" * 64)
    print("MARKET von NEumann ENTROPY  S(rho)  (max = ln N = %.2f)" % nmax)
    print("=" * 64)
    print(f"  mean {S_all.mean():.3f}  range {S_all.min():.3f}..{S_all.max():.3f}")
    lows = dates[np.argsort(S_all)[:4]]
    highs = dates[np.argsort(S_all)[-3:]]
    print(f"  LOWEST entropy (most systemic stress): {[d.date().isoformat() for d in lows]}")
    print(f"  HIGHEST entropy (most diversified):    {[d.date().isoformat() for d in highs]}")
    # validation: entropy should fall when average pairwise correlation rises
    print("  (low entropy should line up with known crises -> sanity check)\n")

    print("QUANTUM MUTUAL INFORMATION between blocs  I(A:B) = S_A+S_B-S_AB:")
    for (a, c), series in MI.items():
        s = np.array(series)
        print(f"  I({a} : {c}): mean {s.mean():.3f}, peak {s.max():.3f} "
              f"on {dates[int(np.argmax(s))].date()}")
    print()

    print("=" * 64)
    print("DIRECTED 'disorder flow' between bloc entropies (classical Liang on S_X(t)):")
    print("=" * 64)
    names = list(BLOCS)
    for a in names:
        for c in names:
            if a == c:
                continue
            T, p = liang_p(np.array(S_bloc[c]), np.array(S_bloc[a]))  # a -> c
            tag = "adds" if T > 0 else "removes"
            sig = "*" if p < 0.05 else " "
            print(f"  {a:13} -> {c:13} T={T:+.3e}  p={p:.3f}{sig}  ({tag} uncertainty)")
    print("\n  (T_{A->B} != T_{B->A}: the asymmetry shows which bloc's disorder leads.)")
    print("\nFirst-pass: rho(t), S(rho) dynamics, bloc mutual information, and directed")
    print("entropy-flow all computed on real data. Full IQLF/DIQLF (eqs 3.5/3.6) needs")
    print("a Lindbladian generator + freezing for the dissipative dynamics — next step.")


if __name__ == "__main__":
    main()
