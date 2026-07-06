"""Full Quantum Liang Flow (IQLF/DIQLF, thesis eqs 3.5/3.6) on market data via a
LEARNED dissipative generator -- the research frontier from the thesis.

Pipeline:
  1. rho(t) = corr(rolling W)/N, sampled every STRIDE days  -> density-matrix path.
  2. Learn empirical Liouvillian L: vec(rho_dot) ~= L vec(rho) (least squares / DMD).
     Report one-step fit R^2 (is a Markovian-linear generator even adequate?).
     NOTE: L is NOT CP-constrained -> a Liouvillian approximation, not a certified
     GKSL Lindbladian. Honest first pass.
  3. Freeze bloc B:  L_{\B} = L with every row & column touching a B-index zeroed
     (asset-space analog of dropping B's Hamiltonian terms).
  4. IQLF (eq 3.5):  T_{B->A}(r) = [ S(rho_A^free(r+f)) - S(rho_A^{\B}(r+f)) ] / f.
     DIQLF (eq 3.6):  integral of T_{B->A}(r) over r  (reported as mean rate x span).
  5. Directed bloc flows EQ / CREDIT / REAL, with circular-shift surrogate p-values.

Model-dependent & exploratory. Free yfinance data.
Run:  python research/lindbladian_flow.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "EEM", "TLT", "HYG", "GLD", "USO", "UUP"]
BLOCS = {"EQ": ["SPY", "QQQ", "IWM", "XLF", "XLE", "EEM"],
         "CREDIT": ["TLT", "HYG"], "REAL": ["GLD", "USO", "UUP"]}
W, STRIDE, PERIOD = 60, 5, "12y"
N_SURR = 20
rng = np.random.default_rng(20250718)


def rho_path(arr):
    """Strided sequence of vec(rho), rho = corr(rolling W)/N."""
    N = arr.shape[1]
    out = []
    for k in range(W, len(arr), STRIDE):
        C = np.corrcoef(arr[k - W:k].T)
        out.append((C / N).flatten())
    return np.array(out).T            # (N^2, T)


def learn_L(V):
    """Liouvillian from vec(rho) path V (N^2 x T): vec(rho_dot) = L vec(rho)."""
    X, Y = V[:, :-1], V[:, 1:] - V[:, :-1]      # per-step difference
    L = Y @ np.linalg.pinv(X)
    pred = X + L @ X
    ss = ((V[:, 1:] - pred) ** 2).sum()
    r2 = 1 - ss / ((V[:, 1:] - V[:, 1:].mean()) ** 2).sum()
    return L, r2


def red_entropy(vrho, idxA, N):
    M = vrho.reshape(N, N)
    M = (M + M.T) / 2
    sub = M[np.ix_(idxA, idxA)]
    tr = np.trace(sub)
    if tr <= 1e-12:
        return 0.0
    ev = np.linalg.eigvalsh(sub / tr)
    ev = ev[ev > 1e-12]
    if ev.sum() <= 0:
        return 0.0
    ev = ev / ev.sum()
    return float(-np.sum(ev * np.log(ev)))


def freeze(L, Bidx, N):
    mask = np.zeros(N * N, bool)
    for p in range(N * N):
        if (p // N) in Bidx or (p % N) in Bidx:
            mask[p] = True
    Lf = L.copy()
    Lf[mask, :] = 0.0
    Lf[:, mask] = 0.0
    return Lf


def diqlf(V, L, Aidx, Bidx, N):
    """Mean IQLF rate T_{B->A} over the path (eq 3.5 integrated -> eq 3.6)."""
    Lf = freeze(L, Bidx, N)
    vals = []
    for r in range(V.shape[1]):
        v = V[:, r]
        free = v + L @ v
        frz = v + Lf @ v
        vals.append(red_entropy(free, Aidx, N) - red_entropy(frz, Aidx, N))
    return float(np.mean(vals))


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
    N = len(syms)
    idx = {s: i for i, s in enumerate(syms)}
    arr = R.to_numpy()
    bidx = {b: [idx[s] for s in names] for b, names in BLOCS.items()}

    V = rho_path(arr)
    L, fit = learn_L(V)
    print(f"{N} assets, {V.shape[1]} density-matrix samples (stride {STRIDE}d).")
    print(f"Learned Liouvillian one-step fit R^2 = {fit*100:.1f}%  "
          f"({'adequate Markov-linear model' if fit > 0.5 else 'crude — interpret cautiously'})\n",
          flush=True)

    names = list(BLOCS)
    obs = {}
    print("=" * 60)
    print("QUANTUM LIANG FLOW between blocs  T_{B->A}  (DIQLF rate, nats/step)")
    print("=" * 60)
    for a in names:
        for b in names:
            if a == b:
                continue
            obs[(b, a)] = diqlf(V, L, bidx[a], bidx[b], N)

    # surrogate null: independently circular-shift each asset's returns
    null = {k: [] for k in obs}
    for _ in range(N_SURR):
        sh = np.column_stack([np.roll(arr[:, j], rng.integers(1, len(arr) - 1))
                              for j in range(N)])
        Vs = rho_path(sh)
        Ls, _ = learn_L(Vs)
        for (b, a) in obs:
            null[(b, a)].append(diqlf(Vs, Ls, bidx[a], bidx[b], N))

    print(f"  {'flow B->A':22}{'T (rate)':>13}{'p':>8}{'effect':>10}")
    for (b, a), T in sorted(obs.items(), key=lambda kv: -abs(kv[1])):
        nul = np.abs(null[(b, a)])
        p = (np.sum(nul >= abs(T)) + 1) / (N_SURR + 1)
        eff = "B adds unc" if T > 0 else "B reduces unc"
        star = "*" if p < 0.05 else " "
        print(f"  {b+' -> '+a:22}{T:>13.2e}{p:>7.2f}{star}{eff:>10}")

    print("\nAsymmetry (net direction) for each bloc pair:")
    seen = set()
    for a in names:
        for b in names:
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b)); seen.add((b, a))
            net = obs[(b, a)] - obs[(a, b)]
            lead = b if net > 0 else a
            print(f"  {a} <-> {b}: net {net:+.2e}  -> {lead} drives")
    print("\nFirst full IQLF/DIQLF on market data. Model-dependent (learned non-CP")
    print("Liouvillian, asset-space freezing). Compare to: classical-return Liang found")
    print("equity->HYG; scalar-entropy-proxy bloc flow was null.")


if __name__ == "__main__":
    main()
