"""Quantum-information factor model: each stock-month is a 10-qubit state.

  Encoding  9 factor qubits, each rotated by the stock's within-month percentile u on that
            factor: |q> = cos(pi*u/2)|0> + sin(pi*u/2)|1>; plus a return qubit R in |0>.
  Dynamics  learned Hamiltonian, 64 terms: Z_k Y_R, X_k Y_R (factor -> return couplings),
            X_k (transverse fields, non-commuting), Z_k Z_j (factor interactions), Y_R (bias).
            Without the non-commuting terms the model collapses to a classical mixture.
  Readout   Born rule: P(R = |1>) after U = exp(-iH) = P(stock beats the cross-sectional
            median next month). Fit by maximum likelihood on 1963-2004 only.
  Test      held-out 2005-2024 vs the classical logistic benchmark on identical inputs.
  Explain   IQLF (thesis eq. 3.5) from each factor qubit k to R: freeze k by dropping every
            Hamiltonian term that touches it, T_{k->R}(r) = d/df [S(rho_R^free(r+f))
            - S(rho_R^{frozen k}(r+f))] at f -> 0, integrated over r in [0,1] (DIQLF, eq. 3.6).
            Run on the 1963-2004 model and on a 2005-2024 model (descriptive, in-sample) to
            show how each factor's information channel changed after publication.

This is a quantum-INSPIRED model simulated classically; it does not claim stocks are quantum.
Run:  python research/quantum_factor_model.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import expm

from factor_benchmark import (COST, DISC_END, HOLD_START, ann, auc, logistic_fit,
                              long_short, nw_t, ols_alpha)
from factor_panel import FACTORS, OUT
from wrds_lib import q

NQ = len(FACTORS) + 1           # 9 factor qubits + return qubit
R = NQ - 1                      # return qubit index (least significant bit)
DIM = 2 ** NQ
STEPS, BATCH, LR, NTRAIN = 300, 16384, 0.03, 200_000
IQLF_STATES, IQLF_R, IQLF_EPS = 1500, 20, 1e-3
torch.manual_seed(20261104)
rng = np.random.default_rng(20261104)


# ------------------------------------------------------------------ Hamiltonian terms
def pauli_terms():
    t = [({k: "Z", R: "Y"}, f"Z{k}Y_R") for k in range(R)]
    t += [({k: "X", R: "Y"}, f"X{k}Y_R") for k in range(R)]
    t += [({k: "X"}, f"X{k}") for k in range(R)]
    t += [({k: "Z", j: "Z"}, f"Z{k}Z{j}") for k in range(R) for j in range(k + 1, R)]
    t += [({R: "Y"}, "Y_R")]
    return t


def sparse_pauli(ops):
    """(rows, cols, values) of a Pauli string; qubit 0 is the most significant bit."""
    b = np.arange(DIM)
    mask, val = 0, np.ones(DIM, complex)
    for qb, op in ops.items():
        bit = (b >> (NQ - 1 - qb)) & 1
        if op in "XY":
            mask |= 1 << (NQ - 1 - qb)
        if op == "Z":
            val = val * (-1.0) ** bit
        if op == "Y":                       # Y|0> = i|1>, Y|1> = -i|0>
            val = val * 1j * (-1.0) ** bit
    return b ^ mask, b, val


TERMS = pauli_terms()
SP = [sparse_pauli(o) for o, _ in TERMS]
ROWS = torch.tensor(np.concatenate([s[0] for s in SP]))
COLS = torch.tensor(np.concatenate([s[1] for s in SP]))
VALS = torch.tensor(np.concatenate([s[2] for s in SP]), dtype=torch.complex64)
TERM_ID = torch.tensor(np.repeat(np.arange(len(TERMS)), DIM))
PROJ = torch.zeros(DIM, dtype=torch.complex64)
PROJ[1::2] = 1                                  # R = |1> on odd basis indices
PROJ_NP = PROJ.numpy().astype(complex)


def readout_torch(theta):
    v = VALS * theta[TERM_ID].to(torch.complex64)
    H = torch.zeros(DIM, DIM, dtype=torch.complex64).index_put((ROWS, COLS), v, accumulate=True)
    U = torch.linalg.matrix_exp(-1j * H)
    M = U.conj().T @ (PROJ[:, None] * U)
    return M[0::2, 0::2].real                   # inputs live in the R = |0> subspace


def h_numpy(theta, drop=None):
    H = np.zeros((DIM, DIM), complex)
    for (ops, _), (r_, c_, v_), th in zip(TERMS, SP, theta):
        if drop is None or drop not in ops:
            H[r_, c_] += th * v_
    return H


def encode(u):
    """Percentile ranks (n, 9) -> product-state amplitudes (n, 512) in the R=|0> subspace."""
    c, s = np.cos(np.pi * u / 2), np.sin(np.pi * u / 2)
    phi = np.ones((len(u), 1), np.float32)
    for k in range(u.shape[1]):
        qk = np.stack([c[:, k], s[:, k]], 1).astype(np.float32)
        phi = (phi[:, :, None] * qk[:, None, :]).reshape(len(u), -1)
    return phi


# ------------------------------------------------------------------ fit / predict
def fit(u, y, label):
    idx = rng.choice(len(u), min(NTRAIN, len(u)), replace=False)
    phi = torch.tensor(encode(u[idx]))
    yy = torch.tensor(y[idx], dtype=torch.float32)
    theta = 0.02 * torch.randn(len(TERMS))
    theta[-1] = np.pi / 4                       # Y_R bias: start at P(beat) = 0.5
    theta.requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=LR)
    t0 = time.time()
    for step in range(STEPS):
        b = torch.randint(0, len(phi), (BATCH,))
        A = readout_torch(theta)
        p = ((phi[b] @ A) * phi[b]).sum(1).clamp(1e-5, 1 - 1e-5)
        loss = -(yy[b] * p.log() + (1 - yy[b]) * (1 - p).log()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"    [{label}] step {step:3d}  log-loss {loss.item():.5f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return theta.detach().numpy().astype(np.float64)


def predict(theta, u, chunk=50_000):
    U = expm(-1j * h_numpy(theta))
    A = (U.conj().T @ (PROJ_NP[:, None] * U))[0::2, 0::2].real
    out = []
    for i in range(0, len(u), chunk):
        phi = encode(u[i:i + chunk]).astype(np.float64)
        out.append(((phi @ A) * phi).sum(1))
    return np.concatenate(out)


# ------------------------------------------------------------------ IQLF / DIQLF
def entropy_R(psi):
    a = psi.reshape(DIM // 2, 2, -1)                         # [factor basis j, r, sample]
    rho = np.einsum("jas,jbs->sab", a, a.conj())
    tr = rho[:, 0, 0].real + rho[:, 1, 1].real
    disc = np.sqrt((rho[:, 0, 0].real - rho[:, 1, 1].real) ** 2 + 4 * np.abs(rho[:, 0, 1]) ** 2)
    lam = np.clip(np.stack([(tr + disc) / 2, (tr - disc) / 2], 1), 1e-15, 1)
    return -(lam * np.log(lam)).sum(1)


def diqlf(theta, u):
    """DIQLF_{k->R} over r in [0,1], averaged over the sample states (nats)."""
    H = h_numpy(theta)
    step = expm(-1j * H / IQLF_R)
    u_eps = expm(-1j * H * IQLF_EPS)
    frozen = [expm(-1j * h_numpy(theta, drop=k) * IQLF_EPS) for k in range(R)]
    psi = np.zeros((DIM, len(u)), complex)
    psi[0::2] = encode(u).T
    psi = expm(-1j * H / (2 * IQLF_R)) @ psi                  # midpoint rule, avoid r = 0
    T = np.zeros(R)
    for _ in range(IQLF_R):
        s_free = entropy_R(u_eps @ psi)
        for k in range(R):
            T[k] += (s_free - entropy_R(frozen[k] @ psi)).mean() / IQLF_EPS / IQLF_R
        psi = step @ psi
    return T


# ------------------------------------------------------------------ evaluation
def evaluate(hold, prob, uni, ff):
    y = hold.beat.to_numpy()
    gross, net = long_short(hold.assign(prob=prob), "prob", COST[uni])
    fx = ff.loc[gross.index + 1, ["mktrf", "smb", "hml", "rmw", "cma", "umd"]].to_numpy()
    a, ta = ols_alpha(gross.to_numpy(), fx)
    return dict(acc=np.mean((prob > 0.5) == y), auc=auc(prob, y), gross=ann(gross),
                t=nw_t(gross), net=ann(net), alpha=a * 12, t_alpha=ta)


def main():
    p = pd.read_pickle(OUT / "panel.pkl").dropna(subset=FACTORS)
    p = p[p.ym >= 1963 * 12 + 7]
    ff = q("SELECT date, mktrf, smb, hml, rmw, cma, umd FROM ff.fivefactors_monthly")
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index(ff.date.dt.year * 12 + ff.date.dt.month)
    cols = [f"r_{f}" for f in FACTORS]
    thetas = {}

    for uni in ("all", "large"):
        d = (p if uni == "all" else p[p.large]).copy()
        for f in FACTORS:
            d[f"r_{f}"] = d.groupby("ym")[f].rank(pct=True)
        disc, hold = d[d.ym <= DISC_END], d[d.ym >= HOLD_START]
        label = "ALL STOCKS" if uni == "all" else "LARGE CAPS"
        print(f"\n=== {label}: fitting on {len(disc):,} discovery stock-months ===", flush=True)

        Xd = np.column_stack([np.ones(len(disc)), disc[cols].to_numpy() - 0.5])
        b = logistic_fit(Xd, disc.beat.to_numpy().astype(float))
        Xh = np.column_stack([np.ones(len(hold)), hold[cols].to_numpy() - 0.5])
        classical = evaluate(hold, 1 / (1 + np.exp(-(Xh @ b))), uni, ff)

        theta = fit(disc[cols].to_numpy(), disc.beat.to_numpy(), f"quantum {uni}")
        thetas[uni] = theta
        quantum = evaluate(hold, predict(theta, hold[cols].to_numpy()), uni, ff)

        print(f"\n  HELD-OUT 2005-2024, {label}        classical     quantum")
        for key, name, fmt in (("acc", "accuracy", "{:.2%}"), ("auc", "AUC", "{:.4f}")):
            print(f"    {name:26}{fmt.format(classical[key]):>12}{fmt.format(quantum[key]):>12}")
        for key, name in (("gross", "long-short gross %/yr"), ("net", "long-short net %/yr")):
            print(f"    {name:26}{classical[key][0]*100:>11.1f}%{quantum[key][0]*100:>11.1f}%")
        print(f"    {'long-short gross Sharpe':26}{classical['gross'][1]:>12.2f}"
              f"{quantum['gross'][1]:>12.2f}")
        print(f"    {'long-short t-stat':26}{classical['t']:>12.2f}{quantum['t']:>12.2f}")
        print(f"    {'alpha vs FF5+mom %/yr':26}{classical['alpha']*100:>11.1f}%"
              f"{quantum['alpha']*100:>11.1f}%")
        print(f"    {'alpha t-stat':26}{classical['t_alpha']:>12.2f}{quantum['t_alpha']:>12.2f}",
              flush=True)

        if uni == "all":
            print("\n  Fitting a 2005-2024 model (in-sample, descriptive) for attribution...",
                  flush=True)
            thetas["all_held"] = fit(hold[cols].to_numpy(), hold.beat.to_numpy(), "quantum held")
            sd = disc[cols].to_numpy()[rng.choice(len(disc), IQLF_STATES, replace=False)]
            sh = hold[cols].to_numpy()[rng.choice(len(hold), IQLF_STATES, replace=False)]
            print("  Computing DIQLF attribution...", flush=True)
            t_disc, t_hold = diqlf(theta, sd), diqlf(thetas["all_held"], sh)

    print("\n" + "=" * 78)
    print("QUANTUM INFORMATION FLOW  factor qubit -> return qubit  (DIQLF, all stocks)")
    print("=" * 78)
    print(f"  {'factor':8}{'1963-2004 model':>18}{'share':>8}{'2005-2024 model':>18}{'share':>8}")
    sd_, sh_ = np.abs(t_disc).sum(), np.abs(t_hold).sum()
    for k in np.argsort(-np.abs(t_disc)):
        print(f"  {FACTORS[k]:8}{t_disc[k]:>+18.5f}{abs(t_disc[k])/sd_*100:>7.0f}%"
              f"{t_hold[k]:>+18.5f}{abs(t_hold[k])/sh_*100:>7.0f}%")
    print(f"  {'total |T|':8}{sd_:>18.5f}{'':>8}{sh_:>18.5f}")
    print("  (T > 0: the factor adds uncertainty to the return qubit; T < 0: it removes it.)")
    np.savez(OUT / "quantum_params.npz", **thetas, terms=np.array([n for _, n in TERMS]))


if __name__ == "__main__":
    main()
