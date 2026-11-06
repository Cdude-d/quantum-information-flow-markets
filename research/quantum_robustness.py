"""Robustness check of the large-cap quantum factor model's held-out long-short result.

Pre-registered rule: the result counts as real only if ALL hold:
  1. >= 4 of 5 independently seeded models have positive held-out gross Sharpe,
  2. the 5-seed ensemble (mean predicted probability) has long-short t >= 1.96,
  3. the ensemble beats the classical logistic model in a paired monthly test, t >= 1.96.
Also reports DIQLF attribution shares per seed (stability of the explanation).

Run:  python research/quantum_robustness.py
"""
import numpy as np
import pandas as pd
import torch

import quantum_factor_model as qm
from factor_benchmark import (COST, DISC_END, HOLD_START, ann, auc, logistic_fit,
                              long_short, nw_t, ols_alpha)
from factor_panel import FACTORS, OUT
from wrds_lib import q

SEEDS = [11, 22, 33, 44, 55]
DIQLF_STATES = 800
UNI = "large"


def summarize(hold, prob, classical_gross, ff):
    gross, net = long_short(hold.assign(prob=prob), "prob", COST[UNI])
    fx = ff.loc[gross.index + 1, ["mktrf", "smb", "hml", "rmw", "cma", "umd"]].to_numpy()
    a, ta = ols_alpha(gross.to_numpy(), fx)
    diff = (gross - classical_gross).dropna()
    return dict(auc=auc(prob, hold.beat.to_numpy()), gross=ann(gross)[0], sharpe=ann(gross)[1],
                t=nw_t(gross), net=ann(net)[0], alpha=a * 12, t_alpha=ta,
                diff=diff.mean() * 12, t_diff=nw_t(diff))


def main():
    p = pd.read_pickle(OUT / "panel.pkl").dropna(subset=FACTORS)
    p = p[(p.ym >= 1963 * 12 + 7) & p.large].copy()
    cols = [f"r_{f}" for f in FACTORS]
    for f in FACTORS:
        p[f"r_{f}"] = p.groupby("ym")[f].rank(pct=True)
    disc, hold = p[p.ym <= DISC_END], p[p.ym >= HOLD_START]
    ff = q("SELECT date, mktrf, smb, hml, rmw, cma, umd FROM ff.fivefactors_monthly")
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index(ff.date.dt.year * 12 + ff.date.dt.month)

    Xd = np.column_stack([np.ones(len(disc)), disc[cols].to_numpy() - 0.5])
    b = logistic_fit(Xd, disc.beat.to_numpy().astype(float))
    Xh = np.column_stack([np.ones(len(hold)), hold[cols].to_numpy() - 0.5])
    c_prob = 1 / (1 + np.exp(-(Xh @ b)))
    c_gross, _ = long_short(hold.assign(prob=c_prob), "prob", COST[UNI])
    classical = summarize(hold, c_prob, c_gross, ff)

    probs, rows, shares = [], [], []
    ud, uh = disc[cols].to_numpy(), hold[cols].to_numpy()
    for s in SEEDS:
        print(f"\n--- seed {s} ---", flush=True)
        qm.rng = np.random.default_rng(s)
        torch.manual_seed(s)
        theta = qm.fit(ud, disc.beat.to_numpy(), f"large seed {s}")
        prob = qm.predict(theta, uh)
        probs.append(prob)
        r = summarize(hold, prob, c_gross, ff)
        r["seed"] = s
        rows.append(r)
        T = qm.diqlf(theta, ud[qm.rng.choice(len(ud), DIQLF_STATES, replace=False)])
        shares.append(np.abs(T) / np.abs(T).sum())
        print(f"    held-out: AUC {r['auc']:.4f}  LS gross {r['gross']*100:+.1f}%/yr  "
              f"Sharpe {r['sharpe']:+.2f}  t {r['t']:+.2f}", flush=True)

    ens = summarize(hold, np.mean(probs, axis=0), c_gross, ff)

    print("\n" + "=" * 92)
    print("LARGE CAPS, held-out 2005-2024 — quantum model across 5 seeds vs classical")
    print("=" * 92)
    hdr = (f"  {'model':16}{'AUC':>8}{'gross %/yr':>12}{'Sharpe':>8}{'t':>7}{'net %/yr':>10}"
           f"{'alpha %':>9}{'t(a)':>7}{'vs classical':>14}{'t(diff)':>9}")
    print(hdr)

    def line(name, r):
        print(f"  {name:16}{r['auc']:>8.4f}{r['gross']*100:>11.1f}%{r['sharpe']:>8.2f}{r['t']:>7.2f}"
              f"{r['net']*100:>9.1f}%{r['alpha']*100:>8.1f}%{r['t_alpha']:>7.2f}"
              f"{r['diff']*100:>13.1f}%{r['t_diff']:>9.2f}")

    line("classical", classical)
    for r in rows:
        line(f"quantum s{r['seed']}", r)
    line("quantum ensemble", ens)
    print("  (original single run: gross +5.3%/yr, Sharpe 0.44, t 1.77)")

    n_pos = sum(r["sharpe"] > 0 for r in rows)
    c1, c2, c3 = n_pos >= 4, ens["t"] >= 1.96, ens["t_diff"] >= 1.96
    print("\nPRE-REGISTERED VERDICT")
    print(f"  1. seeds with positive Sharpe: {n_pos}/5            -> {'PASS' if c1 else 'FAIL'}")
    print(f"  2. ensemble long-short t = {ens['t']:.2f} (>= 1.96)  -> {'PASS' if c2 else 'FAIL'}")
    print(f"  3. ensemble vs classical paired t = {ens['t_diff']:.2f} (>= 1.96) -> "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"  => {'REAL: survives the robustness check' if c1 and c2 and c3 else 'NOT ESTABLISHED: consistent with training luck'}")

    sh = np.array(shares)
    print("\nDIQLF attribution share (large-cap 1963-2004 models), mean ± std across seeds:")
    for k in np.argsort(-sh.mean(0)):
        print(f"  {FACTORS[k]:7}{sh[:, k].mean()*100:>6.0f}% ± {sh[:, k].std()*100:>3.0f}%   "
              + " ".join(f"{v*100:>3.0f}" for v in sh[:, k]))


if __name__ == "__main__":
    main()
