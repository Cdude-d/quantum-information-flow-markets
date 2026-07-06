"""Is market von Neumann entropy S(rho) a useful FORWARD risk signal?

S(rho) (low = everything-correlated = stress) clearly tracks CURRENT stress. The
honest questions for a risk-management use:
  1. Does it predict forward realized vol / drawdown (bucketed by entropy)?
  2. THE key test: does it add anything BEYOND trailing vol? (vol clusters, so
     trailing vol already predicts forward vol -- if entropy just proxies that,
     it's useless). Measured as incremental R^2.
  3. Does it predict its NATIVE target -- forward average correlation?
  4. Is it leading or merely coincident with vol?

This is about RISK timing (de-risk when stressed), NOT return alpha.
Free yfinance data. Run:  python research/entropy_risk_signal.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS = ["SPY", "QQQ", "IWM", "XLF", "XLE", "EEM", "TLT", "HYG", "GLD", "USO", "UUP"]
WINDOW, FWD = 60, 21
PERIOD = "12y"


def vn_entropy(corr):
    rho = corr / np.trace(corr)
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)))


def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - (res @ res) / (((y - y.mean()) ** 2).sum())


def main():
    print("Fetching daily data (free)...", flush=True)
    cl = {}
    for s in ASSETS:
        h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
        if len(h) > 500:
            h.index = h.index.tz_localize(None).normalize()
            cl[s] = h
    R = pd.concat(cl, axis=1).pct_change().dropna()
    spy = R["SPY"].to_numpy()
    arr = R.to_numpy()
    n = len(R)
    print(f"{len(R.columns)} assets, {n} days, entropy window {WINDOW}d, horizon {FWD}d.\n",
          flush=True)

    rows = []
    for k in range(WINDOW, n - FWD):
        C = np.corrcoef(arr[k - WINDOW:k].T)
        ent = vn_entropy(C)
        off = C[~np.eye(len(C), dtype=bool)]
        tvol = spy[k - FWD:k].std() * np.sqrt(252)
        fvol = spy[k:k + FWD].std() * np.sqrt(252)
        cum = np.cumprod(1 + spy[k:k + FWD])
        fdd = (cum / np.maximum.accumulate(cum) - 1).min()
        Cf = np.corrcoef(arr[k:k + FWD].T)
        favg = Cf[~np.eye(len(Cf), dtype=bool)].mean()
        rows.append((ent, tvol, fvol, fdd, off.mean(), favg))
    d = pd.DataFrame(rows, columns=["ent", "tvol", "fvol", "fdd", "corr_now", "fcorr"])

    print("=" * 60)
    print("1) Entropy vs FORWARD 21d realized vol")
    print(f"   corr(entropy, forward vol)     = {d['ent'].corr(d['fvol']):+.2f}  "
          f"(negative = low entropy -> high future vol)")
    print(f"   corr(entropy, CURRENT vol)     = {d['ent'].corr(d['tvol']):+.2f}  "
          f"(coincident link)")
    d["q"] = pd.qcut(d["ent"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    print("\n   Forward vol & drawdown by entropy quintile:")
    print(f"   {'entropy bucket':14}{'fwd vol':>9}{'fwd maxDD':>11}")
    for q, g in d.groupby("q", observed=True):
        print(f"   {str(q):14}{g['fvol'].mean()*100:>8.0f}%{g['fdd'].mean()*100:>10.1f}%")

    print("\n" + "=" * 60)
    print("2) KEY TEST — does entropy add anything beyond trailing vol?")
    base = r2(d["fvol"].to_numpy(), [d["tvol"].to_numpy()])
    full = r2(d["fvol"].to_numpy(), [d["tvol"].to_numpy(), d["ent"].to_numpy()])
    print(f"   R^2 forward vol ~ trailing vol         : {base*100:.1f}%")
    print(f"   R^2 forward vol ~ trailing vol + entropy: {full*100:.1f}%")
    print(f"   INCREMENTAL R^2 from entropy           : {(full-base)*100:+.1f}%  "
          f"{'(adds real info)' if full-base > 0.005 else '(adds ~nothing -> just a vol proxy)'}")

    print("\n" + "=" * 60)
    print("3) Native target — does entropy predict FORWARD average correlation?")
    print(f"   corr(entropy, forward avg corr) = {d['ent'].corr(d['fcorr']):+.2f}")
    bc = r2(d["fcorr"].to_numpy(), [d["corr_now"].to_numpy()])
    bf = r2(d["fcorr"].to_numpy(), [d["corr_now"].to_numpy(), d["ent"].to_numpy()])
    print(f"   incremental R^2 over current corr: {(bf-bc)*100:+.1f}%")

    print("\n" + "=" * 60)
    print("4) Leading or coincident? entropy vs vol at lags")
    for lag in [-FWD, 0, FWD]:
        s = d["ent"]
        v = d["tvol"].shift(-lag) if lag else d["tvol"]
        c = s.corr(v)
        when = "past vol" if lag < 0 else ("current vol" if lag == 0 else "future vol")
        print(f"   corr(entropy_t, {when:12}) = {c:+.2f}")
    print("\nReadout: if forward link ~ coincident link AND incremental R^2 ~ 0,")
    print("entropy is a (valid) stress thermometer but not a forward edge over vol.")


if __name__ == "__main__":
    main()
