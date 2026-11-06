"""Information-theoretic screen: which first-20-minute features carry real information
about the NEXT HOUR (9:50-10:50 ET) in SPY / QQQ / IWM?

Fixed protocol (set before looking at results):
  - Discovery: Jun 2018 - Dec 2022.  Confirmation: Jan 2023 - Oct 2026 (held out).
  - 14 features x 4 targets = 56 tests. Statistic = mutual information on within-symbol
    rank bins, p-value from a within-symbol shuffle null, Benjamini-Hochberg FDR.
  - Discovery survivors must re-pass (p < 0.05, same sign) on the confirmation period.
  - Final bar: confirmed features must add OUT-OF-SAMPLE R^2 beyond a pre-open baseline
    (VIX, trailing 20d vol, yesterday's range), fit on discovery, scored on confirmation.

Data: Databento XNAS.ITCH 1-minute bars (backtest/data/intraday/, git-ignored).
Run:  python research/opening_screen.py
"""
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "backtest" / "data" / "intraday"
FEAT_CSV = DATA / "opening_features.csv"
SPLIT = pd.Timestamp("2023-01-01").date()
WIN, TGT_END = 20, 80          # minutes after 9:30: window [0,20), target [20,80)
NBINS, NPERM, FDR_Q = 8, 500, 0.05
rng = np.random.default_rng(20261103)

FEATURES = ["gap", "open_ret", "first5_ret", "later15_ret", "win_rv", "win_range",
            "rel_vol", "dir_eff", "range_pos", "vwap_dev", "prev_ret", "prev_range",
            "rv20", "vix"]
TARGETS = {"t_ret": "direction (signed return)", "t_absret": "absolute move",
           "t_range": "high-low range", "t_trend": "trend vs chop"}
BASELINE = ["rv20", "vix", "prev_range"]          # all known before the open
LOGGED = {"win_rv", "win_range", "rel_vol", "rv20", "vix", "prev_range",
          "t_absret", "t_range"}


# --------------------------------------------------------------------------- data
def load_bars():
    frames = []
    for f in sorted(DATA.glob("xnas_ohlcv1m_*.dbn.zst")):
        frames.append(db.DBNStore.from_file(f).to_df()[
            ["open", "high", "low", "close", "volume", "symbol"]])
    df = pd.concat(frames)
    df.index = df.index.tz_convert("America/New_York")
    mins = df.index.hour * 60 + df.index.minute - 570          # 0 = 9:30
    df = df[(mins >= 0) & (mins < 390)].copy()
    df["mins"] = mins[(mins >= 0) & (mins < 390)]
    df["date"] = df.index.date
    return df


def window_target(g):
    w, t = g[g.mins < WIN], g[(g.mins >= WIN) & (g.mins < TGT_END)]
    if len(w) < 18 or len(t) < 55:
        return None
    o, cw = w.open.iloc[0], w.close.iloc[-1]
    hi, lo = w.high.max(), w.low.min()
    wc = np.r_[o, w.close.values]
    f5 = w[w.mins < 5]
    typ = (w.high + w.low + w.close) / 3
    vwap = (typ * w.volume).sum() / max(w.volume.sum(), 1)
    tc = np.r_[cw, t.close.values]
    tpath = np.abs(np.diff(tc)).sum()
    return {
        "open_930": o, "win_vol": w.volume.sum(),
        "open_ret": cw / o - 1,
        "first5_ret": f5.close.iloc[-1] / o - 1 if len(f5) else np.nan,
        "later15_ret": cw / f5.close.iloc[-1] - 1 if len(f5) else np.nan,
        "win_rv": np.sqrt((np.diff(np.log(wc)) ** 2).sum()),
        "win_range": (hi - lo) / o,
        "dir_eff": abs(cw - o) / max(np.abs(np.diff(wc)).sum(), 1e-12),
        "range_pos": (cw - lo) / (hi - lo) if hi > lo else 0.5,
        "vwap_dev": cw / vwap - 1,
        "t_ret": t.close.iloc[-1] / cw - 1,
        "t_range": (t.high.max() - t.low.min()) / cw,
        "t_trend": abs(t.close.iloc[-1] - cw) / max(tpath, 1e-12),
    }


def build_features():
    if FEAT_CSV.exists():
        return pd.read_csv(FEAT_CSV, parse_dates=["date"])
    print("Building per-day features from minute bars (one-time)...", flush=True)
    df = load_bars()
    daily = (df.groupby(["symbol", "date"])
               .agg(d_high=("high", "max"), d_low=("low", "min"), d_close=("close", "last"))
               .reset_index())
    early = df[df.mins < TGT_END]
    rows = []
    for (sym, d), g in early.groupby(["symbol", "date"], sort=True):
        r = window_target(g)
        if r:
            r.update(symbol=sym, date=d)
            rows.append(r)
    wt = pd.DataFrame(rows)
    out = []
    for sym, g in daily.merge(wt, on=["symbol", "date"], how="left").groupby("symbol"):
        g = g.sort_values("date").copy()
        prev_close = g.d_close.shift(1)
        g["gap"] = g.open_930 / prev_close - 1
        g["prev_ret"] = prev_close / g.d_close.shift(2) - 1
        g["prev_range"] = ((g.d_high - g.d_low) / g.d_close).shift(1)
        g["rv20"] = np.log(g.d_close).diff().rolling(20).std().shift(1) * np.sqrt(252)
        # min_periods: a halt day (no opening window) must not blank the next 20 days
        g["rel_vol"] = g.win_vol / g.win_vol.rolling(20, min_periods=10).median().shift(1)
        out.append(g)
    feat = pd.concat(out)
    vix = yf.Ticker("^VIX").history(start="2018-01-01", interval="1d")["Close"]
    vix = vix.rename("vix_close").reset_index()
    vix["date"] = pd.to_datetime(vix["Date"]).dt.tz_localize(None).dt.normalize()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = pd.merge_asof(feat.sort_values("date"), vix[["date", "vix_close"]],
                         on="date", allow_exact_matches=False)   # prior-day VIX only
    feat["vix"] = feat["vix_close"]
    feat["t_absret"] = feat["t_ret"].abs()
    feat = feat.dropna(subset=FEATURES + list(TARGETS)).reset_index(drop=True)
    feat.to_csv(FEAT_CSV, index=False)
    return feat


# --------------------------------------------------------------------------- stats
def rank_bins(s, groups, nb=NBINS):
    r = s.groupby(groups).rank(method="first", pct=True)
    return np.minimum((r * nb).astype(int), nb - 1).to_numpy()


def mutual_info(xb, yb, nb=NBINS):
    joint = np.zeros((nb, nb))
    np.add.at(joint, (xb, yb), 1)
    p = joint / joint.sum()
    pxy = p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / pxy[nz])).sum())


def mi_test(d, f, t):
    grp = d["symbol"]
    xb, yb = rank_bins(d[f], grp), rank_bins(d[t], grp)
    obs = mutual_info(xb, yb)
    idx = [np.where(grp.to_numpy() == s)[0] for s in grp.unique()]
    null = np.empty(NPERM)
    for i in range(NPERM):
        yp = yb.copy()
        for ix in idx:
            yp[ix] = yb[rng.permutation(ix)]
        null[i] = mutual_info(xb, yp)
    p = (np.sum(null >= obs) + 1) / (NPERM + 1)
    rho = d[f].groupby(grp).rank().corr(d[t].groupby(grp).rank())
    return obs - null.mean(), p, rho          # bias-corrected MI (nats)


def bh(p, q=FDR_Q):
    p = np.asarray(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    return p <= (p[order][np.where(ok)[0].max()] if ok.any() else -1)


def prep(d, cols, ref):
    """Per-symbol: log skewed vars, winsorize & z-score using DISCOVERY stats only."""
    out = pd.DataFrame(index=d.index)
    for c in cols:
        for sym in d.symbol.unique():
            m, r = d.symbol == sym, ref[ref.symbol == sym]
            x = np.log(d.loc[m, c].clip(lower=1e-9)) if c in LOGGED else d.loc[m, c]
            xr = np.log(r[c].clip(lower=1e-9)) if c in LOGGED else r[c]
            lo, hi = xr.quantile(0.01), xr.quantile(0.99)
            xr = xr.clip(lo, hi)
            out.loc[m, c] = (x.clip(lo, hi) - xr.mean()) / (xr.std() or 1)
    return out


def oos_r2(disc, conf, xs, t):
    Xd = np.column_stack([np.ones(len(disc))] + [disc[c] for c in xs]) if xs else np.ones((len(disc), 1))
    Xc = np.column_stack([np.ones(len(conf))] + [conf[c] for c in xs]) if xs else np.ones((len(conf), 1))
    beta = np.linalg.lstsq(Xd, disc[t].to_numpy(), rcond=None)[0]
    pred = Xc @ beta
    y = conf[t].to_numpy()
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pred


# --------------------------------------------------------------------------- main
def main():
    feat = build_features()
    feat["date_d"] = feat["date"].dt.date
    disc = feat[feat.date_d < SPLIT].reset_index(drop=True)
    conf = feat[feat.date_d >= SPLIT].reset_index(drop=True)
    print(f"{len(feat):,} symbol-days | discovery {len(disc):,} "
          f"({disc.date.min().date()}..{disc.date.max().date()}) | "
          f"confirmation {len(conf):,} ({conf.date.min().date()}..{conf.date.max().date()})\n",
          flush=True)

    tests = [(f, t) for t in TARGETS for f in FEATURES]
    res = {k: mi_test(disc, *k) for k in tests}
    passed = bh([res[k][1] for k in tests])
    surv = [k for k, ok in zip(tests, passed) if ok]
    print(f"Discovery: {len(surv)} of {len(tests)} feature-target pairs pass FDR q={FDR_Q}\n")

    zd = prep(disc, FEATURES + list(TARGETS), disc).assign(symbol=disc.symbol)
    zc = prep(conf, FEATURES + list(TARGETS), disc).assign(symbol=conf.symbol)

    for t, label in TARGETS.items():
        rows = [k for k in surv if k[1] == t]
        base_r2, _ = oos_r2(zd, zc, BASELINE, t)
        print("=" * 78)
        print(f"TARGET: next-hour {label}   (pre-open baseline OOS R^2 = {base_r2*100:+.2f}%)")
        print("=" * 78)
        if not rows:
            print("  no feature passes discovery FDR\n")
            continue
        print(f"  {'feature':12}{'MI disc':>9}{'rho disc':>9}{'p conf':>8}{'rho conf':>9}"
              f"{'confirmed':>11}{'+OOS R2 vs base':>17}")
        for f, _ in sorted(rows, key=lambda k: -res[k][0]):
            mi_d, _, rho_d = res[(f, t)]
            _, p_c, rho_c = mi_test(conf, f, t)
            ok = p_c < 0.05 and np.sign(rho_c) == np.sign(rho_d)
            if f in BASELINE:
                inc = "(baseline)"
            else:
                r2, _ = oos_r2(zd, zc, BASELINE + [f], t)
                inc = f"{(r2 - base_r2)*100:+.2f}%"
            print(f"  {f:12}{mi_d:>9.4f}{rho_d:>+9.2f}{p_c:>8.3f}{rho_c:>+9.2f}"
                  f"{'YES' if ok else 'no':>11}{inc:>17}")
        allf = [f for f in FEATURES if f not in BASELINE]
        r2_all, pred = oos_r2(zd, zc, BASELINE + allf, t)
        print(f"\n  All opening features + baseline: OOS R^2 {r2_all*100:+.2f}% "
              f"(adds {(r2_all - base_r2)*100:+.2f}% over pre-open info)")
        if t == "t_ret":
            hit = np.mean(np.sign(pred) == np.sign(zc[t].to_numpy()))
            print(f"  OOS directional hit rate of that model: {hit*100:.1f}%  (coin flip = 50%)")
        else:
            q = pd.qcut(pred, 5, labels=False)
            raw = conf[t].to_numpy()
            print(f"  OOS: actual next-hour {label} when model predicts lowest vs highest quintile: "
                  f"{np.mean(raw[q == 0])*100:.2f}% vs {np.mean(raw[q == 4])*100:.2f}%")
        print()


if __name__ == "__main__":
    main()
