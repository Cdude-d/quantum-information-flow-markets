"""Build the monthly stock-factor panel (CRSP + Compustat via WRDS), 1960-2024.

One row per stock-month t with 9 factors known at the end of month t and the target
(month t+1 return). Shared input for the classical benchmark and the quantum model.

  Universe: CRSP common stocks (shrcd 10/11) on NYSE/AMEX/NASDAQ (exchcd 1/2/3),
            delisting returns included (Shumway -30% for missing performance delists).
  Price factors (CRSP): mom (t-11..t-1), rev (t), ltrev (t-35..t-12), size (log ME),
            vol (12m std), beta (60m vs market, min 24).
  Fundamentals (Compustat funda, matched on 8-char CUSIP because the CCM link table is
            not licensed; used from datadate + 6 months): bm, gp, ag.
  Targets:  ret_next (month t+1 total return), beat (ret_next > cross-sectional median).

Output: backtest/data/factors/panel.pkl (git-ignored).
Run:  python research/factor_panel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from wrds_lib import q

OUT = Path(__file__).resolve().parent.parent / "backtest" / "data" / "factors"
START, END = "1960-01-01", "2024-12-31"
FACTORS = ["mom", "rev", "ltrev", "size", "vol", "beta", "bm", "gp", "ag"]


def pull_crsp():
    print("Pulling CRSP monthly stock file (this is the big query)...", flush=True)
    msf = q(f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.cusip, b.shrcd, b.exchcd
        FROM crsp_q_stock.msf a
        JOIN crsp_q_stock.msenames b
          ON a.permno = b.permno AND b.namedt <= a.date AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{START}' AND '{END}'
          AND b.shrcd IN (10, 11) AND b.exchcd IN (1, 2, 3)""")
    dl = q("SELECT permno, dlstdt, dlret, dlstcd FROM crsp_q_stock.msedelist")
    print(f"  {len(msf):,} stock-months, {msf.permno.nunique():,} stocks; "
          f"{len(dl):,} delisting records", flush=True)
    return msf, dl


def pull_compustat():
    print("Pulling Compustat annual fundamentals...", flush=True)
    f = q("""
        SELECT gvkey, datadate, cusip, at, revt, cogs, seq, ceq, lt, pstk, pstkrv, pstkl, txditc
        FROM comp.funda
        WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
          AND datadate >= '1958-01-01'""")
    print(f"  {len(f):,} firm-years", flush=True)
    return f


def cached(name, fn):
    path = OUT / f"raw_{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    obj = fn()
    pd.to_pickle(obj, path)
    return obj


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    msf, dl = cached("crsp", pull_crsp)
    comp = cached("compustat", pull_compustat)
    mkt = cached("ff_mkt", lambda: q("SELECT date, mktrf, rf FROM ff.factors_monthly"))

    # ---- returns with delisting -------------------------------------------------
    msf["date"] = pd.to_datetime(msf["date"])
    msf["ym"] = msf.date.dt.year * 12 + msf.date.dt.month
    dl["dlstdt"] = pd.to_datetime(dl["dlstdt"])
    dl["ym"] = dl.dlstdt.dt.year * 12 + dl.dlstdt.dt.month
    perf = dl.dlstcd.between(500, 599)
    dl.loc[dl.dlret.isna() & perf, "dlret"] = -0.30                 # Shumway (1997)
    msf = msf.merge(dl[["permno", "ym", "dlret"]], on=["permno", "ym"], how="left")
    r = msf.ret.astype(float)
    d = msf.dlret.astype(float)
    msf["ret_adj"] = np.where(r.notna() & d.notna(), (1 + r) * (1 + d) - 1,
                              np.where(r.notna(), r, d))
    msf["me"] = msf.prc.abs() * msf.shrout / 1000.0                # $ millions
    msf = (msf.sort_values(["permno", "ym"])
              .drop_duplicates(["permno", "ym"], keep="last").reset_index(drop=True))

    # ---- market return for beta --------------------------------------------------
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["ym"] = mkt.date.dt.year * 12 + mkt.date.dt.month
    msf = msf.merge(mkt[["ym", "mktrf", "rf"]], on="ym", how="left")
    msf["mret"] = msf.mktrf + msf.rf

    # ---- price-based factors (gap-checked by month index) ------------------------
    print("Computing price-based factors...", flush=True)
    g = msf.groupby("permno", sort=False)
    lr = np.log1p(msf.ret_adj.clip(lower=-0.99)).fillna(0.0)
    cum = lr.groupby(msf.permno).cumsum()
    ym = msf.ym

    def lag(s, k):
        return s.groupby(msf.permno).shift(k)

    def contiguous(k):
        return (ym - lag(ym, k)) == k

    msf["rev"] = msf.ret_adj
    msf["mom"] = np.where(contiguous(12), np.expm1(lag(cum, 1) - lag(cum, 12)), np.nan)
    msf["ltrev"] = np.where(contiguous(36), np.expm1(lag(cum, 12) - lag(cum, 36)), np.nan)
    msf["size"] = np.log(msf.me.where(msf.me > 0))
    msf["vol"] = g.ret_adj.transform(lambda s: s.rolling(12, min_periods=10).std())
    rm = msf.ret_adj * msf.mret
    mean = lambda s: s.groupby(msf.permno).transform(  # noqa: E731
        lambda x: x.rolling(60, min_periods=24).mean())
    er, em, erm, em2 = mean(msf.ret_adj), mean(msf.mret), mean(rm), mean(msf.mret ** 2)
    msf["beta"] = (erm - er * em) / (em2 - em ** 2)

    # ---- targets -----------------------------------------------------------------
    nxt = g.ret_adj.shift(-1)
    msf["ret_next"] = np.where((g.ym.shift(-1) - ym) == 1, nxt, np.nan)

    # ---- fundamentals via CUSIP, available 6 months after fiscal year end --------
    print("Merging fundamentals (CUSIP match, 6-month availability lag)...", flush=True)
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp = comp.sort_values(["gvkey", "datadate"])
    # bracket access: comp.at / comp.lt would hit pandas' .at indexer and .lt method
    assets, liab = comp["at"], comp["lt"]
    pref = comp.pstkrv.fillna(comp.pstkl).fillna(comp.pstk).fillna(0)
    se = comp.seq.fillna(comp.ceq + comp.pstk.fillna(0)).fillna(assets - liab)
    comp["be"] = se + comp.txditc.fillna(0) - pref
    comp["gp"] = (comp.revt - comp.cogs) / assets.where(assets > 0)
    prev_assets = assets.groupby(comp.gvkey).shift(1)
    comp["ag"] = assets / prev_assets.where(prev_assets > 0) - 1
    comp["cusip8"] = comp.cusip.str[:8]
    comp["avail"] = comp.datadate + pd.DateOffset(months=6)
    comp = comp.dropna(subset=["cusip8"])[["cusip8", "avail", "be", "gp", "ag"]]
    comp = comp.sort_values("avail")
    msf["cusip8"] = msf.cusip.str[:8]
    msf = msf.sort_values("date")
    msf = pd.merge_asof(msf, comp, left_on="date", right_on="avail", by="cusip8",
                        direction="backward", tolerance=pd.Timedelta(days=548))
    msf["bm"] = msf.be.where(msf.be > 0) / msf.me

    # ---- universe filter + cross-sectional target --------------------------------
    keep = (msf.prc.abs() >= 5) & msf.ret_next.notna()
    p = msf.loc[keep, ["permno", "date", "ym", "exchcd", "me", "ret_next"] + FACTORS].copy()
    p["beat"] = (p.ret_next > p.groupby("ym").ret_next.transform("median")).astype(int)
    nyse_med = p[p.exchcd == 1].groupby("ym").me.median()
    p["large"] = p.me >= p.ym.map(nyse_med)
    p = p.sort_values(["ym", "permno"]).reset_index(drop=True)
    p.to_pickle(OUT / "panel.pkl")

    have = p.dropna(subset=FACTORS)
    print(f"\nPanel: {len(p):,} stock-months ({p.date.min():%Y-%m}..{p.date.max():%Y-%m}), "
          f"{p.permno.nunique():,} stocks")
    print(f"  with all 9 factors: {len(have):,} ({len(have)/len(p)*100:.0f}%), "
          f"large caps: {int(have.large.sum()):,}")
    print("  factor coverage:", {f: f"{p[f].notna().mean()*100:.0f}%" for f in FACTORS})


if __name__ == "__main__":
    build()
