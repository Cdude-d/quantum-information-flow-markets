"""What do the CANONICAL documented anomalies actually deliver? Fama-French daily
factors from WRDS (gold standard): market (mktrf), momentum (umd), value (hml),
size (smb). Long-run and by-decade Sharpe / return / max drawdown -- the real,
published versions of the effects we've been chasing, and whether they've decayed.
"""
import numpy as np
import pandas as pd
from wrds_lib import q


def stats(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 100:
        return dict(ann=np.nan, sh=np.nan, dd=np.nan)
    vol = d.std() * np.sqrt(252)
    cum = np.cumprod(1 + d)
    return dict(ann=(1 + d).prod() ** (252 / len(d)) - 1,
                sh=(d.mean() * 252) / vol if vol > 0 else np.nan,
                dd=(cum / np.maximum.accumulate(cum) - 1).min())


def main():
    df = q("""SELECT date, mktrf, smb, hml, umd, rf FROM ff.factors_daily
              WHERE umd IS NOT NULL ORDER BY date""")
    df["date"] = pd.to_datetime(df["date"])
    # WRDS FF factors are decimals (e.g. 0.0052); confirm via scale
    scale = "decimal" if df["mktrf"].std() < 0.1 else "percent(/100)"
    if df["mktrf"].std() > 0.1:
        for c in ["mktrf", "smb", "hml", "umd", "rf"]:
            df[c] = df[c] / 100.0
    print(f"FF daily factors: {df['date'].min().date()}..{df['date'].max().date()} "
          f"({len(df):,} days, units={scale})\n")

    facs = {"Market (MKT-RF)": "mktrf", "Momentum (UMD)": "umd",
            "Value (HML)": "hml", "Size (SMB)": "smb"}
    print(f"{'factor':18}{'window':>12}{'ann%':>8}{'Sharpe':>8}{'maxDD':>8}")
    print("-" * 54)
    df["decade"] = (df["date"].dt.year // 10) * 10
    for name, col in facs.items():
        s = stats(df[col])
        print(f"{name:18}{'full':>12}{s['ann']*100:>7.1f}{s['sh']:>8.2f}{s['dd']*100:>7.0f}%")
        for dec in sorted(df["decade"].unique()):
            if dec < 1960:
                continue
            sd = stats(df[df["decade"] == dec][col])
            if not np.isnan(sd["sh"]):
                print(f"{'':18}{str(dec)+'s':>12}{sd['ann']*100:>7.1f}{sd['sh']:>8.2f}"
                      f"{sd['dd']*100:>7.0f}%")
        print()

    # momentum recent decade vs history -> decay check
    recent = stats(df[df["date"] >= "2010-01-01"]["umd"])
    early = stats(df[df["date"] < "2010-01-01"]["umd"])
    print(f"Momentum decay check: pre-2010 Sharpe {early['sh']:.2f} vs "
          f"2010+ Sharpe {recent['sh']:.2f}")


if __name__ == "__main__":
    main()
