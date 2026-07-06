"""Free liquidity-vs-richness screen for 'less-looked-at' options (yfinance, $0).

For a gradient of names, measure at the nearest weekly-ish expiry:
  - spread_cost: ATM bid/ask as % of mid = the round-trip friction to trade it
  - iv_rv: ATM implied vol / 20-day realized vol = how 'rich' the premium looks
  - liquidity: ATM option volume + open interest

The point: in illiquid names the premium often LOOKS rich but the spread eats it.
A real candidate needs richness that survives the spread. This is a one-snapshot
SCREEN to find names worth a proper historical/OOS study — not an edge itself.

Run:  python backtest/screen_universe.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout import market_data  # noqa: E402
from options_scout.signals import ewma_vol  # noqa: E402

# A gradient from hyper-liquid to genuinely less-watched.
GROUPS = {
    "liquid-baseline": ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA"],
    "retail-heavy": ["PLTR", "SOFI", "COIN", "HOOD", "MARA", "RIOT", "CVNA", "DKNG"],
    "midcap-less-watched": ["CROX", "DECK", "FSLR", "ENPH", "ON", "CLF", "AA",
                            "BLDR", "YETI", "EXEL", "HALO", "AR", "RRC", "MPWR",
                            "JEF", "SAIA", "WING", "FIVE"],
}


def nearest_expiry(symbol):
    exps = market_data.list_expiries(symbol)
    today = date.today()
    pool = [e for e in exps if timedelta(days=4) <= (e - today) <= timedelta(days=45)]
    return min(pool, key=lambda e: abs((e - today).days - 25)) if pool else None


def atm_metrics(symbol):
    exp = nearest_expiry(symbol)
    if exp is None:
        return None
    hist = market_data.get_history(symbol, "6mo")
    spot = market_data.get_spot(symbol, hist)
    rv = ewma_vol(hist["Close"])
    calls, puts = market_data.get_chain(symbol, exp)
    out = {}
    for df, col in ((calls, "c"), (puts, "p")):
        if df.empty:
            return None
        row = df.iloc[(df["strike"] - spot).abs().argmin()]
        bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
        mid = (bid + ask) / 2
        if mid <= 0:
            return None
        out[col] = {
            "spread_pct": (ask - bid) / mid,
            "iv": float(row.get("impliedVolatility") or np.nan),
            "liq": float((row.get("volume") or 0)) + float((row.get("openInterest") or 0)),
        }
    iv = np.nanmean([out["c"]["iv"], out["p"]["iv"]])
    return {
        "exp_days": (exp - date.today()).days,
        "spread_pct": (out["c"]["spread_pct"] + out["p"]["spread_pct"]) / 2,
        "iv": iv, "rv": rv, "iv_rv": iv / rv if rv > 0 else np.nan,
        "liq": out["c"]["liq"] + out["p"]["liq"],
    }


def main():
    print(f"{'sym':6}{'group':22}{'spread%':>9}{'IV/RV':>7}{'IV%':>6}{'RV%':>6}"
          f"{'ATMliq':>9}{'tradable?':>11}", flush=True)
    print("-" * 76, flush=True)
    rows = []
    for group, syms in GROUPS.items():
        for sym in syms:
            try:
                m = atm_metrics(sym)
            except Exception as e:  # noqa: BLE001
                print(f"{sym:6}{group:22}  skip ({type(e).__name__})", flush=True)
                continue
            if m is None:
                print(f"{sym:6}{group:22}  no usable chain", flush=True)
                continue
            # Tradable-edge heuristic: premium must be rich AND spread must be a
            # small fraction of that richness. Round-trip cost ~ spread_pct of
            # premium; richness ~ (iv_rv - 1). Need cost << richness.
            richness = max(0.0, m["iv_rv"] - 1.0)
            tradable = m["spread_pct"] < 0.5 * richness and m["spread_pct"] < 0.15
            m.update(sym=sym, group=group, tradable=tradable)
            rows.append(m)
            print(f"{sym:6}{group:22}{m['spread_pct']*100:>8.0f}%{m['iv_rv']:>7.2f}"
                  f"{m['iv']*100:>5.0f}%{m['rv']*100:>5.0f}%{m['liq']:>9,.0f}"
                  f"{'  YES' if tradable else '   no':>11}", flush=True)

    if rows:
        cands = [r for r in rows if r["tradable"]]
        print("\n" + "=" * 60)
        print(f"{len(cands)} of {len(rows)} names pass the tradable-richness screen "
              f"(rich premium AND spread < half the richness, < 15%).")
        if cands:
            print("Candidates worth a historical/OOS study:")
            for r in sorted(cands, key=lambda r: r["iv_rv"], reverse=True):
                print(f"  {r['sym']:6} IV/RV {r['iv_rv']:.2f}  spread {r['spread_pct']*100:.0f}%  "
                      f"liq {r['liq']:,.0f}")
        else:
            print("None. As expected, less-watched names' spreads tend to eat the premium.")
        # average spread by group — the core liquidity/efficiency tradeoff
        print("\nAvg ATM spread by group (the friction gradient):")
        for g in GROUPS:
            gr = [r for r in rows if r["group"] == g]
            if gr:
                print(f"  {g:22} spread {np.mean([r['spread_pct'] for r in gr])*100:4.0f}%  "
                      f"IV/RV {np.mean([r['iv_rv'] for r in gr]):.2f}  "
                      f"liq {np.mean([r['liq'] for r in gr]):,.0f}")
    print("\nSnapshot screen only (today). Survivors would still need a full "
          "historical + out-of-sample test before being believed.")


if __name__ == "__main__":
    main()
