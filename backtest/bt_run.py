"""Run the end-of-week strategy over downloaded Databento snapshots.

Feeds each historical chain through the LIVE strategy engine (no model changes),
settles every pick at the real expiry-day close (free yfinance), and reports
win rate / P/L — the honest historical validation.

Key backtest assumptions (stated, not hidden):
- Entry at the captured ~10:00 ET snapshot, mid-price fills.
- Spot is derived from the chain via put-call parity (self-consistent with the
  quotes; historical intraday underlying prices aren't freely available).
- Open-interest liquidity gate is relaxed (cbbo has no OI); the bid/ask-spread
  and min-premium gates still apply, so only quotable spreads are taken.
- Held to expiry, settled at the official close. No intraday management.

Usage:
  python backtest/bt_run.py          # all downloaded weeks
"""
import re
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout import market_data, scoreboard, strategies  # noqa: E402
from options_scout.config import load_config  # noqa: E402
from options_scout.pricing import leg_payoff_at, year_fraction  # noqa: E402
from options_scout.signals import bias_score, compute_signals, events_between  # noqa: E402

import databento as db  # noqa: E402

ET = ZoneInfo("America/New_York")
ETF_SET = {"SPY", "QQQ", "IWM", "DIA"}
CHAINS = Path(__file__).resolve().parent / "data" / "chains"
OSI = re.compile(r"^(?P<root>[A-Z]+)\s+(?P<d>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")

_hist_cache: dict = {}
_earn_cache: dict = {}


def backtest_cfg(blend: float | None = None) -> dict:
    cfg = load_config()
    cfg["min_open_interest_or_volume"] = 0   # no OI in cbbo; rely on spread gate
    if blend is not None:
        cfg["model_vol_blend"] = blend
    return cfg


def load_week(path: Path):
    entry_s, exp_s = path.name.replace(".dbn.zst", "").split("_")
    entry, exp = date.fromisoformat(entry_s), date.fromisoformat(exp_s)
    df = db.DBNStore.from_file(path).to_df().reset_index(drop=True)
    df = df.drop_duplicates("symbol", keep="first").reset_index(drop=True)
    ext = df["symbol"].str.extract(OSI)
    df = pd.concat([df, ext], axis=1).dropna(subset=["root"])
    df["exp"] = pd.to_datetime(df["d"], format="%y%m%d").dt.date
    df["strike"] = df["k"].astype(int) / 1000.0
    df["otype"] = df["cp"].map({"C": "call", "P": "put"})
    df = df[df["exp"] == exp]
    return entry, exp, df


def side_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "strike": rows["strike"].astype(float),
        "bid": pd.to_numeric(rows["bid_px_00"], errors="coerce"),
        "ask": pd.to_numeric(rows["ask_px_00"], errors="coerce"),
        "volume": pd.to_numeric(rows["bid_sz_00"], errors="coerce").fillna(0)
        + pd.to_numeric(rows["ask_sz_00"], errors="coerce").fillna(0),
    })
    out["openInterest"] = 0
    out["impliedVolatility"] = float("nan")  # prep_side re-solves from mid
    return out.dropna(subset=["bid", "ask"]).reset_index(drop=True)


def spot_from_parity(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    c = calls.assign(cmid=(calls.bid + calls.ask) / 2)[lambda d: (d.bid > 0) & (d.ask > 0)]
    p = puts.assign(pmid=(puts.bid + puts.ask) / 2)[lambda d: (d.bid > 0) & (d.ask > 0)]
    m = c[["strike", "cmid"]].merge(p[["strike", "pmid"]], on="strike")
    if m.empty:
        return None
    m = m.assign(diff=(m.cmid - m.pmid).abs())
    row = m.loc[m["diff"].idxmin()]
    return float(row.strike + (row.cmid - row.pmid))


def full_hist(symbol: str):
    if symbol not in _hist_cache:
        _hist_cache[symbol] = market_data.get_history(symbol, "3y")
    return _hist_cache[symbol]


def hist_to(symbol: str, entry: date):
    return full_hist(symbol)[full_hist(symbol).index.date <= entry]


def settle_close(symbol: str, expiry: date) -> float | None:
    """Expiry-day close from the cached 3y history (covers the whole backtest,
    unlike scoreboard._settle_close which only pulls 3 months)."""
    h = full_hist(symbol)
    rows = h[h.index.date == expiry]
    return float(rows["Close"].iloc[-1]) if not rows.empty else None


def earnings_in_window(symbol: str, start: date, end: date) -> bool:
    if symbol in ETF_SET:
        return False
    if symbol not in _earn_cache:
        try:
            df = market_data.get_ticker(symbol).get_earnings_dates(limit=40)
            _earn_cache[symbol] = [ts.date() for ts in df.index] if df is not None else []
        except Exception:  # noqa: BLE001
            _earn_cache[symbol] = []
    return any(start <= d <= end for d in _earn_cache[symbol])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="chains folder (default in-sample)")
    ap.add_argument("--blend", type=float, default=None, help="model_vol_blend override")
    ap.add_argument("--quiet", action="store_true", help="suppress per-week lines")
    args = ap.parse_args()
    cfg = backtest_cfg(args.blend)
    chains = Path(args.dir) if args.dir else CHAINS
    files = sorted(chains.glob("*.dbn.zst"))
    if not files:
        print(f"No chains in {chains}. Run bt_download.py first.")
        return
    lbl = "OOS" if args.dir else "in-sample"
    print(f"Backtesting [{lbl}] blend={cfg['model_vol_blend']} — {len(files)} weeks\n",
          flush=True)

    trades = []
    for path in files:
        entry, exp, df = load_week(path)
        now_et = datetime.combine(entry, dtime(10, 0), tzinfo=ET)
        t = year_fraction(now_et, exp)
        week_cands = []
        for sym, g in df.groupby("root"):
            if earnings_in_window(sym, entry, exp):
                continue
            calls = side_frame(g[g.otype == "call"])
            puts = side_frame(g[g.otype == "put"])
            if len(calls) < 5 or len(puts) < 5:
                continue
            spot = spot_from_parity(calls, puts)
            if not spot or spot <= 0:
                continue
            hist = hist_to(sym, entry)
            if len(hist) < 60:
                continue
            sig = compute_signals(hist, spot, entry)
            bias = bias_score(sig, cfg["bias_weights"])
            events = events_between(entry, exp, cfg["custom_event_dates"])
            week_cands += strategies.generate_candidates(
                sym, "eow", exp, spot, t, calls, puts, sig, bias, cfg, events)

        week_cands.sort(key=lambda c: c["score"], reverse=True)
        seen, picks = set(), []
        for c in week_cands:
            kkey = (c["symbol"], c["structure"], tuple(l["k"] for l in c["legs"]))
            if kkey in seen:
                continue
            seen.add(kkey)
            picks.append(c)
            if len(picks) >= cfg["top_n_per_mode"]:
                break

        wins = unsettled = 0
        for c in picks:
            settle = settle_close(c["symbol"], exp)
            if settle is None:
                unsettled += 1
                continue
            legs = [{"side": l["side"], "type": l["type"], "k": l["k"]} for l in c["legs"]]
            payoff = leg_payoff_at(settle, legs)
            pl = (c["net"] + payoff) * 100.0
            # Realistic fill: sold legs fill at the bid, bought legs at the ask
            # (you pay the full spread). c["net"] used mids.
            net_real = sum((l["bid"] if l["side"] == "sell" else -l["ask"]) for l in c["legs"])
            pl_real = (net_real + payoff) * 100.0
            wins += pl > 0
            trades.append({
                "entry": entry.isoformat(), "expiry": exp.isoformat(),
                "symbol": c["symbol"], "structure": c["structure"],
                "strikes": "/".join(f"{l['k']:g}" for l in c["legs"]),
                "net": round(c["net"], 2), "net_real": round(net_real, 2),
                "max_loss": round(c["max_loss"], 2),
                "pop_model": round(c["pop_model"], 4), "spot": round(c["spot"], 2),
                "settle": round(settle, 2), "pl": round(pl, 2),
                "pl_real": round(pl_real, 2), "win": int(pl > 0),
                "win_real": int(pl_real > 0),
            })
        if not args.quiet:
            u = f", {unsettled} UNSETTLED" if unsettled else ""
            print(f"  {entry} exp {exp}: {len(picks)} pick(s), {wins} win{u}", flush=True)

    if not trades:
        print("\nNo settled trades produced.")
        return
    out = pd.DataFrame(trades)
    out_path = CHAINS.parent / f"bt_trades_{lbl}_b{cfg['model_vol_blend']}.csv"
    out.to_csv(out_path, index=False)

    n = len(out)
    wins = int(out["win"].sum())
    pl = out["pl"].sum()
    pl_real = out["pl_real"].sum()
    wins_real = int(out["win_real"].sum())
    risk = out["max_loss"].sum()
    pop = out["pop_model"].mean()
    print("\n" + "=" * 56)
    print(f"BACKTEST RESULT — {n} trades over {out['entry'].nunique()} weeks")
    print(f"  Win rate:        {wins/n*100:.0f}%  ({wins}W / {n-wins}L)")
    print(f"  Predicted POP:   {pop*100:.0f}%  (model said ~{pop*n:.0f} winners, got {wins})")
    print(f"  MID-fill   P/L:  {pl:+,.0f}   ({pl/risk*100:+.1f}% on risk)  [optimistic]")
    print(f"  REAL-fill  P/L:  {pl_real:+,.0f}   ({pl_real/risk*100:+.1f}% on risk)  "
          f"win {wins_real/n*100:.0f}%  [sell@bid/buy@ask]")
    print(f"  Spread cost:     {pl-pl_real:,.0f}  ({(pl-pl_real)/n:.0f}/trade)")
    print(f"  Best / worst:    {out['pl'].max():+.0f} / {out['pl'].min():+.0f}")
    print("=" * 56)
    print("By structure:")
    for name, grp in out.groupby("structure"):
        print(f"  {name:20} {len(grp):3} trades  "
              f"{grp['win'].mean()*100:3.0f}% win  P/L {grp['pl'].sum():+,.0f}")
    print(f"\nTrades saved to {out_path}")
    print("Assumptions: mid fills, parity spot, OI gate relaxed, held to expiry.")


if __name__ == "__main__":
    main()
