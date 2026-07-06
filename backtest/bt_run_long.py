"""Backtest variant: BUY naked long calls/puts in the bias direction.

Reuses the already-downloaded Databento chains (no new spend). For each week,
for each symbol with a directional bias, buy one option (call if bias>0, put if
bias<0) near the target delta, pay the mid, settle at the expiry-day intrinsic.
Takes the top-N highest-conviction (|bias|) names per week.

This does NOT touch the live model — it only reuses its signals to pick a
direction, then expresses it as long premium instead of short spreads.

Usage:
  python backtest/bt_run_long.py             # ATM (~0.50 delta)
  python backtest/bt_run_long.py 0.35        # OTM target delta
"""
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout import strategies  # noqa: E402
from options_scout.pricing import (leg_payoff_at, prob_above, prob_below,  # noqa: E402
                                   year_fraction)
from options_scout.signals import bias_score, compute_signals  # noqa: E402

from bt_run import (  # noqa: E402  (reuse the validated loaders)
    CHAINS, ETF_SET, backtest_cfg, earnings_in_window, full_hist, hist_to,
    load_week, settle_close, side_frame, spot_from_parity)

ET = ZoneInfo("America/New_York")
BIAS_MIN = 0.20          # only buy when there's some directional conviction
TOP_N = 4                # highest-conviction names per week (matches spread bt)


def build_long(kind, prepped, spot, t, cfg, realized, target_delta):
    pool = prepped[(prepped["mid"] >= cfg["min_leg_mid"])
                   & (prepped["spread_pct"] <= max(0.30, cfg["max_bid_ask_spread_pct"]))]
    if pool.empty:
        return None
    row = pool.iloc[(pool["delta_abs"] - target_delta).abs().argmin()]
    k, debit, iv = float(row["strike"]), float(row["mid"]), float(row["iv"])
    mvol = strategies.blend_vol(iv, realized, cfg)
    r = cfg["risk_free_rate"]
    if kind == "call":
        be = k + debit
        pop = prob_above(spot, be, t, mvol, r)
    else:
        be = k - debit
        pop = prob_below(spot, be, t, mvol, r)
    return {
        "structure": f"Long {kind.title()}", "legs": [{"side": "buy", "type": kind, "k": k}],
        "net": -debit, "max_loss": debit * 100.0, "pop_model": pop, "strike": k,
    }


def main():
    target_delta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
    cfg = backtest_cfg()
    r = cfg["risk_free_rate"]
    files = sorted(CHAINS.glob("*.dbn.zst"))
    print(f"Long-options backtest, target delta {target_delta:.2f}, "
          f"{len(files)} weeks\n", flush=True)

    trades = []
    for path in files:
        entry, exp, df = load_week(path)
        now_et = datetime.combine(entry, dtime(10, 0), tzinfo=ET)
        t = year_fraction(now_et, exp)
        cands = []
        for sym, g in df.groupby("root"):
            if earnings_in_window(sym, entry, exp):
                continue
            calls, puts = side_frame(g[g.otype == "call"]), side_frame(g[g.otype == "put"])
            if len(calls) < 5 or len(puts) < 5:
                continue
            spot = spot_from_parity(calls, puts)
            if not spot or spot <= 0:
                continue
            hist = hist_to(sym, entry)
            if len(hist) < 60:
                continue
            sig = compute_signals(hist, spot, entry)
            bias = bias_score(sig, cfg["bias_weights"])["total"]
            if abs(bias) < BIAS_MIN:
                continue
            kind = "call" if bias > 0 else "put"
            raw = calls if kind == "call" else puts
            prepped = strategies.prep_side(raw, kind, spot, t, r)
            built = build_long(kind, prepped, spot, t, cfg, sig["realized_vol"], target_delta)
            if built:
                built.update({"symbol": sym, "spot": spot, "bias": bias})
                cands.append(built)

        cands.sort(key=lambda c: abs(c["bias"]), reverse=True)
        for c in cands[:TOP_N]:
            settle = settle_close(c["symbol"], exp)
            if settle is None:
                continue
            pl = (c["net"] + leg_payoff_at(settle, c["legs"])) * 100.0
            trades.append({
                "entry": entry.isoformat(), "symbol": c["symbol"],
                "structure": c["structure"], "strike": c["strike"],
                "debit": round(-c["net"], 2), "bias": round(c["bias"], 2),
                "pop_model": round(c["pop_model"], 4), "spot": round(c["spot"], 2),
                "settle": round(settle, 2), "pl": round(pl, 2), "win": int(pl > 0),
            })

    if not trades:
        print("No trades produced.")
        return
    out = pd.DataFrame(trades)
    out.to_csv(CHAINS.parent / f"bt_long_{target_delta:.2f}.csv", index=False)
    n = len(out)
    wins = int(out["win"].sum())
    pl = out["pl"].sum()
    risk = out["max_loss"].sum() if "max_loss" in out else out["debit"].sum() * 100
    wins_df, loss_df = out[out.win == 1]["pl"], out[out.win == 0]["pl"]
    aw = wins_df.mean() if len(wins_df) else 0
    al = loss_df.mean() if len(loss_df) else 0
    be = (-al / (aw - al) * 100) if (aw - al) else float("nan")
    print("\n" + "=" * 60)
    print(f"LONG OPTIONS (delta {target_delta:.2f}) — {n} trades / {out['entry'].nunique()} weeks")
    print(f"  Win rate:      {wins/n*100:.0f}%  ({wins}W / {n-wins}L)")
    print(f"  Predicted POP: {out['pop_model'].mean()*100:.0f}%")
    print(f"  Avg win / loss: +{aw:.0f} / {al:.0f}   (breakeven win rate {be:.0f}%)")
    print(f"  Net P/L:       {pl:+,.0f}  on {out['debit'].sum()*100:,.0f} spent")
    print(f"  Return:        {pl/(out['debit'].sum()*100)*100:+.1f}% of premium paid")
    print(f"  Best / worst:  {out['pl'].max():+.0f} / {out['pl'].min():+.0f}")
    print("=" * 60)
    print("Assumptions: mid fills, parity spot, held to expiry (no stops/trims).")


if __name__ == "__main__":
    main()
