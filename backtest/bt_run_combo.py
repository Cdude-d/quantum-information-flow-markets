"""Backtest variant: directional longs + premium-selling by signal strength.

  bias >= +UP    -> long call          (convex up bet)
  bias <= -DOWN  -> long put           (convex down bet)
  otherwise      -> iron condor        (profits when it DOESN'T move)

Trades every qualifying symbol each week. Reuses downloaded chains (no new
spend). Reports per-sleeve P/L and return-on-risk. In-sample over one bull
regime — treat any profit as unproven until validated out-of-sample.

Usage:  python backtest/bt_run_combo.py [up_thresh] [down_thresh]
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

from bt_run import (  # noqa: E402
    CHAINS, backtest_cfg, earnings_in_window, hist_to, load_week,
    settle_close, side_frame, spot_from_parity)

ET = ZoneInfo("America/New_York")
ATM_DELTA = 0.50


def nearest_delta(prepped, target, cfg):
    pool = prepped[(prepped["mid"] >= cfg["min_leg_mid"])
                   & (prepped["spread_pct"] <= max(0.30, cfg["max_bid_ask_spread_pct"]))]
    if pool.empty:
        return None
    return pool.iloc[(pool["delta_abs"] - target).abs().argmin()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--up", type=float, default=0.30)
    ap.add_argument("--down", type=float, default=0.30)
    ap.add_argument("--dir", default=None, help="chains folder (default in-sample)")
    args = ap.parse_args()
    up, down = args.up, args.down
    cfg = backtest_cfg()
    r = cfg["risk_free_rate"]
    chains = Path(args.dir) if args.dir else CHAINS
    files = sorted(chains.glob("*.dbn.zst"))
    label = "OOS" if args.dir else "in-sample"
    print(f"[{label}] {chains}", flush=True)
    print(f"Combo backtest: call if bias>=+{up}, put if <=-{down}, else iron condor. "
          f"{len(files)} weeks\n", flush=True)

    trades = []
    for path in files:
        entry, exp, df = load_week(path)
        t = year_fraction(datetime.combine(entry, dtime(10, 0), tzinfo=ET), exp)
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
            cp = strategies.prep_side(calls, "call", spot, t, r)
            pp = strategies.prep_side(puts, "put", spot, t, r)

            if bias >= up:
                row = nearest_delta(cp, ATM_DELTA, cfg)
                if row is None:
                    continue
                k, debit = float(row["strike"]), float(row["mid"])
                legs = [{"side": "buy", "type": "call", "k": k}]
                net, max_loss, sleeve = -debit, debit * 100.0, "Call"
            elif bias <= -down:
                row = nearest_delta(pp, ATM_DELTA, cfg)
                if row is None:
                    continue
                k, debit = float(row["strike"]), float(row["mid"])
                legs = [{"side": "buy", "type": "put", "k": k}]
                net, max_loss, sleeve = -debit, debit * 100.0, "Put"
            else:
                ic = strategies.build_iron_condor(cp, pp, spot, t, cfg, sig["realized_vol"])
                if ic is None:
                    continue
                legs = [{"side": l["side"], "type": l["type"], "k": l["k"]} for l in ic["legs"]]
                net, max_loss, sleeve = ic["net"], ic["max_loss"], "Condor"

            settle = settle_close(sym, exp)
            if settle is None:
                continue
            pl = (net + leg_payoff_at(settle, legs)) * 100.0
            trades.append({"entry": entry.isoformat(), "symbol": sym, "sleeve": sleeve,
                           "bias": round(bias, 2), "net": round(net, 2),
                           "max_loss": round(max_loss, 2), "settle": round(settle, 2),
                           "pl": round(pl, 2), "win": int(pl > 0)})

    out = pd.DataFrame(trades)
    out.to_csv(CHAINS.parent / "bt_combo.csv", index=False)
    n, wins, pl = len(out), int(out["win"].sum()), out["pl"].sum()
    risk = out["max_loss"].sum()
    print("=" * 62)
    print(f"COMBO — {n} trades / {out['entry'].nunique()} weeks")
    print(f"  Win rate {wins/n*100:.0f}%  | net ${pl:+,.0f} | return on risk "
          f"{pl/risk*100:+.1f}%  (vs +1.2% for condor-only)")
    print(f"  median trade ${out['pl'].median():.0f} | best ${out['pl'].max():+,.0f} "
          f"worst ${out['pl'].min():+,.0f}")
    print("By sleeve (return on risk = net / capital at risk):")
    for s, gg in out.groupby("sleeve"):
        ror = gg["pl"].sum() / gg["max_loss"].sum() * 100 if gg["max_loss"].sum() else 0
        print(f"  {s:8} {len(gg):3} trades  {gg['win'].mean()*100:3.0f}% win  "
              f"net ${gg['pl'].sum():+7,.0f}  RoR {ror:+5.1f}%")
    wk = out.groupby("entry")["pl"].sum()
    top3 = out.nlargest(3, "pl")["pl"].sum()
    print(f"Outlier check: top 3 trades = ${top3:,.0f} of ${pl:,.0f} net; "
          f"profitable weeks {(wk>0).sum()}/{len(wk)}")
    big = out.nlargest(5, "pl")["sleeve"].value_counts().to_dict()
    print(f"  top-5 winners by sleeve: {big}  (all-call => regime bet, not edge)")
    print("=" * 62)
    print("In-sample, one bull regime. Unproven until validated out-of-sample.")


if __name__ == "__main__":
    main()
