"""Replay a stored chain snapshot: regenerate the picks the scanner WOULD have
made from that captured data, then settle them against the real close.

This is the honest "what would have happened if we'd traded" tool. It reuses
the live strategy engine unchanged — no model logic is altered. It can only be
as good as the snapshot: a 9:45 capture replays a 9:45 entry; a midday capture
replays a midday entry.

Usage:
  python replay.py                 # replay today's snapshots
  python replay.py 2026-06-16      # replay a specific date
  python replay.py 2026-06-16 --log  # also append results to a paper-trade log
"""
import argparse
import csv
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from options_scout import scoreboard, strategies
from options_scout.config import load_config
from options_scout.pricing import leg_payoff_at, year_fraction
from options_scout.signals import bias_score, compute_signals, events_between
from options_scout.market_data import get_history

ET = ZoneInfo("America/New_York")
ETF_SET = {"SPY", "QQQ", "IWM", "DIA"}


def load_snapshot(path: Path):
    df = pd.read_csv(path)
    meta = {
        "symbol": df["symbol"].iloc[0],
        "mode": df["mode"].iloc[0],
        "expiry": date.fromisoformat(str(df["expiry"].iloc[0])),
        "spot": float(df["spot"].iloc[0]),
        "snapshot_ts": datetime.fromisoformat(df["snapshot_ts"].iloc[0]),
    }
    calls = df[df["type"] == "call"].reset_index(drop=True)
    puts = df[df["type"] == "put"].reset_index(drop=True)
    return meta, calls, puts


def regenerate(meta, calls, puts, cfg):
    """Run the live engine against the snapshot to get the picks it would emit."""
    snap_dt = meta["snapshot_ts"]
    session_date = snap_dt.date()
    hist = get_history(meta["symbol"], "2y")
    hist = hist[hist.index.date <= session_date]  # only data available at capture
    sig = compute_signals(hist, meta["spot"], session_date)
    bias = bias_score(sig, cfg["bias_weights"])
    events = events_between(session_date, meta["expiry"], cfg["custom_event_dates"])
    t = year_fraction(snap_dt, meta["expiry"])
    return strategies.generate_candidates(
        meta["symbol"], meta["mode"], meta["expiry"], meta["spot"], t,
        calls, puts, sig, bias, cfg, events)


def settle(cand):
    """Grade a candidate against the expiry-day close. Returns (status, settle, pl)."""
    sp = scoreboard._settle_close(cand["symbol"], cand["expiry"])
    if sp is None:
        return "PENDING", None, None
    legs = [{"side": l["side"], "type": l["type"], "k": l["k"]} for l in cand["legs"]]
    pl = (cand["net"] + leg_payoff_at(sp, legs)) * 100.0
    return ("WIN" if pl > 0 else "LOSS"), sp, pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default=None, help="date YYYY-MM-DD (default today)")
    ap.add_argument("--log", action="store_true", help="append to data/replay_paper.csv")
    args = ap.parse_args()

    cfg = load_config()
    target = (date.fromisoformat(args.target) if args.target
              else datetime.now(ET).date())
    snap_dir = Path(cfg["data_dir"]) / "chains" / target.isoformat()
    if not snap_dir.exists():
        print(f"No snapshots stored for {target} ({snap_dir} not found).")
        print("Snapshots are captured starting from each scan run going forward.")
        return

    files = sorted(snap_dir.glob("*.csv"))
    print(f"\n=== REPLAY {target:%a %b %d, %Y} — {len(files)} chain snapshot(s) ===")
    cap_ts = None
    by_mode = {}
    for f in files:
        meta, calls, puts = load_snapshot(f)
        cap_ts = meta["snapshot_ts"]
        for c in regenerate(meta, calls, puts, cfg):
            by_mode.setdefault(c["mode"], []).append(c)

    print(f"Captured at {cap_ts:%H:%M %Z} "
          f"({'open-ish' if cap_ts.time().hour < 11 else 'MIDDAY — not a 9:45 entry'}).\n")

    rows_out = []
    grand = {"n": 0, "wins": 0, "pl": 0.0, "pending": 0}
    for mode in ("eod", "eow"):
        cands = sorted(by_mode.get(mode, []), key=lambda c: c["score"], reverse=True)
        # mirror the live scan: dedup + top N
        seen, picks = set(), []
        for c in cands:
            key = (c["symbol"], c["structure"], tuple(l["k"] for l in c["legs"]))
            if key in seen:
                continue
            seen.add(key)
            picks.append(c)
            if len(picks) >= cfg["top_n_per_mode"]:
                break
        if not picks:
            continue
        label = "END OF DAY (0DTE)" if mode == "eod" else "END OF WEEK"
        print(f"--- {label} ---")
        for c in picks:
            status, sp, pl = settle(c)
            strikes = "/".join(f"{l['k']:g}" for l in c["legs"])
            net = f"{'cr' if c['net'] >= 0 else 'db'} {abs(c['net']):.2f}"
            tail = (f"settle {sp:.2f} -> {pl:+.0f}" if pl is not None
                    else f"expires {c['expiry']:%b %d} (pending)")
            print(f"  {c['symbol']:5} {c['structure']:18} {strikes:20} ({net}) "
                  f"POP {c['pop_model']*100:.0f}%  [{status}] {tail}")
            grand["n"] += 1
            if pl is None:
                grand["pending"] += 1
            else:
                grand["wins"] += int(pl > 0)
                grand["pl"] += pl
            rows_out.append({
                "replay_date": target.isoformat(), "captured": cap_ts.isoformat(),
                "mode": mode, "symbol": c["symbol"], "structure": c["structure"],
                "strikes": strikes, "net": round(c["net"], 2),
                "pop_model": round(c["pop_model"], 4), "expiry": c["expiry"].isoformat(),
                "status": status, "settle": "" if sp is None else round(sp, 2),
                "pl": "" if pl is None else round(pl, 2),
            })
        print()

    settled = grand["n"] - grand["pending"]
    print("=" * 56)
    if settled:
        print(f"Settled {settled} trade(s): {grand['wins']} win / "
              f"{settled - grand['wins']} loss, net {grand['pl']:+,.0f} per 1-lot.")
        was_good = grand["pl"] < 0
        verdict = ("Not trading AVOIDED a net loss." if was_good
                   else "Not trading MISSED a net gain.")
        print(f"Verdict for this capture: {verdict}")
    if grand["pending"]:
        exp = max(date.fromisoformat(r["expiry"]) for r in rows_out)
        print(f"{grand['pending']} trade(s) still pending — settle after {exp:%a %b %d} close. "
              f"Re-run replay then for the real result.")

    if args.log and rows_out:
        path = Path(cfg["data_dir"]) / "replay_paper.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows_out)
        print(f"Logged {len(rows_out)} paper trade(s) to {path.name}")


if __name__ == "__main__":
    main()
