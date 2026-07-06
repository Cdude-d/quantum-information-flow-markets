"""Build a shareable summary of scored recommendations over the last N days."""
import csv
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from options_scout import scoreboard
from options_scout.config import load_config

cfg = load_config()
scoreboard.score_pending(cfg)  # settle anything newly expired

days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
cutoff = date.today() - timedelta(days=days)
path = Path(cfg["data_dir"]) / "recommendations.csv"
rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
        if r["status"] == "scored"
        and datetime.fromisoformat(r["run_ts"]).date() >= cutoff]

by_day = defaultdict(list)
for r in rows:
    by_day[datetime.fromisoformat(r["run_ts"]).date()].append(r)

def agg(rs):
    n = len(rs)
    wins = sum(int(r["win"]) for r in rs)
    pl = sum(float(r["pl_per_contract"]) for r in rs)
    risk = sum(float(r["max_loss"]) for r in rs)
    pop = sum(float(r["pop_model"]) for r in rs) / n
    return n, wins, pl, risk, pop

print(f"\n=== SCANNER RESULTS — picks from {cutoff:%b %d} to {date.today():%b %d, %Y} ===")
print("(Held-to-expiry, settled at expiry-day close, no profit-taking. P/L per 1 contract.)\n")
print(f"{'Date':<12}{'Trades':>7}{'Wins':>6}{'WinRate':>9}{'NetP/L':>10}{'AvgPredPOP':>12}")
print("-" * 56)
for d in sorted(by_day):
    n, wins, pl, risk, pop = agg(by_day[d])
    print(f"{d:%a %b %d}".ljust(12) + f"{n:>7}{wins:>6}{wins/n*100:>8.0f}%{pl:>+10.0f}{pop*100:>11.0f}%")

n, wins, pl, risk, pop = agg(rows)
print("-" * 56)
print(f"{'TOTAL':<12}{n:>7}{wins:>6}{wins/n*100:>8.0f}%{pl:>+10.0f}{pop*100:>11.0f}%")
print(f"\nRealized win rate {wins/n*100:.0f}% vs model-predicted {pop*100:.0f}% "
      f"(predicted ~{pop*n:.0f} winners, got {wins}).")
print(f"Total return on risk deployed: {pl/risk*100:+.1f}%  (P/L {pl:+.0f} on {risk:.0f} max risk).")
print(f"Best: {max(rows, key=lambda r: float(r['pl_per_contract']))['symbol']} "
      f"{max(float(r['pl_per_contract']) for r in rows):+.0f} | "
      f"Worst: {min(rows, key=lambda r: float(r['pl_per_contract']))['symbol']} "
      f"{min(float(r['pl_per_contract']) for r in rows):+.0f}")

# open positions still in flight
opens = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] == "open"]
if opens:
    print(f"\nStill open (not yet expired): {len(opens)} picks expiring "
          f"{min(r['expiry'] for r in opens)}–{max(r['expiry'] for r in opens)}.")
