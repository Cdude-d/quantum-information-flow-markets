"""Track record: log every recommendation, score it after expiry, report stats.

Scoring assumes the trade is held to expiry and settled at the expiry-day
close — no intraday management. Real results with profit-taking rules will
differ (usually fewer full losses, more small wins).
"""

import csv
import json
import logging
import uuid
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from . import market_data
from .pricing import leg_payoff_at

log = logging.getLogger("options_scout")
ET = ZoneInfo("America/New_York")

FIELDS = [
    "run_id", "run_ts", "mode", "symbol", "structure", "expiry", "legs_json",
    "net", "max_gain", "max_loss", "pop_model", "pop_market", "ev_model",
    "score", "spot_at_rec", "status", "settle_price", "pl_per_contract", "win",
]


def _csv_path(cfg: dict) -> Path:
    return Path(cfg["data_dir"]) / "recommendations.csv"


def log_picks(picks: list[dict], cfg: dict) -> None:
    path = _csv_path(cfg)
    new_file = not path.exists()
    run_id = uuid.uuid4().hex[:8]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for c in picks:
            w.writerow({
                "run_id": run_id,
                "run_ts": datetime.now(ET).isoformat(timespec="seconds"),
                "mode": c["mode"], "symbol": c["symbol"], "structure": c["structure"],
                "expiry": c["expiry"].isoformat(),
                "legs_json": json.dumps([{k: l[k] for k in ("side", "type", "k")}
                                         for l in c["legs"]]),
                "net": round(c["net"], 2), "max_gain": round(c["max_gain"], 2),
                "max_loss": round(c["max_loss"], 2),
                "pop_model": round(c["pop_model"], 4), "pop_market": round(c["pop_market"], 4),
                "ev_model": round(c["ev_model"], 2), "score": c["score"],
                "spot_at_rec": round(c["spot"], 2), "status": "open",
                "settle_price": "", "pl_per_contract": "", "win": "",
            })
    log.info("Logged %d picks to %s", len(picks), path)


def _settle_close(symbol: str, expiry: date) -> float | None:
    try:
        hist = market_data.get_history(symbol, "3mo")
        rows = hist[hist.index.date == expiry]
        if not rows.empty:
            return float(rows["Close"].iloc[-1])
    except Exception as e:  # noqa: BLE001
        log.warning("settle fetch failed %s %s: %s", symbol, expiry, e)
    return None


def score_pending(cfg: dict) -> dict:
    """Score expired open picks in place; return aggregate stats."""
    path = _csv_path(cfg)
    stats = {"n_scored": 0, "wins": 0, "total_pl": 0.0, "sum_r": 0.0, "recent": []}
    if not path.exists():
        return stats

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    now = datetime.now(ET)
    changed = False
    for row in rows:
        if row["status"] != "open":
            continue
        expiry = date.fromisoformat(row["expiry"])
        expired = expiry < now.date() or (
            expiry == now.date() and now.time() >= time(16, 15))
        if not expired:
            continue
        settle = _settle_close(row["symbol"], expiry)
        if settle is None:
            continue
        legs = json.loads(row["legs_json"])
        pl = (float(row["net"]) + leg_payoff_at(settle, legs)) * 100.0
        row.update({
            "status": "scored", "settle_price": f"{settle:.2f}",
            "pl_per_contract": f"{pl:.2f}", "win": "1" if pl > 0 else "0",
        })
        changed = True

    if changed:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    scored = [r for r in rows if r["status"] == "scored"]
    for r in scored:
        pl = float(r["pl_per_contract"])
        stats["n_scored"] += 1
        stats["wins"] += int(r["win"])
        stats["total_pl"] += pl
        max_loss = float(r["max_loss"])
        stats["sum_r"] += pl / max_loss if max_loss > 0 else 0.0
    stats["win_rate"] = stats["wins"] / stats["n_scored"] if stats["n_scored"] else None
    stats["avg_r"] = stats["sum_r"] / stats["n_scored"] if stats["n_scored"] else None
    stats["recent"] = [{
        "date": r["run_ts"][:10], "symbol": r["symbol"], "structure": r["structure"],
        "mode": r["mode"], "expiry": r["expiry"], "pop_model": float(r["pop_model"]),
        "pl": float(r["pl_per_contract"]), "win": r["win"] == "1",
    } for r in scored[-10:]]
    return stats
