"""Options Morning Scout — scan for end-of-day and end-of-week option trades.

Usage:
  python morning_scan.py                 # both horizons, open HTML report
  python morning_scan.py --mode eow      # weekly trades only
  python morning_scan.py --no-open       # don't launch the browser
  python morning_scan.py --dry-run       # don't log picks to the track record
"""

import argparse
import logging
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from options_scout import (market_data, reasoning, report, scoreboard, snapshots,
                           strategies)
from options_scout.config import load_config
from options_scout.pricing import year_fraction
from options_scout.signals import (bias_score, compute_signals, events_between,
                                   is_trading_day, next_trading_day)

ET = ZoneInfo("America/New_York")
ETF_SET = {"SPY", "QQQ", "IWM", "DIA"}
log = logging.getLogger("options_scout")


def setup_logging(data_dir: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(Path(data_dir) / "scan.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("peewee").setLevel(logging.ERROR)


def resolve_session(now_et: datetime):
    """Trading session this scan targets. After ~15:45 ET (or on a closed day)
    roll forward to the next session."""
    d = now_et.date()
    if is_trading_day(d) and now_et.time() < time(15, 45):
        return d
    return next_trading_day(d + timedelta(days=1))


def resolve_expiry(symbol: str, mode: str, session_date) -> object | None:
    try:
        expiries = market_data.list_expiries(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("%s: expiry list failed (%s)", symbol, e)
        return None
    if mode == "eod":
        return session_date if session_date in expiries else None
    friday = session_date + timedelta(days=4 - session_date.weekday())
    pool = [e for e in expiries if session_date <= e <= friday]
    return max(pool) if pool else None


def scan_symbol(symbol: str, mode: str, session_date, now_et, cfg, cache, notes):
    expiry = resolve_expiry(symbol, mode, session_date)
    if expiry is None:
        if mode == "eod":
            notes.append(f"{symbol}: no same-day expiry listed — skipped for EOD.")
        else:
            notes.append(f"{symbol}: no expiry left this week — skipped for EOW.")
        return []

    if symbol not in cache:
        hist = market_data.get_history(symbol, "2y")
        spot = market_data.get_spot(symbol, hist)
        sig = compute_signals(hist, spot, session_date)
        cache[symbol] = (spot, sig)
    spot, sig = cache[symbol]
    bias = bias_score(sig, cfg["bias_weights"])

    events = events_between(session_date, expiry, cfg["custom_event_dates"])
    if symbol not in ETF_SET:
        earnings = market_data.next_earnings_date(symbol)
        if earnings and session_date <= earnings <= expiry:
            if cfg["skip_earnings_before_expiry"]:
                notes.append(f"{symbol}: earnings {earnings:%b %d} before expiry — skipped "
                             f"(binary event).")
                return []
            events = events + [(earnings, f"{symbol} earnings")]

    calls, puts = market_data.get_chain(symbol, expiry)
    snapshots.save_chain(cfg, symbol, mode, expiry, spot, calls, puts, now_et)
    t = year_fraction(now_et, expiry)
    cands = strategies.generate_candidates(
        symbol, mode, expiry, spot, t, calls, puts, sig, bias, cfg, events)
    log.info("%s [%s] exp %s: %d candidate(s)", symbol, mode, expiry, len(cands))
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(description="Options morning scanner")
    ap.add_argument("--mode", choices=["eod", "eow", "both"], default="both")
    ap.add_argument("--no-open", action="store_true", help="don't open the HTML report")
    ap.add_argument("--dry-run", action="store_true", help="don't log picks to track record")
    ap.add_argument("--config", default=None, help="path to config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["data_dir"])

    now_et = datetime.now(ET)
    session_date = resolve_session(now_et)
    if session_date != now_et.date():
        log.info("Market closed for today's purposes — scanning next session %s", session_date)

    modes = ["eod", "eow"] if args.mode == "both" else [args.mode]
    cache: dict = {}
    notes: list[str] = []
    all_picks: dict[str, list] = {}
    seen_keys: set = set()

    for mode in modes:
        cands = []
        for symbol in cfg[f"universe_{mode}"]:
            try:
                cands.extend(scan_symbol(symbol, mode, session_date, now_et, cfg, cache, notes))
            except Exception as e:  # noqa: BLE001
                log.warning("%s [%s]: scan failed — %s", symbol, mode, e)
                notes.append(f"{symbol}: data error, skipped ({e}).")
        cands.sort(key=lambda c: c["score"], reverse=True)
        picks = []
        for c in cands:
            key = (c["symbol"], c["structure"], tuple(l["k"] for l in c["legs"]), c["expiry"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            picks.append(c)
            if len(picks) >= cfg["top_n_per_mode"]:
                break
        all_picks[mode] = picks

    # Market context from whatever we scanned, plus VIX and this week's calendar.
    vix = market_data.get_vix()
    friday = session_date + timedelta(days=4 - session_date.weekday())
    ctx = {
        "vix": vix,
        "snapshots": {s: cache[s][1] for s in ("SPY", "QQQ") if s in cache},
        "events_this_week": events_between(session_date, friday, cfg["custom_event_dates"]),
    }
    context_text = reasoning.market_context_text(ctx)

    stats = scoreboard.score_pending(cfg)
    flat_picks = [c for mode in modes for c in all_picks.get(mode, [])]
    if flat_picks and not args.dry_run:
        scoreboard.log_picks(flat_picks, cfg)

    reasonings = {id(c): reasoning.build_reasoning(c) for c in flat_picks}
    html = report.render_html(now_et, session_date, context_text, all_picks,
                              reasonings, stats, notes)
    report_path = Path(cfg["report_dir"]) / f"scan_{now_et:%Y-%m-%d_%H%M}.html"
    report_path.write_text(html, encoding="utf-8")

    print()
    print(report.console_summary(session_date, context_text, all_picks, stats, report_path))
    if not args.no_open:
        os.startfile(report_path)  # noqa: S606 - Windows: open in default browser
    return 0


if __name__ == "__main__":
    sys.exit(main())
