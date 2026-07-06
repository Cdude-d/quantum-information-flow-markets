"""Configuration: defaults merged with config.json in the project root."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    # Tickers scanned for same-day (0DTE) trades. Only symbols that actually
    # list an expiry for the session date are used; others are skipped.
    "universe_eod": ["SPY", "QQQ", "IWM"],
    # Tickers scanned for end-of-week trades.
    "universe_eow": [
        "SPY", "QQQ", "IWM", "DIA",
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX",
    ],
    "risk_free_rate": 0.04,
    # Weight on EWMA realized vol when blending with implied vol for the
    # "model" probability. 0 = trust the market fully, 1 = trust history fully.
    "model_vol_blend": 0.5,
    "target_short_delta": 0.18,   # short strike target for credit spreads / condor wings
    "target_long_delta": 0.38,    # bought strike target for debit spreads
    "spread_width_pct": 0.008,    # target vertical width as fraction of spot
    "min_open_interest_or_volume": 100,
    "max_bid_ask_spread_pct": 0.18,  # reject legs with wider relative bid/ask
    "min_leg_mid": 0.05,
    "min_credit_to_width": 0.18,  # credit spreads must collect >= 18% of width
    "max_debit_to_width": 0.65,   # debit spreads must cost <= 65% of width
    "top_n_per_mode": 4,
    "skip_earnings_before_expiry": True,
    # Set to a dollar amount (e.g. 25000) to get contract-count suggestions.
    "account_size": None,
    "risk_per_trade_pct": 1.0,
    # Extra event dates to flag, e.g. ["2026-07-15: CPI release"]
    "custom_event_dates": [],
    "bias_weights": {"trend": 0.40, "meanrev": 0.30, "momentum": 0.30},
    "report_dir": "reports",
    "data_dir": "data",
}


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.json"
    cfg = dict(DEFAULTS)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    else:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, indent=2)
    cfg["report_dir"] = str(PROJECT_ROOT / cfg["report_dir"])
    cfg["data_dir"] = str(PROJECT_ROOT / cfg["data_dir"])
    Path(cfg["report_dir"]).mkdir(exist_ok=True)
    Path(cfg["data_dir"]).mkdir(exist_ok=True)
    return cfg
