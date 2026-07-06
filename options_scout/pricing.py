"""Black-Scholes pricing, implied vol, and probability estimates.

Probabilities here are lognormal-model estimates, not guarantees. "Market"
probabilities use the option's implied vol (what the market is pricing);
"model" probabilities use a blend of implied and recent realized vol.
"""

import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from scipy.optimize import brentq
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
MIN_SIG_T = 1e-5


def year_fraction(now_et: datetime, expiry: date) -> float:
    """ACT/365 time to 16:00 ET on the expiry date, floored at ~9 minutes."""
    exp_dt = datetime.combine(expiry, time(16, 0), tzinfo=ET)
    seconds = (exp_dt - now_et).total_seconds()
    return max(seconds, 540.0) / (365.0 * 24 * 3600)


def _d1_d2(s: float, k: float, t: float, sigma: float, r: float, q: float = 0.0):
    sig_t = max(sigma * math.sqrt(t), MIN_SIG_T)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sig_t
    return d1, d1 - sig_t


def bs_price(kind: str, s: float, k: float, t: float, sigma: float,
             r: float = 0.0, q: float = 0.0) -> float:
    d1, d2 = _d1_d2(s, k, t, sigma, r, q)
    if kind == "call":
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


def bs_delta(kind: str, s: float, k: float, t: float, sigma: float,
             r: float = 0.0, q: float = 0.0) -> float:
    d1, _ = _d1_d2(s, k, t, sigma, r, q)
    if kind == "call":
        return math.exp(-q * t) * norm.cdf(d1)
    return -math.exp(-q * t) * norm.cdf(-d1)


def implied_vol(kind: str, price: float, s: float, k: float, t: float,
                r: float = 0.0) -> float | None:
    intrinsic = max(0.0, s - k) if kind == "call" else max(0.0, k - s)
    if price <= intrinsic + 1e-4:
        return None
    try:
        return brentq(lambda sig: bs_price(kind, s, k, t, sig, r) - price,
                      1e-3, 8.0, xtol=1e-5)
    except ValueError:
        return None


def prob_above(s: float, k: float, t: float, sigma: float, r: float = 0.0) -> float:
    """P(S_T > K) under lognormal dynamics with drift r."""
    _, d2 = _d1_d2(s, k, t, sigma, r)
    return float(norm.cdf(d2))


def prob_below(s: float, k: float, t: float, sigma: float, r: float = 0.0) -> float:
    return 1.0 - prob_above(s, k, t, sigma, r)


def prob_between(s: float, lo: float, hi: float, t: float, sigma: float,
                 r: float = 0.0) -> float:
    return max(0.0, prob_above(s, lo, t, sigma, r) - prob_above(s, hi, t, sigma, r))


def leg_payoff_at(settle: float, legs: list[dict]) -> float:
    """Intrinsic payoff per share of a leg list at settlement.

    Leg dict: {"side": "buy"|"sell", "type": "call"|"put", "k": strike}.
    Excludes the entry cash flow.
    """
    total = 0.0
    for leg in legs:
        intr = max(0.0, settle - leg["k"]) if leg["type"] == "call" else max(0.0, leg["k"] - settle)
        total += intr if leg["side"] == "buy" else -intr
    return total


def structure_model_value(legs: list[dict], s: float, t: float, sigma: float,
                          r: float = 0.0) -> float:
    """Theoretical value per share of the leg list under the given vol."""
    total = 0.0
    for leg in legs:
        v = bs_price(leg["type"], s, leg["k"], t, sigma, r)
        total += v if leg["side"] == "buy" else -v
    return total
