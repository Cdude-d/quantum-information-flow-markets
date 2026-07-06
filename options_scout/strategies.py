"""Candidate trade construction, filtering, and scoring.

Structures generated per ticker, gated by signals:
- Put/call credit spread  — premium-selling, directional lean
- Iron condor             — premium-selling, range-bound lean
- Call/put debit spread   — directional, used when premium is NOT rich

Sign convention: `net` is per-share cash at entry (+credit / -debit), so for
every structure  P/L per share at settle = net + leg_payoff_at(settle, legs)
and             EV per share (model)     = net + structure_model_value(legs).
"""

import logging
import math

import numpy as np
import pandas as pd

from . import pricing

log = logging.getLogger("options_scout")


# --- Chain preparation ------------------------------------------------------

def prep_side(df: pd.DataFrame, kind: str, spot: float, t: float, r: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("bid", "ask", "volume", "openInterest", "impliedVolatility"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[(out["bid"].fillna(0) > 0) | (out["ask"].fillna(0) > 0)]
    out["mid"] = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2.0
    out["spread_pct"] = np.where(out["mid"] > 0,
                                 (out["ask"].fillna(0) - out["bid"].fillna(0)) / out["mid"],
                                 np.inf)
    out["liq"] = out[["openInterest", "volume"]].fillna(0).max(axis=1)

    # Yahoo IV is unreliable at the tails; re-solve from mid when it looks wrong.
    ivs = []
    for _, row in out.iterrows():
        iv = row["impliedVolatility"]
        if not (0.005 < (iv or 0) < 4.0):
            iv = pricing.implied_vol(kind, float(row["mid"]), spot, float(row["strike"]), t, r)
        ivs.append(iv)
    out["iv"] = ivs
    out = out.dropna(subset=["iv"])
    out["delta_abs"] = [
        abs(pricing.bs_delta(kind, spot, float(k), t, float(iv), r))
        for k, iv in zip(out["strike"], out["iv"])
    ]
    return out.sort_values("strike").reset_index(drop=True)


def leg_ok(row: pd.Series, cfg: dict, strict: bool) -> bool:
    """Liquidity gate. Short (premium-collecting) legs are strict; long wings
    bought for pennies get a relaxed check so condors aren't rejected."""
    if strict:
        return (row["mid"] >= cfg["min_leg_mid"]
                and row["spread_pct"] <= cfg["max_bid_ask_spread_pct"]
                and row["liq"] >= cfg["min_open_interest_or_volume"])
    if row["ask"] <= 0:
        return False
    if row["mid"] >= 0.10 and row["spread_pct"] > max(0.5, cfg["max_bid_ask_spread_pct"] * 2):
        return False
    return row["liq"] >= cfg["min_open_interest_or_volume"] or row["ask"] <= 0.10


def _nearest_by_delta(df: pd.DataFrame, target: float, cfg: dict) -> pd.Series | None:
    pool = df[[leg_ok(r, cfg, strict=True) for _, r in df.iterrows()]]
    if pool.empty:
        return None
    return pool.iloc[(pool["delta_abs"] - target).abs().argmin()]


def _nearest_strike(df: pd.DataFrame, price_target: float, exclude: float,
                    side: str, cfg: dict) -> pd.Series | None:
    pool = df[df["strike"] < exclude] if side == "below" else df[df["strike"] > exclude]
    pool = pool[[leg_ok(r, cfg, strict=False) for _, r in pool.iterrows()]]
    if pool.empty:
        return None
    return pool.iloc[(pool["strike"] - price_target).abs().argmin()]


def _strike_increment(df: pd.DataFrame, spot: float) -> float:
    near = df[(df["strike"] > spot * 0.95) & (df["strike"] < spot * 1.05)]["strike"]
    diffs = near.sort_values().diff().dropna()
    return float(diffs.median()) if len(diffs) else 1.0


def atm_metrics(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> dict:
    """ATM implied vol and straddle-based expected move."""
    ivs, em = [], None
    try:
        c = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        p = puts.iloc[(puts["strike"] - spot).abs().argmin()]
        ivs = [v for v in (c.get("iv"), p.get("iv")) if v and 0.005 < v < 4.0]
        if c["mid"] > 0 and p["mid"] > 0:
            em = float(c["mid"] + p["mid"])
    except Exception:  # noqa: BLE001
        pass
    return {"atm_iv": float(np.mean(ivs)) if ivs else None, "expected_move": em}


# --- Structure builders -----------------------------------------------------

def _legs_meta(rows: list[tuple[str, str, pd.Series]]) -> list[dict]:
    return [{
        "side": side, "type": kind, "k": float(r["strike"]),
        "mid": float(r["mid"]), "bid": float(r["bid"] or 0), "ask": float(r["ask"] or 0),
        "iv": float(r["iv"]), "delta": float(r["delta_abs"]), "liq": int(r["liq"]),
    } for side, kind, r in rows]


def build_credit_spread(kind: str, side_df: pd.DataFrame, spot: float, t: float,
                        cfg: dict, realized_vol: float) -> dict | None:
    """kind='put' -> bullish put credit spread; kind='call' -> bearish.

    Tries the configured short delta first; if the spread can't collect enough
    credit there (common on the call side when skew is steep), steps the short
    strike closer to the money. Lower POP at higher delta is priced in honestly.
    """
    for bump in (0.0, 0.07, 0.12):
        cand = _try_credit_spread(kind, side_df, spot, t, cfg, realized_vol,
                                  cfg["target_short_delta"] + bump)
        if cand is not None:
            return cand
    return None


def _try_credit_spread(kind: str, side_df: pd.DataFrame, spot: float, t: float,
                       cfg: dict, realized_vol: float, short_delta: float) -> dict | None:
    r = cfg["risk_free_rate"]
    short = _nearest_by_delta(side_df, short_delta, cfg)
    if short is None:
        return None
    ks = float(short["strike"])
    if (kind == "put" and ks >= spot) or (kind == "call" and ks <= spot):
        return None
    inc = _strike_increment(side_df, spot)
    width_target = max(inc, spot * cfg["spread_width_pct"])
    direction = "below" if kind == "put" else "above"
    sign = -1 if kind == "put" else 1
    long = _nearest_strike(side_df, ks + sign * width_target, ks, direction, cfg)
    if long is None:
        return None
    kl = float(long["strike"])
    width = abs(ks - kl)
    credit = float(short["mid"] - long["mid"])
    if width <= 0 or credit <= 0 or credit / width < cfg["min_credit_to_width"]:
        return None

    legs = _legs_meta([("sell", kind, short), ("buy", kind, long)])
    iv = float(short["iv"])
    model_vol = blend_vol(iv, realized_vol, cfg)
    if kind == "put":
        breakeven = ks - credit
        pop_mkt = pricing.prob_above(spot, breakeven, t, iv, r)
        pop_mod = pricing.prob_above(spot, breakeven, t, model_vol, r)
        name = "Put Credit Spread"
    else:
        breakeven = ks + credit
        pop_mkt = pricing.prob_below(spot, breakeven, t, iv, r)
        pop_mod = pricing.prob_below(spot, breakeven, t, model_vol, r)
        name = "Call Credit Spread"
    ev = (credit + pricing.structure_model_value(legs, spot, t, model_vol, r)) * 100.0
    return _finish(name, legs, net=credit, width=width, max_gain=credit * 100.0,
                   max_loss=(width - credit) * 100.0, breakevens=[breakeven],
                   pop_market=pop_mkt, pop_model=pop_mod, ev_model=ev, model_vol=model_vol)


def build_iron_condor(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, t: float,
                      cfg: dict, realized_vol: float) -> dict | None:
    r = cfg["risk_free_rate"]
    ps = _nearest_by_delta(puts, cfg["target_short_delta"], cfg)
    cs = _nearest_by_delta(calls, cfg["target_short_delta"], cfg)
    if ps is None or cs is None:
        return None
    kps, kcs = float(ps["strike"]), float(cs["strike"])
    if not (kps < spot < kcs):
        return None
    inc = _strike_increment(puts, spot)
    width_target = max(inc, spot * cfg["spread_width_pct"])
    pl = _nearest_strike(puts, kps - width_target, kps, "below", cfg)
    cl = _nearest_strike(calls, kcs + width_target, kcs, "above", cfg)
    if pl is None or cl is None:
        return None
    credit = float(ps["mid"] - pl["mid"] + cs["mid"] - cl["mid"])
    width = max(kps - float(pl["strike"]), float(cl["strike"]) - kcs)
    if credit <= 0 or credit / width < cfg["min_credit_to_width"]:
        return None

    legs = _legs_meta([("sell", "put", ps), ("buy", "put", pl),
                       ("sell", "call", cs), ("buy", "call", cl)])
    iv = float((ps["iv"] + cs["iv"]) / 2.0)
    model_vol = blend_vol(iv, realized_vol, cfg)
    lo, hi = kps - credit, kcs + credit
    pop_mkt = pricing.prob_between(spot, lo, hi, t, iv, r)
    pop_mod = pricing.prob_between(spot, lo, hi, t, model_vol, r)
    ev = (credit + pricing.structure_model_value(legs, spot, t, model_vol, r)) * 100.0
    return _finish("Iron Condor", legs, net=credit, width=width, max_gain=credit * 100.0,
                   max_loss=(width - credit) * 100.0, breakevens=[lo, hi],
                   pop_market=pop_mkt, pop_model=pop_mod, ev_model=ev, model_vol=model_vol)


def build_debit_spread(kind: str, side_df: pd.DataFrame, spot: float, t: float,
                       cfg: dict, realized_vol: float) -> dict | None:
    """kind='call' -> bullish call debit spread; kind='put' -> bearish."""
    r = cfg["risk_free_rate"]
    buy = _nearest_by_delta(side_df, cfg["target_long_delta"], cfg)
    sell = _nearest_by_delta(side_df, cfg["target_short_delta"], cfg)
    if buy is None or sell is None or buy["strike"] == sell["strike"]:
        return None
    kb, ksell = float(buy["strike"]), float(sell["strike"])
    if (kind == "call" and ksell <= kb) or (kind == "put" and ksell >= kb):
        return None
    debit = float(buy["mid"] - sell["mid"])
    width = abs(ksell - kb)
    if debit <= 0 or debit / width > cfg["max_debit_to_width"]:
        return None

    legs = _legs_meta([("buy", kind, buy), ("sell", kind, sell)])
    iv = float(buy["iv"])
    model_vol = blend_vol(iv, realized_vol, cfg)
    if kind == "call":
        breakeven = kb + debit
        pop_mkt = pricing.prob_above(spot, breakeven, t, iv, r)
        pop_mod = pricing.prob_above(spot, breakeven, t, model_vol, r)
        name = "Call Debit Spread"
    else:
        breakeven = kb - debit
        pop_mkt = pricing.prob_below(spot, breakeven, t, iv, r)
        pop_mod = pricing.prob_below(spot, breakeven, t, model_vol, r)
        name = "Put Debit Spread"
    ev = (-debit + pricing.structure_model_value(legs, spot, t, model_vol, r)) * 100.0
    return _finish(name, legs, net=-debit, width=width, max_gain=(width - debit) * 100.0,
                   max_loss=debit * 100.0, breakevens=[breakeven],
                   pop_market=pop_mkt, pop_model=pop_mod, ev_model=ev, model_vol=model_vol)


def blend_vol(implied: float, realized: float, cfg: dict) -> float:
    w = cfg["model_vol_blend"]
    if not realized or realized < 0.01:
        return implied
    return w * realized + (1 - w) * implied


def _finish(name, legs, *, net, width, max_gain, max_loss, breakevens,
            pop_market, pop_model, ev_model, model_vol) -> dict:
    return {
        "structure": name, "legs": legs, "net": round(net, 2), "width": width,
        "max_gain": max_gain, "max_loss": max_loss,
        "breakevens": [round(b, 2) for b in breakevens],
        "pop_market": pop_market, "pop_model": pop_model,
        "ev_model": ev_model, "ev_per_risk": ev_model / max_loss if max_loss > 0 else 0.0,
        "model_vol": model_vol, "warnings": [],
    }


# --- Generation + scoring ---------------------------------------------------

PREMIUM_SELLERS = {"Put Credit Spread", "Call Credit Spread", "Iron Condor"}


def generate_candidates(symbol: str, mode: str, expiry, spot: float, t: float,
                        calls_raw: pd.DataFrame, puts_raw: pd.DataFrame,
                        sig: dict, bias: dict, cfg: dict,
                        events_before_expiry: list) -> list[dict]:
    r = cfg["risk_free_rate"]
    calls = prep_side(calls_raw, "call", spot, t, r)
    puts = prep_side(puts_raw, "put", spot, t, r)
    if calls.empty and puts.empty:
        return []

    atm = atm_metrics(calls, puts, spot)
    realized = sig["realized_vol"]
    vrp = (atm["atm_iv"] / realized) if (atm["atm_iv"] and realized > 0.01) else None
    b = bias["total"]

    raw: list[dict] = []
    if b >= 0.10:
        raw.append(build_credit_spread("put", puts, spot, t, cfg, realized))
    if b <= -0.10:
        raw.append(build_credit_spread("call", calls, spot, t, cfg, realized))
    if abs(b) <= 0.35:
        raw.append(build_iron_condor(calls, puts, spot, t, cfg, realized))
    if abs(b) >= 0.30 and (vrp is None or vrp <= 1.10):
        kind = "call" if b > 0 else "put"
        side = calls if kind == "call" else puts
        raw.append(build_debit_spread(kind, side, spot, t, cfg, realized))

    out = []
    for cand in filter(None, raw):
        cand.update({
            "symbol": symbol, "mode": mode, "expiry": expiry, "spot": spot,
            "t_years": t, "atm_iv": atm["atm_iv"], "expected_move": atm["expected_move"],
            "vrp_ratio": vrp, "bias": bias, "signals": sig,
        })
        _score(cand, events_before_expiry, cfg)
        out.append(cand)
    return out


def _score(c: dict, events_before_expiry: list, cfg: dict) -> None:
    spreads = [(l["ask"] - l["bid"]) / l["mid"] for l in c["legs"] if l["mid"] > 0.05]
    avg_spread = float(np.mean(spreads)) if spreads else 0.5
    min_liq = min(l["liq"] for l in c["legs"])
    liquidity = 10.0 - min(6.0, avg_spread * 40.0)
    if min_liq < 250:
        liquidity -= 2.0
    c["liquidity_score"] = max(0.0, liquidity)

    b = c["bias"]["total"]
    name = c["structure"]
    if name == "Put Credit Spread" or name == "Call Debit Spread":
        align = max(0.0, b) * 10.0
    elif name == "Call Credit Spread" or name == "Put Debit Spread":
        align = max(0.0, -b) * 10.0
    else:  # Iron Condor
        align = (1.0 - abs(b)) * 10.0
    if name in PREMIUM_SELLERS and c["vrp_ratio"]:
        align *= min(1.25, max(0.6, c["vrp_ratio"]))  # richer premium -> better seller alignment
    c["align_score"] = align

    ev_term = float(np.clip(c["ev_per_risk"], -0.25, 0.25)) * 120.0
    score = 60.0 * c["pop_model"] + ev_term + c["liquidity_score"] + c["align_score"]

    if events_before_expiry:
        labels = ", ".join(f"{d:%a %b %d}: {lbl}" for d, lbl in events_before_expiry)
        c["warnings"].append(f"Scheduled macro event(s) before expiry — {labels}")
        if name in PREMIUM_SELLERS:
            score *= 0.92

    c["score"] = round(score, 1)
    if cfg.get("account_size"):
        budget = cfg["account_size"] * cfg["risk_per_trade_pct"] / 100.0
        c["sizing"] = {"risk_budget": budget,
                       "contracts": int(budget // c["max_loss"]) if c["max_loss"] > 0 else 0}
    else:
        c["sizing"] = None
