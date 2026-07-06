"""Rule-based qualitative reasoning: turns computed signals into the written
case for (and against) each candidate trade. Every claim cites a number the
scanner actually computed — no vibes."""


def _pct(x, digits=1):
    return f"{x:+.{digits}f}%" if x is not None else "n/a"


def _money(x):
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def setup_paragraph(c: dict) -> str:
    s, sig, b = c["spot"], c["signals"], c["bias"]
    bits = [f"{c['symbol']} is trading at {s:,.2f}"]
    if sig.get("gap_pct") is not None:
        if sig.get("since_open_pct") is not None:
            bits.append(f"after opening {_pct(sig['gap_pct'])} vs yesterday's close "
                        f"({_pct(sig['since_open_pct'])} since the open)")
        else:
            bits.append(f"{_pct(sig['gap_pct'])} vs the last close")
    trend_txt = {
        "up": "The trend is up — price above rising 20- and 50-day EMAs",
        "down": "The trend is down — price below falling 20- and 50-day EMAs",
        "mixed": "The trend is mixed — price tangled with its 20/50-day EMAs",
    }[sig["trend"]]
    parts = [", ".join(bits) + ".", trend_txt + ".",
             f"Short-term, RSI(2) sits at {sig['rsi2']:.0f} "
             f"and 5-day momentum is {_pct(sig['ret_5d_pct'])}."]

    gs = sig.get("gap_stats", {})
    gap = sig.get("gap_pct")
    if gap is not None and abs(gap) >= gs.get("min_gap", 0.002) * 100:
        if gap > 0 and gs.get("up_fill_rate") is not None:
            parts.append(f"Over the past two years, opening gaps up this size filled the same "
                         f"day {gs['up_fill_rate']*100:.0f}% of the time ({gs['n_up']} cases).")
        elif gap < 0 and gs.get("down_fill_rate") is not None:
            parts.append(f"Over the past two years, opening gaps down this size filled the same "
                         f"day {gs['down_fill_rate']*100:.0f}% of the time ({gs['n_down']} cases).")
    parts.append(f"Net directional bias score: {b['total']:+.2f} "
                 f"(trend {b['trend']:+.1f}, mean-reversion {b['meanrev']:+.1f}, "
                 f"momentum {b['momentum']:+.1f}).")
    return " ".join(parts)


def vol_paragraph(c: dict) -> str:
    sig = c["signals"]
    parts = []
    if c["expected_move"]:
        em_pct = c["expected_move"] / c["spot"] * 100
        horizon = "the session close" if c["mode"] == "eod" else "expiry"
        parts.append(f"The at-the-money straddle prices a ±{em_pct:.2f}% move "
                     f"(±{c['expected_move']:.2f} pts) by {horizon}.")
    if c["atm_iv"]:
        parts.append(f"ATM implied vol is {c['atm_iv']*100:.0f}% vs {sig['realized_vol']*100:.0f}% "
                     f"recent realized (EWMA).")
        if c["vrp_ratio"]:
            if c["vrp_ratio"] >= 1.15:
                parts.append(f"Premium is rich (IV/RV {c['vrp_ratio']:.2f}) — the odds tilt "
                             f"toward selling options rather than buying them.")
            elif c["vrp_ratio"] <= 0.95:
                parts.append(f"Premium is cheap (IV/RV {c['vrp_ratio']:.2f}) — buying options "
                             f"is unusually attractive relative to how much this name moves.")
            else:
                parts.append(f"Premium is roughly fair (IV/RV {c['vrp_ratio']:.2f}).")
    parts.append(f"Average daily range over the last month: {sig['avg_daily_range_pct']:.2f}%.")
    return " ".join(parts)


def trade_paragraph(c: dict) -> str:
    legs_txt = " / ".join(
        f"{l['side'].upper()} {l['k']:g} {l['type']}" for l in c["legs"])
    if c["net"] >= 0:
        cash = f"collecting ≈ {_money(c['net'] * 100)} credit per contract"
        win_cond = _win_condition(c)
    else:
        cash = f"paying ≈ {_money(-c['net'] * 100)} per contract"
        win_cond = _win_condition(c)
    return (f"The trade: {legs_txt} (exp {c['expiry']:%a %b %d}), {cash}. "
            f"{win_cond} Model probability of profit ≈ {c['pop_model']*100:.0f}% "
            f"(market-implied ≈ {c['pop_market']*100:.0f}%); model expected value "
            f"≈ {_money(c['ev_model'])} per contract against {_money(c['max_loss'])} max risk.")


def _win_condition(c: dict) -> str:
    s = c["symbol"]
    be = c["breakevens"]
    n = c["structure"]
    if n == "Put Credit Spread":
        return (f"You keep the full credit if {s} closes above {c['legs'][0]['k']:g} at expiry; "
                f"breakeven {be[0]:g} ({(be[0]/c['spot']-1)*100:+.1f}% from here).")
    if n == "Call Credit Spread":
        return (f"You keep the full credit if {s} closes below {c['legs'][0]['k']:g} at expiry; "
                f"breakeven {be[0]:g} ({(be[0]/c['spot']-1)*100:+.1f}% from here).")
    if n == "Iron Condor":
        return (f"Max profit if {s} closes between the short strikes; profitable between "
                f"{be[0]:g} and {be[1]:g} ({(be[0]/c['spot']-1)*100:+.1f}% / "
                f"{(be[1]/c['spot']-1)*100:+.1f}%).")
    direction = "above" if n == "Call Debit Spread" else "below"
    return (f"Profitable if {s} closes {direction} {be[0]:g} at expiry "
            f"({(be[0]/c['spot']-1)*100:+.1f}% from here); max gain at the short strike.")


def risk_paragraph(c: dict) -> str:
    parts = [f"Max loss is {_money(c['max_loss'])} per contract — defined and known up front."]
    for w in c["warnings"]:
        parts.append(f"⚠ {w}.")
    if c["structure"] in ("Put Credit Spread", "Call Credit Spread", "Iron Condor"):
        parts.append("Standard management: take profit at 50–65% of the credit, cut the "
                     "position if the loss reaches ~2× the credit received, and don't hold "
                     "a threatened short strike into the final hour.")
    else:
        parts.append("Standard management: consider taking profit at 50–100% gain; debit "
                     "spreads lose to time decay if the move doesn't come.")
    if c["sizing"] and c["sizing"]["contracts"] >= 0:
        parts.append(f"At your configured risk budget ({_money(c['sizing']['risk_budget'])} "
                     f"per trade), that's {c['sizing']['contracts']} contract(s).")
    return " ".join(parts)


def build_reasoning(c: dict) -> dict:
    return {
        "setup": setup_paragraph(c),
        "volatility": vol_paragraph(c),
        "trade": trade_paragraph(c),
        "risk": risk_paragraph(c),
    }


def market_context_text(ctx: dict) -> str:
    parts = []
    vix = ctx.get("vix") or {}
    if vix.get("level"):
        regime = ("calm" if vix["level"] < 14 else
                  "normal" if vix["level"] < 20 else
                  "elevated" if vix["level"] < 28 else "stressed")
        parts.append(f"VIX {vix['level']:.1f} ({_pct(vix.get('change_pct'))} vs yesterday) — "
                     f"{regime} volatility regime.")
    for sym in ("SPY", "QQQ"):
        snap = ctx.get("snapshots", {}).get(sym)
        if snap and snap.get("gap_pct") is not None:
            parts.append(f"{sym} {_pct(snap['gap_pct'])} into the session.")
    events = ctx.get("events_this_week") or []
    if events:
        ev_txt = "; ".join(f"{d:%a %b %d} — {lbl}" for d, lbl in events)
        parts.append(f"Scheduled events this week: {ev_txt}.")
    else:
        parts.append("No major scheduled macro events through Friday.")
    return " ".join(parts)
