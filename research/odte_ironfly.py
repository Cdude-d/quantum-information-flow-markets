"""Defined-risk version: 0DTE SPY iron butterfly held to expiry, 2018-2026.

  Sell the 9:50 ATM call + put at strike K (as in odte_to_close.py), buy a call at K + w and a
  put at K - w, with w = round(MULT x the 9:50 ATM straddle mid), MULT in {1.0, 1.5, 2.0}.
  Max loss = (w - credit) x 100 = the capital a normal account must post.
  Unhedged and delta-hedged every 5 min (BS deltas at each leg's entry IV); settle at the
  4 PM close. Real fills: ATM legs at the bid, wings at the ask, $0.65 x 4 contracts.

  python research/odte_ironfly.py download --estimate   # free: resolve strikes + price data
  python research/odte_ironfly.py download              # fetch wing quotes (credits)
  python research/odte_ironfly.py backtest
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import databento as db
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from options_scout.envfile import load_env  # noqa: E402
from options_scout.pricing import bs_delta, implied_vol  # noqa: E402

from odte_delta_hedged import HALF_SPREAD, R, YEAR_MIN, minute_quotes  # noqa: E402
from odte_download import DATASET, SCHEMA, WORKERS, osi, OUT as ATM_DIR  # noqa: E402
from odte_to_close import rest_of_day  # noqa: E402
from odte_to_close_audit import summ  # noqa: E402
from opening_screen import load_bars  # noqa: E402

ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
WING_DIR = ATM_DIR.parent / "odte_wings"
PLAN = WING_DIR / "plan.csv"
MULTS = (1.0, 1.5, 2.0)
ENTRY_M, CLOSE_M, REHEDGE = 20, 390, 5
COMM = 4 * 0.65


def atm_days():
    """Days with an ATM quote file and a full session: date, strike K, spot, 9:50 legs."""
    bars = load_bars()
    spy = bars[bars.symbol == "SPY"]
    rows = []
    for d, g in spy.groupby("date"):
        if not (ATM_DIR / f"{d}.dbn.zst").exists() or g.mins.max() < 385:
            continue
        pre = g[g.mins <= 19]
        if pre.empty:
            continue
        spot = float(pre.close.iloc[-1])
        k = float(round(spot))
        legs = minute_quotes(d, k)
        c0, p0 = legs["C"].loc[ENTRY_M], legs["P"].loc[ENTRY_M]
        if c0.isna().any() or p0.isna().any() or c0.ask_px_00 <= 0 or p0.ask_px_00 <= 0:
            continue
        rows.append(dict(date=d, K=k, spot=spot, cbid=c0.bid_px_00, cask=c0.ask_px_00,
                         pbid=p0.bid_px_00, pask=p0.ask_px_00))
    out = pd.DataFrame(rows)
    out["straddle_mid"] = (out.cbid + out.cask + out.pbid + out.pask) / 2
    return out, bars


def window(d):
    s = datetime.combine(d, dtime(9, 49), tzinfo=ET).astimezone(UTC)
    return s, s + pd.Timedelta(minutes=3)


def download(estimate):
    load_env()
    c = db.Historical(os.environ["DATABENTO_API_KEY"])
    days, _ = atm_days()
    WING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(days)} days with ATM quotes", flush=True)

    def choose(r):
        d, K = r.date, r.K
        cands = {}
        for m in MULTS:
            w = max(1, round(m * r.straddle_mid))
            cands[m] = ([K + w + j for j in (0, 1, -1, 2, -2)], [K - w - j for j in (0, 1, -1, 2, -2)])
        strikes = sorted({k for cs, ps in cands.values() for k in cs} |
                         {k for cs, ps in cands.values() for k in ps})
        syms = {(cp, k): osi(d, cp, k) for k in strikes for cp in ("C", "P")
                if (cp == "C" and k > K) or (cp == "P" and k < K)}
        res = c.symbology.resolve(dataset=DATASET, symbols=list(syms.values()),
                                  stype_in="raw_symbol", stype_out="instrument_id",
                                  start_date=d, end_date=d + pd.Timedelta(days=1))
        missing = set(res.get("not_found", []))
        out = []
        for m, (cs, ps) in cands.items():
            kc = next((k for k in cs if syms.get(("C", k)) and syms[("C", k)] not in missing), None)
            kp = next((k for k in ps if syms.get(("P", k)) and syms[("P", k)] not in missing), None)
            out.append(dict(date=d, K=K, mult=m, kc=kc, kp=kp))
        return out

    if PLAN.exists():
        plan = pd.read_csv(PLAN, parse_dates=["date"])
        plan["date"] = plan.date.dt.date
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            plan = pd.DataFrame([x for xs in pool.map(choose, days.itertuples()) for x in xs])
        plan.to_csv(PLAN, index=False)
    ok = plan.dropna(subset=["kc", "kp"])
    print(f"wing strikes found for {ok.date.nunique()} days "
          f"({len(ok)}/{len(plan)} day-width combos)", flush=True)
    per_day = {d: sorted({osi(d, "C", k) for k in g.kc} | {osi(d, "P", k) for k in g.kp})
               for d, g in ok.groupby("date")}

    if estimate:
        sample = list(per_day.items())[:: max(1, len(per_day) // 15)]
        cost = [float(c.metadata.get_cost(dataset=DATASET, schema=SCHEMA, symbols=s,
                                          stype_in="raw_symbol", start=window(d)[0],
                                          end=window(d)[1])) for d, s in sample]
        print(f"ESTIMATED TOTAL ${np.mean(cost) * len(per_day):.2f} (no credits spent)")
        return

    def fetch(item):
        d, syms = item
        path, part = WING_DIR / f"{d}.dbn.zst", WING_DIR / f"{d}.dbn.zst.part"
        if path.exists():
            return None
        for attempt in range(4):
            try:
                c.timeseries.get_range(dataset=DATASET, schema=SCHEMA, symbols=syms,
                                       stype_in="raw_symbol", start=window(d)[0],
                                       end=window(d)[1], path=str(part))
                part.replace(path)
                return None
            except Exception as ex:  # noqa: BLE001
                part.unlink(missing_ok=True)
                if attempt == 3:
                    return (d, str(ex).splitlines()[0][:80])
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fails = [f for f in pool.map(fetch, per_day.items()) if f]
    print(f"done. failures: {len(fails)}", fails[:5])


def wing_quote(d, sym):
    q = db.DBNStore.from_file(WING_DIR / f"{d}.dbn.zst").to_df()
    q.index = q.index.tz_convert("America/New_York")
    q = q[(q.symbol == sym) & (q.index.time <= dtime(9, 50))]
    if q.empty:
        return None
    b, a = float(q.bid_px_00.iloc[-1]), float(q.ask_px_00.iloc[-1])
    return (b, a) if a > 0 and b >= 0 and a >= b else None


def backtest():
    days, bars = atm_days()
    _, paths = rest_of_day(bars)
    plan = pd.read_csv(PLAN, parse_dates=["date"]).dropna(subset=["kc", "kp"])
    plan["date"] = plan.date.dt.date
    days = days.set_index("date")
    rows = []
    for r in plan.itertuples():
        d = r.date
        if d not in days.index or d not in paths or not (WING_DIR / f"{d}.dbn.zst").exists():
            continue
        a = days.loc[d]
        cw, pw = wing_quote(d, osi(d, "C", r.kc)), wing_quote(d, osi(d, "P", r.kp))
        if cw is None or pw is None:
            continue
        S = paths[d].reindex(range(ENTRY_M, CLOSE_M + 1)).ffill().to_numpy()
        if np.isnan(S).any():
            continue
        K, t0 = r.K, (CLOSE_M - ENTRY_M) / YEAR_MIN
        legs = {"C": ((a.cbid + a.cask) / 2, K), "P": ((a.pbid + a.pask) / 2, K),
                "Cw": ((cw[0] + cw[1]) / 2, r.kc), "Pw": ((pw[0] + pw[1]) / 2, r.kp)}
        iv = {}
        for name, (mid, k) in legs.items():
            kind = "call" if name.startswith("C") else "put"
            iv[name] = implied_vol(kind, mid, S[0], k, t0, R) if mid > 0 else None
        if iv["C"] is None and iv["P"] is None:
            continue
        iv["C"], iv["P"] = iv["C"] or iv["P"], iv["P"] or iv["C"]
        iv["Cw"] = iv["Cw"] or iv["C"]
        iv["Pw"] = iv["Pw"] or iv["P"]

        shares, hedge, traded = 0.0, 0.0, 0.0       # hedge for the SHORT fly
        for i, m in enumerate(range(ENTRY_M, CLOSE_M)):
            if i % REHEDGE == 0:
                t = max(CLOSE_M - m, 1) / YEAR_MIN
                dlt = (bs_delta("call", S[i], K, t, iv["C"], R) + bs_delta("put", S[i], K, t, iv["P"], R)
                       - bs_delta("call", S[i], r.kc, t, iv["Cw"], R)
                       - bs_delta("put", S[i], r.kp, t, iv["Pw"], R))
                target = 100 * dlt                  # short fly delta is -dlt -> hold +100*dlt
                traded += abs(target - shares)
                shares = target
            hedge += shares * (S[i + 1] - S[i])
        traded += abs(shares)
        ST = S[-1]
        loss = abs(ST - K) - max(0.0, ST - r.kc) - max(0.0, r.kp - ST)
        credit_real = a.cbid + a.pbid - cw[1] - pw[1]
        width = max(r.kc - K, K - r.kp)
        # naked straddle on the same day, for comparison
        straddle_real = (a.cbid + a.pbid - abs(ST - K)) * 100 - 2 * 0.65
        rows.append(dict(
            date=pd.Timestamp(d), mult=r.mult, iv=iv["C"],
            fly=(credit_real - loss) * 100 - COMM,
            fly_h=(credit_real - loss) * 100 - COMM + hedge - traded * HALF_SPREAD,
            capital=(width - credit_real) * 100,
            credit_share=credit_real / ((a.cbid + a.pbid)),
            straddle=straddle_real))
    t = pd.DataFrame(rows)
    t["period"] = np.where(t.date < "2023-01-01", "2018-22", "2023-26")

    print(f"\n{t.date.nunique()} days with all four legs quoted, "
          f"{t.date.min().date()}..{t.date.max().date()}\n")
    for m, g in t.groupby("mult"):
        g = g.sort_values("date")
        print("=" * 98)
        print(f"WINGS AT {m:.1f}x THE STRADDLE PRICE  ({len(g)} days)  — keep "
              f"{g.credit_share.mean()*100:.0f}% of the straddle's premium; "
              f"capital at risk ${g.capital.mean():,.0f}/trade")
        print("=" * 98)
        for name, col in (("iron fly, unhedged", "fly"), ("iron fly, delta-hedged", "fly_h"),
                          ("naked straddle, unhedged", "straddle")):
            print(f"  {name:26} {summ(g[col])}")
            print(f"  {'':26}   worst day ${g[col].min():+,.0f} | 2018-22 "
                  f"${g[g.period == '2018-22'][col].mean():+.1f}/trade | 2023-26 "
                  f"${g[g.period == '2023-26'][col].mean():+.1f}/trade")
        roc = g.fly_h.mean() / g.capital.mean()
        print(f"  hedged fly return on capital at risk: {roc*100:+.2f}%/trade "
              f"(~{roc*252*100:+.0f}%/yr if traded every day)")
        print(f"  extreme days (9:50 IV > 80%, n={int((g.iv > 0.8).sum())}): hedged fly "
              f"${g[g.iv > 0.8].fly_h.mean():+.1f}/trade, unhedged fly ${g[g.iv > 0.8].fly.mean():+.1f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "backtest"])
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    download(a.estimate) if a.mode == "download" else backtest()


if __name__ == "__main__":
    main()
