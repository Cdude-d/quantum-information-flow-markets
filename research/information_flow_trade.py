"""Decisive test: does the strongest Granger link (equities -> HYG, reversal)
actually TRADE net of costs and out-of-sample, or is it a stale-price artifact?

Strategy: at each close, take HYG position = -sign(leader return today) [the
'reversal' the Granger test found], hold over the next day, pay a realistic
cost whenever the position flips. Report gross vs net, full sample and each half.

If net is positive and stable across both halves -> a real lead-lag edge.
If gross looks fine but net dies, or it's unstable -> microstructure mirage.

Free yfinance data. Run:  python research/information_flow_trade.py
"""
import numpy as np
import pandas as pd
import yfinance as yf

LEADERS = ["QQQ", "SPY"]
TARGET = "HYG"
PERIOD = "12y"
COST_PER_FLIP = 0.0002   # ~2 bps round-trip on a liquid ETF (HYG is tight)


def ann(daily):
    d = daily.dropna()
    vol = d.std() * np.sqrt(252)
    return {"ret": (1 + d).prod() ** (252 / len(d)) - 1 if len(d) else np.nan,
            "sharpe": (d.mean() * 252) / vol if vol > 0 else np.nan}


def main():
    cl = {}
    for s in set(LEADERS + [TARGET]):
        h = yf.Ticker(s).history(period=PERIOD, interval="1d", auto_adjust=True)["Close"]
        h.index = h.index.tz_localize(None).normalize()
        cl[s] = h
    R = pd.concat(cl, axis=1).pct_change().dropna()
    half = len(R) // 2
    tgt = R[TARGET]
    print(f"{len(R)} days, {R.index[0].date()}..{R.index[-1].date()}, "
          f"cost {COST_PER_FLIP*1e4:.0f}bps/flip\n")
    print(f"{'strategy':16}{'window':>10}{'GROSS ann/Sharpe':>22}{'NET ann/Sharpe':>22}")
    print("-" * 70)

    for leader in LEADERS:
        sig = -np.sign(R[leader])            # reversal: leader up -> short HYG
        pos = sig.shift(1).fillna(0)         # act on yesterday's signal, hold today
        gross = pos * tgt
        flips = (pos != pos.shift(1)).astype(float)
        net = gross - flips * COST_PER_FLIP
        for tag, sl in (("full", slice(None)), ("1st-half", slice(0, half)),
                        ("2nd-half", slice(half, None))):
            g, n = ann(gross.iloc[sl]), ann(net.iloc[sl])
            print(f"{leader+'->'+TARGET:16}{tag:>10}"
                  f"{g['ret']*100:>13.1f}%{g['sharpe']:>8.2f}"
                  f"{n['ret']*100:>13.1f}%{n['sharpe']:>8.2f}")
        print(f"  trades/yr ~{flips.mean()*252:.0f}")
        print()

    print("Verdict: NET Sharpe positive AND similar in both halves -> real tradable edge.")
    print("If gross is fine but NET collapses, or halves disagree -> stale-price mirage.")


if __name__ == "__main__":
    main()
