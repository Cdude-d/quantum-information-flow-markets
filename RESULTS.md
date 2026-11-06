# Results & Findings

A condensed summary of the investigation. The through-line: **every approach was held to an
honest standard** — out-of-sample, net of transaction costs, free of survivorship bias, and
corrected for multiple testing — and the conclusions are reported whether or not they were the
ones hoped for.

---

## 1. Options premium-selling — regime-dependent, not an edge

Defined-risk credit spreads / iron condors, backtested on real historical option quotes
(Databento OPRA), settled at expiry.

| Period | Win rate | Return on risk |
|---|---|---|
| In-sample | 71% | +1.2% |
| **Out-of-sample** | 53% | **−31.9%** |

The strategy's ~70% win rate is *seductive but misleading*: for these payoff structures the
**breakeven win rate is itself ~70%**, so a high win rate is consistent with zero (or negative)
expectancy. The out-of-sample collapse is the honest verdict — the in-sample result was regime
luck, and the probability model was ~8 points overconfident out-of-sample.

## 2. Buying options — negative expectancy in every regime (36 years)

Long ATM straddles priced off the VIX vs. subsequently-realized S&P moves, 1990–2026, bucketed
by implied-vol rank:

| IV rank (cheap → rich) | Straddle-buyer edge / month |
|---|---|
| Cheapest 0–20% | **−0.99%** |
| Richest 80–100% | −1.46% |
| **Overall** | **−1.09%** |

Buying options loses in *every* volatility regime, including when they are cheapest — the
variance risk premium is a structural cost paid by the buyer. "Buy undervalued options" is not
a viable edge.

## 3. Information-flow causality — real structure, arbitraged away

Directed causality across a 12-asset cross-asset basket via **Granger causality** (FDR-corrected)
and the **Liang–Kleeman Information Flow** (T = 0 ⇔ independence). Both flagged equities →
high-yield credit (SPY/QQQ → HYG) as the dominant, stable, directed relationship. Traded net of
costs, out-of-sample:

- **QQQ → HYG lead-lag:** net Sharpe **0.31**, positive in both sub-periods — but statistically
  indistinguishable from zero (t ≈ 1.1), cost-fragile, and a known effect already arbitraged to
  the cost floor.

## 4. Quantum market state — a validated risk signal, not an edge

The correlation matrix as a density operator, ρ(t) = C(t)/Tr(C(t)); its **von Neumann entropy**
as a systemic-risk measure; and a first implementation of the **Quantum Liang Flow (IQLF/DIQLF)**
on empirical market dynamics via a learned Liouvillian generator.

- **von Neumann entropy correctly identifies the March-2020 crash as peak systemic stress**, with
  no labels — a genuine, validated collective risk signal.
- But it adds **+0.1% incremental R²** over trailing volatility for forecasting future risk —
  i.e., it is *coincident*, an elegant descriptor rather than a forward edge.

## 5. Cross-sectional momentum — a live demonstration of survivorship bias

| Universe | Momentum Sharpe |
|---|---|
| Hand-picked current large-caps (biased) | 0.48–0.80 |
| **Survivorship-free sector ETFs** | **0.27–0.33** |
| Benchmark: equal-weight buy-and-hold | **0.74** |

Removing survivorship bias roughly *halved* the Sharpe, and the "sophisticated" market-neutral
strategy underperformed simply owning the sectors. A clean illustration of how much bias inflates
a naive backtest.

## 6. Cross-validation against a century of gold-standard data (Fama–French, 1926–2026)

| Factor | Full-sample Sharpe | Max drawdown | Decay |
|---|---|---|---|
| Market (equity premium) | 0.46 | −85% | the one durable premium |
| **Momentum (UMD)** | 0.52 | **−72%** | 1990s: 1.76 → 2020s: 0.27 |
| Value (HML) | 0.38 | −56% | *negative* in the 2010s |
| Size (SMB) | 0.11 | −61% | effectively gone since ~1980 |

The canonical anomalies are **real but modest (Sharpe ~0.3 today), crash-prone (−50% to −85%),
and decaying** as they are crowded out. Critically, the survivorship-free sector-momentum result
(Sharpe ~0.3) **matches** the canonical UMD factor in the recent era — confirming the whole
investigation measured reality correctly.

## 7. Published anomalies and a quantum factor model — nothing confirms held-out

Survivorship-free CRSP monthly panel (2.56M stock-months, delisting returns), Compustat
fundamentals lagged six months, nine factors. Discovery 1963–2004, held-out 2005–2024.

- **Classical:** discovery reproduces the literature (momentum +17.8%/yr, t 4.9; reversal, size,
  asset growth). **None confirm held-out** — momentum falls to +6% (t 1.0) and reversal, size,
  asset growth and book-to-market flip sign. A logistic model on all nine factors: held-out AUC
  0.521 (all stocks) and 0.506 (large caps).
- **Quantum model** (10 qubits, 64-term learned Hamiltonian, Born-rule probability): ties the
  classical model on all stocks. On large caps one run gave +5.3%/yr gross (t 1.77) — but under a
  **pre-registered 5-seed rule** the ensemble is **+1.3%/yr, t 0.37: not established**. It does
  beat the classical model on every seed (+4.0%/yr, paired t 1.81), i.e. it *loses less*.
- **Attribution:** IQLF/DIQLF factor shares are **unstable across seeds** (asset growth 3–30%,
  profitability 6–31%); only volatility is consistent (~19%). Weak-signal fits are not
  identifiable, so information-flow attributions need seed averaging and uncertainty bands —
  a methodological finding in its own right. (My prior that momentum would dominate was wrong.)

## 8. Opening-window screen — volatility is predictable, direction is not

14 features from the first 20 minutes (SPY/QQQ/IWM, 1-minute bars 2018–2026) against four
next-hour targets; mutual information vs a shuffle null, BH-FDR on 2018–2022, confirmation on
held-out 2023–2026, then out-of-sample R² over a pre-open baseline (VIX, 20-day realized vol,
prior range).

| Target (9:50 → 10:50) | Baseline OOS R² | With opening features |
|---|---|---|
| **High-low range** | 24% | **37.5%** (early realized vol +13 pts, relative volume +4) |
| Absolute move | 2.5% | 4.0% |
| Direction | — | **~52% hit rate (coin flip)** |
| Trend vs chop | — | nothing |

Mutual information flagged volatility features for direction too — the OOS R² check caught that
they predict dispersion, not sign.

## 9. From a volatility forecast to a trade — the one survivor

| Version (0DTE SPY ATM straddle, sold 9:50) | Real fills, $/trade | Verdict |
|---|---|---|
| Forecast-timed, closed at 10:50 | −$9.7 | fully priced in |
| Delta-hedged, closed at 10:50 | $0.0 (+$6 at mid, Sharpe 2–2.8) | spread eats the premium |
| **Delta-hedged every 5 min, held to the 4 PM close** | **+$22/trade, Sharpe ~5** | **survives** |

- **Why the forecast alone wasn't enough:** it *does* beat the straddle price at explaining the
  next hour's range (R² 49% vs 44%; both 50.8%, forecast t = 11.2). But an unhedged straddle pays
  on the **net** move, which nobody predicts, and closing after an hour crosses the bid/ask twice.
- **Why held-to-expiry works:** one spread crossing instead of two (entry spread ≈ 0.9% of
  premium), settlement at intrinsic value, and the hedge converts the payoff into
  realized-vs-implied variance (hedged P&L correlates +0.50 with realized variance, −0.05 with the
  net move). Implied exceeds realized vol by only ~4% on average but on **73% of days**.
- **The forecast is a mild enhancer, not the source:** selling only when the forecast is below
  the straddle price gives +$32.8/trade (permutation p = 0.015) vs +$22 always-sell. The edge is
  the 0DTE variance risk premium itself — a known, compensated risk.

## 10. Trying to break it

| Test | Result |
|---|---|
| Held-out 2023–2026 (950 days) | +$22.2/trade, 70% win, Sharpe 4.56, every year positive |
| **Crash test 2018–2022 (723 days)** | +$21.9/trade, Sharpe 6.21; 2018 flat, 2019–22 positive |
| Feb 20 – Apr 30, 2020 (31 days, incl. halt days) | **+$58.7/trade**, worst day −$178 |
| Worst day, 2018–2026 | −$1,102 (Apr 7, 2025); unhedged it would have been up to −$2,995 |
| Hedge every 1 / 5 / 15 / 60 min | Sharpe 3.9 / 4.6 / ~4 / 2.3 |
| Costs proportional to IV | +$15.8/trade, Sharpe 3.6 |
| Costs ×2 / **×3** | +$6.8 / **−$2.2 — the break point** |
| Where the money comes from (9:50 IV) | <20%: +$15 · 20–40%: +$26 · 40–80%: +$33 · **>80%: −$41** (24 days) |

Only 17% of the profit comes from days with IV above 40% — it is a normal-day premium, not a
crisis bet. (Skipping extreme-IV days is a *post-hoc* hypothesis, not a tested rule.) Along the
way an audit caught a data bug — a rolling median without `min_periods` silently dropped
March 9 – April 14, 2020 from the sample; fixed, everything rerun, conclusions unchanged.

## 11. Defined risk costs half the edge

Iron butterfly: same short straddle plus long wings at K ± (multiple × straddle price), real
fills on all four legs, 1,677 days.

| Wings | Hedged $/trade | Sharpe | Worst day | Unhedged $/trade (Sharpe) |
|---|---|---|---|---|
| 1.0× | −$1.3 | −0.50 | −$338 | −$3.0 (−0.46) |
| 1.5× | +$5.6 | 1.66 | −$345 | +$2.5 (0.26) |
| **2.0×** | **+$10.3** | **2.77** | **−$440** | +$5.4 (0.47) |
| Naked, for comparison | +$22.0 | 5.08 | −$1,102 | +$15.9 (1.10) |

The wings are options too, bought at the same overpricing (plus put skew and their own spreads),
so protection consumes the premium. The version that needs no automation (unhedged) is too weak
to be worth trading.

---

## Unifying conclusion

The market's directional and cross-sectional structure is **real and describable** but
**arbitrage-priced**. The most sophisticated tools applied here (quantum information flow, a
learned 10-qubit Hamiltonian) and the simplest benchmark (buy-and-hold) reach the same place for
predicting *which way* or *which stock*.

What survived is not a mispricing but a **risk premium collected efficiently**: option sellers
are paid to absorb volatility risk, and the 0DTE straddle held to expiry with a continuous hedge
is a structure that keeps enough of that premium after real costs. It comes with real
requirements and open risks:

- a margin account (~$11.7k Reg-T margin per naked straddle, or ~$300 at risk per 2× iron fly
  plus margin for the stock hedge) and pattern-day-trader equity (≥ $25k);
- **automated** 5-minute hedging — the hand-tradeable version does not clear costs;
- costs: the edge is gone at ~3× the modelled slippage;
- untested: live fills, SPY's physical settlement and after-hours moves, crowding as 0DTE volume
  grows, and the post-hoc extreme-IV filter.

**The value of the project is the rigor** — a negative result reached honestly is worth more than
a positive one reached by fooling yourself, and the one positive here is believable *because*
everything else was allowed to fail.
