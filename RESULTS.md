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

---

## Unifying conclusion

The market's structure is **real and describable** but **arbitrage-priced**. The most
sophisticated tool applied here (a quantum information-flow method) and the simplest benchmark
(buy-and-hold) reach the same place: there is no durable, tradable edge available to an individual
with public data in liquid markets, and the documented edges that *do* exist are small, fading,
and painful to hold.

**The value of the project is the rigor, not a strategy** — a negative result reached honestly is
worth more than a positive one reached by fooling yourself, and in quantitative research, knowing
*why* something doesn't work is the actual skill.
