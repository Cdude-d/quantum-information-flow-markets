# Quantitative Alpha Research — An Honest Search for Edge in Public Markets

A self-directed research project that started as a naive "options picker" and turned into a
systematic, evidence-driven investigation of a single question:

> **Can an individual, using public data and a laptop, find a durable, tradable edge in
> liquid markets — and if not, *why not*?**

The answer, reached across ~17 distinct approaches ranging from moving averages to a
quantum-information-flow method applied to market dynamics, is a rigorously-supported **no** —
and the *value of this project is in how that conclusion was reached*: out-of-sample, net of
costs, free of survivorship bias, and cross-validated against a century of gold-standard
academic factor data.

> **Why a negative result is the point.** In quant research, a tester who fools themselves is
> worse than useless — they lose money confidently. This project repeatedly built promising
> ideas and then *killed them with evidence*. That discipline — not a cherry-picked backtest —
> is the skill it demonstrates.

---

## What's inside

### 1. A live options scanner (`options_scout/`, `morning_scan.py`)
A production-style system that runs every morning, pulls live option chains, and proposes
defined-risk trades with probability estimates and written reasoning.
- **Black–Scholes pricing** from scratch: implied-vol solving (Brent), Greeks, lognormal
  probability-of-profit, expected value.
- **Signal engine**: EWMA realized vol, trend/mean-reversion/momentum, gap statistics, an
  event calendar (FOMC, NFP, earnings, NYSE holidays).
- **Strategy construction**: delta-targeted credit spreads, iron condors, and debit spreads,
  with liquidity/credit filters and a composite ranking score.
- **Self-scoring track record**: every pick is logged and settled at expiry — an honest,
  pre-committed scoreboard rather than hindsight.

### 2. A rigorous options backtester (`backtest/`)
Backtesting on **real historical option quotes** (Databento OPRA), built to avoid the classic
ways backtests lie.
- **Cost-estimate-before-you-spend** workflow against a paid data API.
- **Out-of-sample discipline**: separate in-sample and out-of-sample periods; a strategy that
  looks great in one and dies in the other is exposed, not celebrated.
- **Realistic fills**: mid-price *and* bid/ask (cross-the-spread) P&L, so the transaction-cost
  reality is explicit.
- **Survivorship-aware** universe handling; multiple strategy variants (premium selling, long
  options, combos) all put through the same gauntlet.

### 3. Information-flow causality (`research/information_flow.py`, `liang_flow.py`)
Moving beyond correlation to *directed* causality between assets.
- **Granger causality** across a cross-asset basket, with **false-discovery-rate correction**
  (to separate signal from multiple-testing noise) and sub-period stability checks.
- **Liang–Kleeman Information Flow** (Liang 2014) — a rigorous causality measure with the
  property *T = 0 ⇔ independence* — implemented from the estimator and significance-tested with
  circular-shift surrogates.

### 4. A quantum-information approach to market structure (`research/quantum_market_flow.py`, `lindbladian_flow.py`)
The most novel component: treating the market's correlation structure as a **quantum density
matrix** and measuring directed information flow between asset blocs.
- ρ(t) = C(t)/Tr(C(t)) — the normalized correlation matrix, a valid density operator.
- **von Neumann entropy** as a systemic-risk signal (validated: it correctly identifies the
  March-2020 crash as peak systemic stress with no labels).
- A first implementation of the **Quantum Liang Flow (IQLF/DIQLF)** on empirical market
  dynamics via a *learned Liouvillian generator* — extending an information-flow causality
  method (Liang–Kleeman; quantum extension after Yi & Bose 2015) into econophysics.

### 5. Factor analysis & sizing (`research/ff_factors.py`, `speculation_sizing.py`, `cross_sectional.py`)
- **Fama–French factors** (via WRDS, 1926–2026): what the *canonical* momentum/value/size
  anomalies actually deliver, and how they've **decayed** since publication.
- **Cross-sectional momentum**, tested on both a survivorship-biased and a survivorship-free
  (sector-ETF) universe — a live demonstration of how much bias inflates a result.
- **Monte-Carlo bankroll simulation** (Kelly): why position sizing controls *survival*, not
  the sign of the edge.

---

## Headline findings

| Approach | Honest result |
|---|---|
| Options premium-selling (in vs out of sample) | ~breakeven in-sample → **−32% out-of-sample** |
| Long options / speculation | negative-EV; profitable only as outlier-driven regime luck |
| Buying "cheap" options (36y, VIX+SPX) | **−1% / month in every IV-rank regime** — the variance risk premium |
| Granger / Liang-Kleeman causality | real, dense, directional structure — **all arbitraged to the cost floor** |
| Quantum market-state entropy | **validated systemic-risk signal**, but no forward return edge |
| Cross-sectional momentum (survivorship-free) | Sharpe collapses 0.8 → **0.3**, matching canonical Fama-French UMD |
| Fama-French factors, 1926-2026 | anomalies are **real but modest (Sharpe ~0.3), crash-prone (−50% to −85%), and decaying** |

**Unifying conclusion:** the market's structure is *real and describable* but *arbitrage-priced*.
The most sophisticated tool applied (a quantum information-flow method) and the simplest
benchmark (buy-and-hold) agree. Documented edges exist but are small, fading, and gut-wrenching
— not an "unfair advantage" available to a retail trader.

---

## Skills demonstrated

- **Python scientific stack** — numpy, pandas, scipy, sqlalchemy.
- **Financial data engineering** — REST/market-data APIs (yfinance, Databento), and direct
  **WRDS PostgreSQL** access (CRSP, Compustat, Fama-French, OptionMetrics).
- **Derivatives pricing** — Black-Scholes, implied vol, Greeks, probability-of-profit, EV.
- **Statistical rigor** — out-of-sample testing, false-discovery-rate correction, survivorship
  bias, transaction-cost modeling, bootstrap/surrogate significance.
- **Time-series causality** — Granger causality, Liang-Kleeman information flow.
- **Advanced methods** — density matrices, von Neumann entropy, learned generators (DMD /
  Liouvillian), quantum information flow.
- **Risk & sizing** — Monte-Carlo bankroll simulation, Kelly criterion.
- **Scientific communication and, above all, intellectual honesty.**

---

## Repository structure

```
options_scout/      Live scanner package: pricing, signals, strategies, reasoning, report, scoreboard
morning_scan.py     Scanner entry point
backtest/           Real-option-data backtesting: cost-estimate, download, run, and strategy variants
research/           The investigation: causality, quantum market state, factor analysis, sizing
config.json         Scanner configuration (defaults)
requirements.txt    Dependencies
```

Data outputs (`data/`, `backtest/data/`, `reports/`) are git-ignored — they contain
**license-restricted** Databento/WRDS data and a personal trade log, which must not be
redistributed.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python morning_scan.py --dry-run --no-open          # the live scanner (free yfinance data)
python research/ff_factors.py                        # factor analysis (needs WRDS credentials)
```
API-based components (Databento backtests, WRDS queries) require your own credentials in a
local `.env` (never committed). Free-data research scripts under `research/` run as-is.

## References / methods

- E. Liang, "Unraveling the cause-effect relation between time series," *Phys. Rev. E* 90 (2014).
- X. San Liang, "Normalized multivariate time series causality analysis," (information-flow estimator).
- Yi & Bose, quantum extension of the Liang-Kleeman information flow (Cumulative Quantum Liang Flow), 2015.
- Fama & French factor data via WRDS; OptionMetrics (IvyDB); CRSP.

*This is a research and educational project. It is not investment advice, and its central
finding is that it did **not** produce a tradable edge.*
