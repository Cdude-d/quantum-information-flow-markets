# Quantitative Alpha Research — An Honest Search for Edge in Public Markets

A self-directed research project that started as a naive "options picker" and turned into a
systematic, evidence-driven investigation of a single question:

> **Can an individual, using public data and a laptop, find a durable, tradable edge in
> liquid markets — and if not, *why not*?**

Across ~25 distinct approaches — from moving averages to a quantum-information-flow method and a
10-qubit learned-Hamiltonian factor model — the answer for **direction and cross-sectional stock
selection is a rigorously supported no**. Exactly **one** result survived every test thrown at it:
selling the same-day-expiry (0DTE) S&P 500 straddle at 9:50 ET, delta-hedging it, and holding it
to the close. That is not a hidden mispricing; it is a known **risk premium** (the variance risk
premium), and the project measures exactly what it takes to collect it after real costs.

> **Why the negative results are the point.** In quant research, a tester who fools themselves
> is worse than useless — they lose money confidently. This project repeatedly built promising
> ideas and then *tried to kill them with evidence*: out-of-sample splits, real bid/ask fills,
> survivorship-free data, multiple-testing correction, pre-registered pass/fail rules, seed
> robustness, crash and slippage stress tests. The one survivor is reported with every caveat
> that came with it.

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
ways backtests lie: cost-estimate-before-you-spend, separate in-sample / out-of-sample periods,
mid *and* bid/ask fills, survivorship-aware universes.

### 3. Information-flow causality (`research/information_flow.py`, `liang_flow.py`)
- **Granger causality** across a cross-asset basket, with **false-discovery-rate correction**
  and sub-period stability checks.
- **Liang–Kleeman Information Flow** (Liang 2014) — a causality measure with *T = 0 ⇔
  independence* — implemented from the estimator and tested against circular-shift surrogates.

### 4. Quantum-information market structure (`research/quantum_market_flow.py`, `lindbladian_flow.py`)
- ρ(t) = C(t)/Tr(C(t)) — the normalized correlation matrix as a density operator.
- **von Neumann entropy** as a systemic-risk signal (identifies the March-2020 crash as peak
  stress with no labels).
- **Quantum Liang Flow (IQLF/DIQLF)** on empirical market dynamics via a learned Liouvillian.

### 5. Factor analysis & sizing (`research/ff_factors.py`, `speculation_sizing.py`, `cross_sectional.py`, `sector_momentum.py`)
- **Fama–French factors** (WRDS, 1926–2026): what the canonical anomalies deliver and how they
  have decayed since publication.
- **Cross-sectional momentum** on survivorship-biased vs survivorship-free universes.
- **Monte-Carlo bankroll simulation** (Kelly).

### 6. Opening-window information screen (`research/opening_screen.py`, `intraday_download.py`)
Which of 14 first-20-minute features (opening gap, early realized vol, relative volume, VWAP
deviation, …) carry information about the **next hour** in SPY/QQQ/IWM, using 8 years of
1-minute bars?
- **Mutual information** against a shuffle null, **Benjamini–Hochberg** FDR on a 2018–2022
  discovery set, confirmation on held-out 2023–2026, then out-of-sample R² against a pre-open
  baseline (VIX, 20-day realized vol, previous day's range).
- Result: the next-hour **range** is strongly predictable (OOS R² 24% → 37.5%); **direction is a
  coin flip** (~52%); trend-vs-chop shows nothing.

### 7. The 0DTE straddle study (`research/odte_*.py`)
Taking the volatility forecast to the options market, one honest step at a time:

| Script | Question |
|---|---|
| `odte_download.py` | Pull 1-minute consolidated quotes for the 9:50 at-the-money 0DTE SPY call + put (parallel, idempotent, cost-estimated) |
| `odte_straddle_backtest.py` | Trade the 1-hour range forecast with straddles → **priced in** |
| `odte_gap_test.py` | Does the forecast know anything the straddle price doesn't? (yes, for range — but an unhedged straddle pays on the *net* move) |
| `odte_delta_hedged.py` | Delta-hedged 1-hour version → +$6/trade at mid, **$0 after real costs** |
| `odte_to_close.py` | Hold to expiry, hedge every 5 min, settle at the close → **first after-cost positive** |
| `odte_to_close_audit.py` | Try to break it: hedge sanity, by year, tails, hedge frequency, slippage grid, margin |
| `odte_history.py` | Crash test on 2018–2022 (Dec 2018, March 2020 halts, 2022 bear) |
| `odte_slippage_stress.py` | Regime-dependent costs: where does it break, and which days earn the money? |
| `odte_ironfly.py` | Defined-risk version: buy wings to cap the worst day |

### 8. Quantum factor model (`research/factor_panel.py`, `factor_benchmark.py`, `quantum_factor_model.py`, `quantum_robustness.py`)
Each stock-month is represented as a **10-qubit state**: 9 factor qubits (momentum, reversal,
long-term reversal, size, volatility, beta, book-to-market, gross profitability, asset growth)
angle-encoded by within-month percentile, plus a return qubit. A **64-term Hamiltonian**
(including non-commuting terms) is learned by autograd through the matrix exponential;
P(beat the median next month) is the Born-rule probability of the return qubit. IQLF/DIQLF
attribution freezes one factor qubit at a time and measures the change in the return qubit's
entropy.
- Survivorship-free **CRSP** monthly panel (2.56M stock-months, delisting returns included) with
  **Compustat** fundamentals lagged six months; discovery 1963–2004, held-out 2005–2024.
- Benchmarked against a classical logistic model and published-anomaly long-shorts on the
  identical split, then checked across **5 seeds under a pre-registered rule**.

---

## Headline findings

| Approach | Honest result |
|---|---|
| Options premium-selling (in vs out of sample) | ~breakeven in-sample → **−32% out-of-sample** |
| Buying "cheap" options (36y, VIX+SPX) | **−1% / month in every IV-rank regime** — the variance risk premium |
| Granger / Liang-Kleeman causality | real, dense, directional structure — **all arbitraged to the cost floor** |
| Quantum market-state entropy | **validated systemic-risk signal**, but no forward return edge |
| Cross-sectional momentum (survivorship-free) | Sharpe collapses 0.8 → **0.3**, matching canonical Fama-French UMD |
| Fama-French factors, 1926-2026 | anomalies **real but modest (Sharpe ~0.3), crash-prone, decaying** |
| Published anomalies, CRSP 1963-2004 → 2005-2024 | reproduce in discovery; **none confirm held-out** |
| Quantum factor model (large caps) | +5.3%/yr in one run → **+1.3%/yr (t 0.37) across 5 seeds: not established** |
| Opening 20 minutes → next hour | **range predictable** (OOS R² 24% → 37.5%); **direction a coin flip** |
| 1-hour 0DTE straddles, hedged or not | **priced in** — the spread eats the premium |
| **0DTE SPY straddle, sold 9:50, hedged, held to close** | **+$22/trade after real fills, Sharpe ~5, 8 of 9 years positive, 1,673 days** |
| Same, as an iron butterfly (2× wings) | +$10/trade, Sharpe 2.8, worst day −$440 — wings cost half the edge |

**Unifying conclusion:** the market's directional and cross-sectional structure is *real and
describable* but *arbitrage-priced*. What survives is a **risk premium** — option sellers are paid
to absorb volatility risk — and only in a form that holds to expiry (one bid/ask crossing instead
of two), hedges continuously, and therefore needs a margin account and automation. It breaks if
real costs run ~3× the modelled ones, and it has not yet been validated live. See
[RESULTS.md](RESULTS.md) for the full numbers and caveats.

---

## Skills demonstrated

- **Python scientific stack** — numpy, pandas, scipy, sqlalchemy, PyTorch (autograd through
  matrix exponentials).
- **Financial data engineering** — Databento (OPRA consolidated option quotes, Nasdaq ITCH
  minute bars; parallel, idempotent, cost-estimated downloads) and direct **WRDS PostgreSQL**
  access (CRSP, Compustat, Fama-French).
- **Derivatives** — Black-Scholes, implied vol, Greeks, delta-hedging / gamma-scalping
  simulation, the variance risk premium, margin and settlement mechanics.
- **Statistical rigor** — discovery / held-out splits, mutual information with shuffle nulls,
  Benjamini-Hochberg FDR, permutation tests, Newey-West t-stats, pre-registered decision rules,
  seed robustness, survivorship bias, transaction-cost and slippage stress testing.
- **Time-series causality** — Granger causality, Liang-Kleeman information flow.
- **Quantum information** — density matrices, von Neumann entropy, learned Hamiltonians and
  Liouvillians, quantum information flow (IQLF/DIQLF) as an attribution method.
- **Scientific communication and, above all, intellectual honesty** — including reporting where
  my own predictions were wrong.

---

## Repository structure

```
options_scout/      Live scanner package: pricing, signals, strategies, reasoning, report, scoreboard
morning_scan.py     Scanner entry point
backtest/           Real-option-data backtesting: cost-estimate, download, run, and strategy variants
research/           The investigation:
  information_flow*, liang_flow, quantum_market_flow, lindbladian_flow   causality & quantum market state
  ff_factors, cross_sectional, sector_momentum, speculation_sizing       factors & sizing
  intraday_download, opening_screen                                      opening-window screen
  odte_*                                                                 0DTE straddle study
  factor_panel, factor_benchmark, quantum_factor_model, quantum_robustness   quantum factor model
  wrds_*                                                                 WRDS connection & access probes
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

The 0DTE study, in order (needs a Databento key; every download script has an `--estimate` mode
that prices the request without spending credits — the whole study cost under $3):

```bash
python research/intraday_download.py        # SPY/QQQ/IWM 1-minute bars, 2018-2026
python research/opening_screen.py           # opening-window screen (builds the feature cache)
python research/odte_download.py --estimate # then without --estimate; add --from-bars --start 2018-06-01 --end 2022-12-31 for the crash test
python research/odte_to_close.py
python research/odte_to_close_audit.py
python research/odte_history.py
python research/odte_slippage_stress.py
python research/odte_ironfly.py download --estimate   # then: download, backtest
```

The quantum factor model needs WRDS (CRSP + Compustat): `factor_panel.py` →
`factor_benchmark.py` → `quantum_factor_model.py` → `quantum_robustness.py`.

API-based components require your own credentials in a local `.env` (never committed).
Free-data research scripts under `research/` run as-is.

## References / methods

- X. S. Liang, "Unraveling the cause-effect relation between time series," *Phys. Rev. E* 90 (2014).
- X. S. Liang, "Normalized multivariate time series causality analysis," (information-flow estimator).
- Yi & Bose, quantum extension of the Liang-Kleeman information flow (Cumulative Quantum Liang Flow), 2015.
- Y. Benjamini & Y. Hochberg, "Controlling the false discovery rate," *JRSS B* 57 (1995).
- Fama & French factor data, CRSP and Compustat via WRDS; Databento OPRA and Nasdaq ITCH.

*This is a research and educational project, not investment advice. Backtested results are not
live results; the one positive finding has not been validated with real trading.*
