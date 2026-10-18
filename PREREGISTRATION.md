# Pre-registration — NQ / BTC intraday strategy research

This file fixes, **before** the strategy selection is run and **before** any
test or holdout data is examined, every rule that decides the outcome. It is
committed to git ahead of `experiments/e04_strategies.py`'s full run; the frozen
configuration that run produces is committed ahead of
`experiments/e05_final.py`. The commit history is the audit trail.

## 1. What has been examined before this document

| Step | Data touched | Purpose |
|---|---|---|
| e01 data integrity | all files (row counts, gaps, tick grid, cross-vendor overlap 2025) | data validity only — no returns studied |
| e02 hypothesis tests | NQ train + validation | serial dependence, vol clustering, stationarity, conditional moves, intraday momentum |
| e03 model investigations | NQ train + validation; NQ-B 1-second bars (Mar–Apr 2026) for the order-flow estimate only | vol forecasts, regimes, probability models |
| e04 `--quick` plumbing run | NQ train + validation, 2 values per parameter | code check |

Honest disclosure: e02/e03 results on the validation period were seen before
Strategy C's two variants were written down, and C2 (session router) exists
because e02/e03 found session to be the only regime variable stable across
train and validation. C2 therefore carries validation-period information in its
*design*; its routing *decisions* are fitted on train only. It is counted as a
trial in the deflated Sharpe, and only test/holdout can confirm it.

The NQ-B 1-second file overlaps the holdout window (Mar–Apr 2026). It was used
only for the tick-rule order-flow test, whose result (no predictive value) did
not feed into any strategy. The holdout 1-minute bars have not been loaded by
any experiment.

**Not examined by anything yet:** NQ test (2025-04-01 → 2025-12-11), NQ holdout
(NQ-B 1-minute, 2026-01-21 → 2026-04-15), all BTC returns.

## 2. Data and splits

| Market | Train | Validation | Test (once) | Holdout (once) |
|---|---|---|---|---|
| NQ (NQ-A dev, NQ-B holdout) | 2023-01-03 → 2024-06-30 | 2024-07-01 → 2025-03-31 | 2025-04-01 → 2025-12-11 | NQ-B 2026-01-27 → 2026-04-15 (first 5 sessions of the file are feature warm-up) |
| BTC (Bitstamp) | 2021-01-01 → 2023-06-30 | 2023-07-01 → 2024-06-30 | 2024-07-01 → 2025-06-30 | 2025-07-01 → 2026-10-16 |

Trading instrument for NQ research: **MNQ** ($2/pt, $0.62/side, 1 tick slippage
RTH / 2 ticks ETH, targets need 1 tick trade-through). BTC: perp taker 5 bps +
1 bp slippage per side. Edge is measured in R with exact fractional sizing and
no risk governor; account-level results (integer contracts, governor on) are
computed after selection and never used to select.

## 3. Candidates

* **A — trend**: fresh crossing of z_mom(W) through ±τ. Grid W∈{10,20,40},
  τ∈{1.5,2.0,2.5}, k∈{1.0,1.5,2.0} (stop = k·ATR14), RR∈{1.5,2.0,2.5,3.0},
  holding cap 2W bars.
* **B — mean reversion**: fresh crossing of z_bb(W) beyond ±z, faded. Grid
  W∈{20,40,80}, z∈{2.0,2.5,3.0}, k∈{1.0,1.5,2.0}, RR∈{1.5,2.0,2.5,3.0},
  holding cap W bars.
* Timeframes 1, 5, 15, 60 minutes for both: 864 configurations.
* **C — regime-adaptive**, built at each timeframe from that timeframe's best
  A and best B (by validation score):
  * C1: variance-ratio regime (TREND→A, RANGE→B, NEUTRAL→none), blocked in
    CUSUM volatility transitions and in the top-5% (train) volatility state.
  * C2: session router — RTH and ETH each routed to A, B or nothing according
    to which had positive and higher **train** expectancy there.

## 4. Selection rule

* score = validation mean R − 1.96 × day-clustered SE.
* An A/B configuration is **eligible** if train mean > 0, validation mean > 0,
  ≥ 30 validation trades, and ≥ 60% of its one-step grid neighbours have
  validation mean > 0 (plateau).
* Family finalist = eligible config with the highest validation score (if none
  is eligible, the highest-scoring config, flagged as failing).
* C finalist = C config with train mean > 0, validation mean > 0, ≥ 30 trades
  and the highest validation score (same fallback).

## 5. Development gates (all must hold for a finalist to "pass development")

* D1 validation mean > 0 · D2 train mean > 0 · D3 plateau ≥ 60% (A/B only)
* D4 walk-forward: the family's selection procedure (12-month in-sample, next
  quarter out, 2024Q1–2025Q1) has positive out-of-sample mean. For C the
  adaptive "A or B" procedure stands in, since C has no grid of its own.
* D5 random-entry null: the finalist's validation mean beats ≥ 95% of 300
  random-timing replications with identical exits (p < 0.05).
* Deflated Sharpe ratio is reported (informational; ≥ 0.95 is "strong").

**Selected strategy** = the passing finalist with the highest validation score.
If none passes, the system's output is **NO TRADE** and the frozen file records
the three finalists only so they can be reported on test/holdout.

## 6. Unseen-data gates (evaluated once, frozen configuration, no changes after)

* T1 test: mean R > 0, profit factor > 1, and day-block bootstrap one-sided
  p(mean ≤ 0) < 0.10.
* H1 holdout: mean R > 0 (point estimate; ~55 sessions, low power — a
  necessary, not sufficient, condition).
* Verdict **VALIDATED (provisional, paper-trade next)** only if the selected
  strategy passes development, T1 and H1. Anything else: **NOT VALIDATED → NO
  TRADE**. All three finalists are reported on test and holdout regardless,
  and nothing observed there may be used to change a parameter.

## 7. Reported regardless of outcome

Performance table on identical periods for A, B and C; long/short; by session;
by regime; Monte Carlo drawdown distribution at 0.5% risk; account-level MNQ
and CFD results at $5,000; cost sensitivity.
