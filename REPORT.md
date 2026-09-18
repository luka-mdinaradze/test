# Quant Research Brief — 1h Crypto, $100 Account, 1% Risk

**Date:** 2026-09-18 · **Universe:** BTC/USDT (primary), Nasdaq 100 (portfolio leg) · **Timeframe:** 1h
**Capital:** $100 · **Risk per trade:** 1% ($1)

---

## 0. Read this before anything else

Two things shape every answer below.

**(a) I could not download price data in this environment.** The network policy blocks
Binance, Kraken and Stooq (verified: the proxy returns 403 on CONNECT to all three).
So I have **not** invented backtest numbers for sections 2, 4 and 8. Instead I built the
research system that produces them, verified it against hand-checked test cases, and it
runs the moment it has data:

```bash
python3 run_all.py                      # auto-fetches Binance 1h if reachable
python3 run_all.py --csv BTCUSDT_1h.csv # or point it at your own file
```

Every section of the output prints its data source, and results computed on the built-in
simulator are stamped `SYNTHETIC` so they can never be mistaken for evidence.

**(b) Your binding constraint is not strategy quality — it is friction.** This is exact
arithmetic and needs no price data. With risk-based sizing, position size is set by the
stop distance, so notional `= risk / stop%`, and round-trip cost in units of R is:

```
cost_R = 2 × (fee_bps + slippage_bps) / 10,000 ÷ stop%
```

At 7bps per side (5bps taker + 2bps slippage — roughly Binance spot/futures taker):

| Stop width | Notional at $1 risk | Leverage | Cost per trade | Break-even win rate @2:1 |
|---|---|---|---|---|
| 0.30% | $333 | 3.3× | **0.467 R** | 48.9% (vs 33.3% costless) |
| 0.50% | $200 | 2.0× | **0.280 R** | 42.7% |
| 1.00% | $100 | 1.0× | **0.140 R** | 38.0% |
| 1.50% | $67 | 0.7× | **0.093 R** | 36.4% |
| 3.00% | $33 | 0.3× | **0.047 R** | 34.9% |

Read the first row again: with a 0.3% stop you must win **49%** of 2:1 trades just to break
even, and you need 3.3× leverage to take the trade at all. Annualised:

| Stop | Trades/yr | Friction bill |
|---|---|---|
| 0.5% | 900 | **252% of equity per year** |
| 1.0% | 500 | 70% |
| 1.5% | 300 | 28% |
| 3.0% | 120 | **5.6%** |

**This is why most 1h retail systems fail, and it is decided before the first trade.**
Everything below is built around it: wide ATR-based stops, low trade counts, and no
strategy that needs a sub-1% stop.

Three consequences for a $100 account specifically:

1. **Minimum order size bites.** Binance's $5 minimum notional means a stop wider than
   ~20% of price is untradeable at $1 risk. Not a real constraint on 1h bars, but the
   engine enforces and counts it.
2. **Leverage is forced, not chosen.** A 1% stop needs $100 notional = 1× leverage. A
   0.5% stop needs 2×. The leverage is an artefact of risk sizing, not aggression — but
   the liquidation engine does not care about your intent.
3. **$1 of risk cannot be sized precisely.** Rounding to the lot step makes realised risk
   lumpy. Expect ±10% variance around the nominal 1%.

**The honest recommendation: paper-trade this at $100 and fund it at $2,000+ or trade
lower frequency.** At $100 with 1% risk, a *good* year (+47% on the S1 profile below) is
$47 — less than the value of the hours spent. The system is worth building; $100 is the
wrong size to run it at. It is your call, and everything below works at any account size.

---

## 1. Strategy generation — three strategies

All three: BTC/USDT, 1h bars, 1% risk sized between entry and stop, no pyramiding, one
position at a time. Implemented in `quant/strategies.py`.

### S1 — Donchian Trend Breakout *(the core allocation)*

**Indicators (exact settings)**
| Indicator | Setting | Role |
|---|---|---|
| Donchian channel | 55 bars, **excluding the current bar** | breakout level |
| EMA | 200 (close) | regime direction filter |
| ADX | 14 (Wilder) | trend-strength gate, min 20 |
| ATR | 14 (Wilder) | stop distance and trailing |
| Realised vol percentile | 24-bar vol, ranked over 500 bars | min 35th percentile |

**Entry rules (step by step)**
1. At each bar close, compute the prior 55-bar high/low (shifted — the current bar is excluded).
2. Gate: `ADX(14) ≥ 20` **and** realised-vol percentile `≥ 0.35`. If either fails, no trade.
3. Long if `close > prior 55-bar high` **and** `close > EMA200`.
4. Short if `close < prior 55-bar low` **and** `close < EMA200`.
5. Execute at the **next bar's open**. Never at the signal bar's close.

**Exit rules**
- Initial stop: `2.0 × ATR(14)` from entry.
- Trailing: chandelier at `3.0 × ATR(14)` from the close, updated at each bar's close, effective the next bar. Ratchets one way only.
- No fixed target — trend edges live in the right tail, and a target amputates it.
- Time stop: 240 bars (10 days). If it has not worked in 10 days, it is not working.

**Where it works best:** directional regimes with expanding volatility — post-halving
trends, macro-driven repricings, liquidation cascades. **Worst:** low-ADX chop, where
every break fails and it dies by a thousand cuts.

**Why it has an edge.** Crypto is retail-dominated, 24/7, and structurally over-levered.
When price clears a multi-day range, stop-losses and liquidation engines fire *mechanically*
— a forced-seller (or buyer) cascade that is not an opinion and does not reprice instantly.
That produces genuine short-horizon serial correlation after range breaks. The deeper
reason trend-following persists across 200 years of data is that it is a **risk premium**:
you accept a 65–70% loss rate and long drawdowns, and get paid for holding through pain
that most participants quit. That is an edge that cannot be arbitraged away, because
arbitraging it requires enduring exactly the thing people won't endure.

### S2 — Range Mean-Reversion *(the diversifier)*

**Indicators**
| Indicator | Setting | Role |
|---|---|---|
| Bollinger Bands | 20 bars, **2.5σ** | stretch threshold |
| RSI | **2 bars** (Wilder) | exhaustion trigger |
| ADX | 14, max 18 | range-regime gate |
| ATR | 14 | stop distance |

**Entry rules**
1. Gate: `ADX(14) ≤ 18`. Trending market → stand down entirely.
2. Long when `close < lower band (2.5σ)` **and** `RSI(2) < 5`.
3. Short when `close > upper band (2.5σ)` **and** `RSI(2) > 95`.
4. Fill at the next bar's open.

**Exit rules**
- Stop: `1.5 × ATR(14)`.
- Target: the 20-bar mean (the band midline) — roughly `1.5 × ATR`, so ≈1:1 gross.
- Hard time stop: 48 bars. A mean-reversion trade that hasn't reverted in two days has been re-rated by information, not noise.

**Where it works best:** quiet, rangebound consolidation — the 60–70% of the time crypto
does nothing. **Worst:** the first day of a new trend, and any news-driven gap.

**Why it has an edge.** In the absence of information, order flow is dominated by
inventory: market makers who get filled on one side must offload, and leveraged retail
stops out at extremes. Fading a 2.5σ move with no news is **selling liquidity to a
one-sided but uninformed flow**. The edge is real but structurally small and, as section
3 shows, friction eats ~65% of it. It earns its place as a diversifier — it makes money
when S1 loses — not as a standalone.

### S3 — Volatility Squeeze Expansion *(the convexity sleeve)*

**Indicators**
| Indicator | Setting | Role |
|---|---|---|
| Bollinger Bands | 20 bars, 2.0σ | compression measure |
| Keltner Channel | 20 EMA, 1.5 × ATR(20) | compression reference + trigger level |
| Squeeze | BB inside KC for **≥ 6 consecutive bars** | setup condition |
| EMA | 50, slope | direction tiebreak |
| ATR | 14 | stop distance |

**Entry rules**
1. Detect squeeze: both Bollinger bands inside the Keltner channel, ≥6 bars running.
2. Wait for release: the squeeze ends.
3. Long if the release bar closes **above** the upper Keltner band and EMA50 is rising; short if it closes **below** the lower band and EMA50 is falling.
4. Fill at the next bar's open.

**Exit rules**
- Stop: `1.2 × ATR(14)` (tight — compression means small ATR, and a failed expansion fails fast).
- Target: `3.0 R`, with a `2.5 × ATR` trail running underneath.
- Time stop: 72 bars.

**Where it works best:** after multi-day compression — weekend ranges, pre-CPI/FOMC coils,
post-capitulation basing. **Worst:** choppy medium volatility, where the squeeze/release
flag flickers and produces false triggers.

**Why it has an edge.** It does not forecast direction — it forecasts **variance**, and
variance is the single most forecastable quantity in finance. Volatility clustering
(the GARCH effect) is among the most robust empirical regularities in every market ever
studied. Low vol *reliably* precedes high vol. So the strategy buys a synthetic straddle
with a mechanical stop: 30% win rate, 3:1 payoff, positive skew. Most traders cannot run
it because losing 70% of the time feels like being broken, even while the equity curve rises.

**How the three fit together:** S1 needs trend, S2 needs range, S3 needs a volatility
transition. Their ADX gates are mutually exclusive by construction (S1 needs ADX ≥ 20,
S2 needs ≤ 18), so they rarely fire at once — which is the point. Run all three and you
have a strategy for each regime rather than one strategy praying for its regime.

---

## 2. Backtesting

**I did not run a 5–10 year backtest, because this environment cannot reach market data,
and a fabricated table would be worse than no table.** What exists instead is the harness
that produces one, with the anti-self-deception properties that decide whether a backtest
is worth reading.

**Run it:**
```bash
pip install -r requirements.txt
python3 run_all.py                  # Binance 1h BTCUSDT, Aug-2017 → today (~70k bars)
python3 run_all.py --csv my.csv     # your own OHLCV
python3 tests/test_engine.py        # verify the engine before trusting it
```

**What the engine guarantees** (`quant/engine.py`, each one verified by a test in
`tests/test_engine.py`, all passing):

| Property | Why it matters |
|---|---|
| Signal at bar *t* fills at bar *t+1*'s **open** | The most common source of fake backtest alpha is filling at the signal bar's close. |
| Bar hitting **both** stop and target books the **stop** | 1h bars often touch both. Assuming the good one first is pure fiction. |
| Gaps fill at the **open**, not the stop price | Verified to produce a −2R loss on a gap, not a polite −1R. |
| Fees **and** slippage charged on both sides | A flat round trip loses money, as it does in reality. |
| Sub-minimum orders **rejected and counted** | The $100-account trap, made visible instead of silently skipped. |
| Position sizing risks exactly 1% between entry and stop | Verified: a stop-out costs $1.00 on $100. |

**Metrics produced:** CAGR, Sharpe, Sortino, max drawdown, Calmar, Ulcer index, win rate,
expectancy in R, payoff ratio, profit factor, total cost drag, trades/year, plus a
per-episode drawdown table with recovery times.

**Pipeline validation.** Running the whole thing on 6 years of simulated bars
(`--synthetic`) exercises all 52,560 bars and every module. All three strategies lose
money there — **which is the correct result**: the simulator is a regime-switching random
walk with no exploitable structure, so after costs the expectancy must be negative. An
engine that produced profit on noise would be broken. Those numbers say nothing about BTC.

### When it performs best / worst

| | S1 Trend | S2 Mean-Reversion | S3 Squeeze |
|---|---|---|---|
| **Best** | Sustained directional moves, expanding vol, ADX > 25 | Quiet ranges, ADX < 15, no scheduled news | After ≥6h compression, especially pre-event |
| **Worst** | Chop: repeated false breaks at range edges | Trend days and news gaps — the fade becomes a catastrophic short | Medium-vol chop where squeeze flickers on/off |
| **Historical analogue** | Q4-2020, mid-2021 legs, Jan-2024 ETF run | Most of summer 2019, H2-2023 | Pre-halving coils, pre-FOMC compressions |

### What breaks each strategy

1. **Regime persistence collapse.** If BTC's realised vol keeps compressing as the asset
   institutionalises, S1's trade count falls and its edge/cost ratio degrades. Watch the
   ratio of gross expectancy to `cost_R`; below ~3:1, stop trading the system.
2. **Fee or spread changes.** The friction table is linear in fees. Losing a fee tier, or
   trading a thinner pair, can flip positive expectancy negative without any change in signal.
3. **News gaps.** Both fade strategies (S2 especially) assume mean reversion. An
   exchange failure, an ETF decision, or a regulatory headline turns a 2.5σ stretch into
   a 10σ move. This is the tail that ends accounts — hence the hard time stop and the
   never-negotiate-a-stop rule.
4. **Parameter drift.** The 55-bar Donchian is not sacred. If it only works at 55 and
   dies at 50 and 60, it was curve-fit. Section 6 tests exactly that.
5. **Funding costs on perps** (if leveraged). At 1bp/8h the drag is small; in an
   overheated market, funding can hit 0.1%/8h = ~110% annualised against the crowded side.
   `apply_funding=True` in `ExecConfig` models it.

---

## 3. Risk / reward analysis

### Risk per trade
$1.00 (1% of $100), between entry and stop. Realised risk can be *lower* if the 3×
leverage cap binds on a tight stop, and *higher* on a gap — an overnight gap through the
stop cost **−2.0R** in the engine's test case. **Plan for −1R, budget for −2R.**

### Reward-to-risk, after friction
Computed by `analyze_profiles.py` from stated archetype assumptions (not backtest results):

| Strategy | Win rate | Payoff | Stop | Cost/trade | Gross exp. | **Net exp.** | Cost eats | Break-even WR | Margin |
|---|---|---|---|---|---|---|---|---|---|
| S1 trend | 33% | 3.0R | 2.5% | 0.056R | 0.320R | **0.264R** | 17.5% | 26.4% | +6.6pts |
| S2 mean-rev | 62% | 0.9R | 1.2% | 0.117R | 0.178R | **0.061R** | **65.5%** | 58.8% | +3.2pts |
| S3 squeeze | 30% | 3.0R | 1.6% | 0.087R | 0.200R | **0.113R** | 43.7% | 27.2% | +2.8pts |

The decisive column is *cost eats*. **S2 surrenders two-thirds of its gross edge to
friction** because it uses a tight stop at high frequency — and its margin over break-even
is 3.2 points, well inside estimation error. S1 keeps 82% of its edge because it stops
wide and trades rarely. This ranking is structural and would hold on real data too.

### Drawdown patterns (from 20,000 Monte Carlo paths, 300 trades, 1% risk)

| Strategy | Median maxDD | 95th pct maxDD | Worst losing streak | P(end below $100) |
|---|---|---|---|---|
| S1 trend | −15.7% | −26.2% | 35 trades | 0.6% |
| S2 mean-rev | −11.7% | −21.3% | 14 trades | 13.6% |
| S3 squeeze | −20.9% | −36.2% | 44 trades | 11.6% |

The shapes differ in kind, not just depth:
- **S1 and S3 (low win rate):** long, shallow, sawtooth grinds punctuated by sharp
  recoveries. A **44-trade losing streak** is normal for S3 — at ~140 trades/year that is
  roughly *four months* of losing. You will be certain it's broken. It won't be.
- **S2 (high win rate):** long placid stretches, then a sudden cliff when a trend starts
  and three or four fades fail back to back. The drawdown arrives fast and feels like a
  regime change, because it is one.

### Three improvements that reduce risk

1. **Volatility-scaled risk instead of fixed 1%.** Replace the constant with
   `risk% = 1% × (median ATR% / current ATR%)`, clamped to [0.4%, 1.5%]. Position risk
   then shrinks automatically going into high-volatility regimes, which is where clustered
   stop-outs happen. Typically cuts max drawdown 20–30% for roughly the same CAGR — it's
   the highest-value single change on this list.
2. **A portfolio-level daily stop.** After −3R in a rolling 24h, stand down until the next
   UTC day. Losing days cluster (volatility clustering again), so this truncates the left
   tail of the daily distribution at a small cost in expectancy.
3. **Correlation gate across the three strategies.** Never hold S1 and S3 in the same
   direction simultaneously — they are both long-volatility breakout bets and will stop
   out together. Cap total open risk at 2R across all strategies.

### Two ways to increase returns *without* increasing risk

1. **Cut friction — the free 18–65%.** Use post-only limit entries at the breakout level
   instead of market orders (maker fee is often negative), and a fee tier or exchange
   token discount. Going from 7bps to 3bps per side lifts S2's net expectancy from 0.061R
   to **0.128R — more than doubling it** — with zero change in risk. This is the single
   highest-return-per-unit-effort change available, and it is pure arithmetic.
2. **Run all three strategies concurrently at the same 1% per trade.** Because their
   regime gates are near-mutually-exclusive, their returns are weakly correlated. Three
   uncorrelated streams at the same per-trade risk raise portfolio return roughly linearly
   while portfolio volatility rises only with √n — Sharpe improves by up to ~√3 ≈ 1.7×.
   Diversification across *edges* is the only genuine free lunch in the business.

---

## 4. Market regime detection — BTC/USDT

**I cannot tell you what BTC is doing right now.** No market data is reachable from this
environment, and my training data ends before this date. Any "current" read I produced
would be invention. What I can give you is the mechanical classifier, which answers the
question the moment you run it:

```bash
python3 -c "
from quant import data, regime
df = data.load_btc_1h()          # or data.load_csv('your.csv')
print(regime.render(regime.classify(df)))"
```

**What it measures** (`quant/regime.py`):

| Dimension | Measurement | Classification |
|---|---|---|
| **Trend** | ADX(14), ±DI, price vs EMA50 vs EMA200 | ADX<18 → *sideways*; price>EMA50>EMA200 & +DI>−DI → *aligned bull*; mirror → *aligned bear*; else *weakening/corrective* |
| **Volatility** | 24-bar realised vol, percentile-ranked over 2000 bars | >0.80 *high* · >0.55 *elevated* · >0.25 *normal* · ≤0.25 **compressed — expansion risk** |
| **Volume** | 24h mean ÷ 30d mean | >1.3× *expanding, confirms* · <0.75× *drying up, breakouts suspect* |

**The decision table it applies** — this is the part worth internalising:

| Regime | Run | Avoid |
|---|---|---|
| Aligned trend, vol ≥ 35th pct | **S1** — directional, trail wide, let winners run | Fading extremes. Mean-reversion shorts into an ADX>25 uptrend is the classic account-killer. |
| Sideways, vol < 60th pct | **S2** — fade band extremes to the mean | Breakout entries. Most breaks in a low-ADX range are false; S1 bleeds by a thousand cuts. |
| Compressed vol (bottom quartile) | **S3** — position for the break, don't predict direction | Wide stops and large size. Compression resolves violently and gaps through stops. |
| Transition / mixed | Reduced size across all three | Adding risk. Transition regimes are where correlated stops cluster. |

---

## 5. Multi-factor strategy

Cross-sectional, for a basket (BTC + majors, or equities). Daily rebalance frequency —
1h bars are the wrong horizon for factor investing, and the friction table above explains
why. Implemented in `quant/multifactor.py`.

### Exact formulas

**Momentum (35%)** — 12-month return skipping the most recent month, risk-adjusted:
```
momentum_i = (P_{t-21} / P_{t-252} − 1) ÷ σ_i,63d_annualised
```
The 21-bar skip avoids the well-documented short-term reversal effect. Dividing by
volatility means a quiet 40% ranks above a violent 40%.

**Trend (30%)** — time-series participation score in [−1, +1]:
```
trend_i = mean( sign(P − MA50), sign(P − MA100), sign(P − MA200) )
```
Unlike momentum, this is absolute, not relative — it is what keeps the book out of a
falling market where *everything* has negative momentum.

**Volatility (20%)** — low-volatility premium:
```
volatility_i = − σ_i,63d_annualised
```
Negated, because the persistent anomaly is that low-volatility assets deliver better
risk-adjusted returns (leverage-constrained investors bid up high-beta names).

**Value (15%)** — price versus a long-horizon anchor:
```
value_i = − ( P_t / median(P, 504 bars) − 1 )
```
Below the 2-year median = cheap = positive score. Crypto has no earnings, so the anchor
replaces a fundamental. For equities, substitute earnings or book yield directly.

### Combination and allocation
1. At each rebalance, z-score every factor **cross-sectionally** (across assets, at that date).
2. Composite `= 0.35·z_mom + 0.30·z_trend + 0.20·z_vol + 0.15·z_value`.
3. Select the top N (default 3) with a **positive** composite. If none is positive, hold cash — a real, frequently-correct output.
4. Weight the selection by **inverse volatility**, cap any single name at 40%.
5. Scale the whole book down (never up) to a 15% annualised volatility target.

### Rebalancing
**Monthly (`ME`)**, at the month's last close. Weekly (`W-FRI`) is supported and is the
practical floor: factor signals decay over months, so faster rebalancing adds turnover
cost without adding information. Turnover is charged at 10bps in the backtest.

### Example portfolio ($100, medium risk)

| Sleeve | Weight | Rationale |
|---|---|---|
| Top factor asset #1 | 21% | Highest composite, inverse-vol weighted |
| Top factor asset #2 | 21% | Second-highest, low correlation to #1 |
| Cash | 58% | What the 15% vol target leaves unallocated |

That cash weight is not timidity — it is what volatility targeting *does* when the
candidates are 60%-vol assets. Forcing it to 100% invested would run the book at ~35%
volatility, more than double the mandate.

---

## 6. Strategy optimisation

**The core claim: any in-sample optimisation result you have ever seen is worthless.**
With 36 parameter sets over 6 years of 1h bars, the best in-sample Sharpe is essentially a
draw from the maximum of 36 noise samples. The harness therefore reports the only number
that means anything — **stitched walk-forward out-of-sample** — alongside the in-sample
figure, so the gap is visible.

**Method** (`quant/optimize.py`): five anchored folds; within each, parameters are chosen
on the first 70% and traded blind on the remaining 30%; equity compounds across folds;
the objective is Sharpe penalised by drawdown (`sharpe × (1 + maxDD)`), which stops the
search from selecting a knife-edge equity curve.

```bash
python3 run_all.py            # prints the before/after comparison
```

Output format (run on real data for real values):

| Fold | OOS window | IS Sharpe | OOS Sharpe | OOS return | OOS maxDD | Trades |
|---|---|---|---|---|---|---|
| 1..5 | *(per fold)* | … | … | … | … | … |
| | **Mean IS** | *x.xx* | **Mean OOS** *y.yy* | | **Overfitting tax = x − y** | |

**How to read it: the overfitting tax is the finding.** If mean IS Sharpe is 1.8 and mean
OOS is 0.3, the strategy is a curve-fit and the 1.8 never existed. A tax under ~0.3 Sharpe
points on a parameter set that is *stable across folds* is the signature of a real edge.

### Specific improvements the search should test

**Indicator settings** — test *neighbourhoods*, never single values:
- Donchian 34 / 55 / 89 — if 55 works and its neighbours don't, it's noise.
- Stop 1.5 / 2.0 / 2.5 ATR — wider stops cut friction per the section-0 table, so the optimum sits wider than intuition suggests.
- Trail 2.5 / 3.5 ATR; ADX gate 18 / 25.

**Entry/exit timing**
- *Retest entry:* instead of buying the breakout close, place a limit at the broken level on a pullback. Lower fill rate, better average entry, and **maker fees instead of taker** — worth ~0.05R per trade on its own.
- *Asymmetric time stops:* currently 240 bars for S1. Test scaling it by ATR — trades in fast markets should be given less time, not the same.
- *Partial exits:* take 50% at 2R, trail the rest. Raises win rate and smooths equity; slightly lowers total return in a strong trend. A drawdown-reduction trade, not a return improvement — worth it if it's what keeps you running the system.

**Filters**
- *Trend:* 4h EMA200 alignment instead of 1h — a higher-timeframe filter cuts trade count ~30% and removes the worst counter-trend breakouts.
- *Volume:* require breakout-bar volume > 1.3× the 24h average. Unconfirmed breaks are disproportionately false.
- *Volatility:* the existing 35th-percentile floor; also test a *ceiling* (skip above the 95th percentile, where slippage explodes and stops gap).
- *Session:* test excluding 00:00–06:00 UTC, historically the thinnest book.

**Discipline that makes the search honest:** a parameter earns its place only if it
improves OOS Sharpe *and* its neighbours also improve. Two-dimensional plateaus, never peaks.

---

## 7. Portfolio construction — BTC + Nasdaq

**Assumptions (change them in `run_all.py`; the allocation follows automatically).** These
are long-run estimates stated as inputs, not forecasts I'm asserting: BTC 25% return /
60% vol, Nasdaq 100 10% / 22%, cash 4% / 1%, BTC–Nasdaq correlation **0.45**, t-bills
uncorrelated. Drawdowns are simulated with Student-t(4) shocks over 2 years, because
Gaussian simulation understates crypto tails badly.

**Assumed risk tolerance: medium; horizon: 2 years** (you left both as placeholders — the
other two profiles are shown so you can pick).

| Profile | BTC | Nasdaq | Cash | Exp. return | Vol | Sharpe | Median 2y | 5th pct 2y | P(loss) | Median maxDD | 95th maxDD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Low | 5% | 9% | 86% | 5.6% | 4.4% | 0.36 | +11.7% | +0.9% | 4% | −4% | −7% |
| **Medium** | **15%** | **21%** | **64%** | **8.4%** | **11.8%** | **0.37** | **+16.8%** | **−11.4%** | **18%** | **−13%** | **−24%** |
| High | 35% | 48% | 17% | 14.2% | 27.4% | 0.37 | +23.5% | −34.8% | 29% | −30% | −51% |

Note all three profiles have near-identical Sharpe (~0.37) — **the risk profile changes
how much risk you take, not how efficiently you take it.** That is exactly what a
correctly-built frontier looks like. The lever is the cash weight.

**Why each asset is included**

- **BTC (15%)** — the highest expected return in the set and the only holding that can
  perform in a monetary-debasement scenario where both equities and bonds struggle. It is
  also the entire drawdown budget: at 60% vol and −77% historical peak-to-trough, a 15%
  weight contributes more portfolio variance than the 21% Nasdaq sleeve. It is capped by
  the risk preset rather than the optimiser, deliberately — mean-variance optimisers
  overweight the highest-return asset precisely when its return estimate is least reliable.
- **Nasdaq 100 (21%)** — long-run earnings growth with vastly deeper liquidity, and it
  carries the portfolio through a multi-year crypto winter. The 0.45 correlation to BTC
  is the weak point: these two are *not* independent, and in a genuine liquidity crisis
  that correlation goes to ~0.8. Do not count on this pair for crisis diversification.
- **Cash / T-bills (64%)** — pays a real yield, funds rebalancing into drawdowns, and is
  the only asset certain to be *available* at the bottom. At a 13% vol target with a
  60%-vol asset in the book, this weight is arithmetic, not pessimism.

**Rebalance quarterly, or when any weight drifts >25% relative** (e.g. BTC 15% → 18.75%).
Rebalancing is what mechanically sells crypto strength and buys crypto weakness — most
of the realised edge in a portfolio like this comes from that, not from the weights.

---

## 8. Trade setups

**I cannot generate live entry prices** — no market data, and the date is past my training
cutoff. Levels invented here would be fiction dressed as analysis. The generator produces
all of it from the latest bar of whatever data you load:

```bash
python3 -c "
from quant import data, setups
df = data.load_btc_1h()
print(setups.render(setups.generate(df, equity=100, risk_pct=0.01), top=3))"
```

For each of the three strategies it emits **entry, stop, take-profit, R:R, exact position
size, notional, dollar risk, exit rule**, and a status:
- **ARMED** — every filter passes; the trigger can fire on the next bar.
- **PENDING** — the setup is forming (e.g. squeeze streak building).
- **BLOCKED** — a regime filter rejects it; the level is shown so you know what would change that.

Worked example of the output shape (levels from simulated data — the *mechanism*, not a recommendation):

```
[ARMED] S1 trend breakout - LONG
  Trigger    1h close above 22,901.90 (prior 55-bar high)
  Filter     ADX 22.5 vs min 20; price > EMA200 20,858.19
  Entry      22,913.35
  Stop       21,705.12   (5.27% away)
  Target     26,538.05   R:R 3.00
  Size       0.000828 units = $18.96 notional, risking $1.00
  Exit       chandelier trail 3xATR from the close; time-stop 240 bars
  Viability  ok
```

Note the size line: a 5.27% stop means **$18.96 of notional** to risk $1 — no leverage
needed, and well above the $5 minimum. Reasoning is attached to every setup as the
`Trigger` (technical) and `Filter` (regime) fields.

**On the macro half of "reasoning":** macro context that matters for BTC on a 1h chart is
almost entirely *event risk*, not level analysis — CPI and FOMC releases, ETF flow prints,
large liquidation events. The practical rule: **do not hold a 1h mean-reversion position
into a scheduled macro release.** Section 11 covers the macro layer properly.

---

## 9. Monte Carlo simulation

20,000 resampled paths, 300 trades, 1% risk from $100, costs embedded in the R
distribution (`quant/montecarlo.py`). Run on a real trade log it resamples actual
outcomes; the table below uses the section-3 archetype assumptions.

| | S1 trend | S2 mean-rev | S3 squeeze |
|---|---|---|---|
| **Probability of loss** (end < $100) | **0.6%** | 13.6% | 11.6% |
| Probability of losing half | 0.00% | 0.00% | 0.08% |
| Median final equity | $226.57 | $120.95 | $144.32 |
| 5th percentile | $130.09 | $92.66 | $86.20 |
| 95th percentile | $379.28 | $157.88 | $241.62 |
| Worst path | $71.79 | $62.13 | $45.71 |
| Median max drawdown | −15.7% | −11.7% | −20.9% |
| 95th-pct max drawdown | −26.2% | −21.3% | −36.2% |
| Worst losing streak | 35 | 14 | **44** |

**Return distribution shape:** heavily right-skewed for S1 and S3 (median $227 vs 95th
percentile $379 vs worst $72) — a few big trends carry everything. S2's distribution is
tight and symmetric, and its downside is *more* likely despite a 62% win rate, because
its edge per trade is thin.

**Worst-case scenarios.** The worst simulated S3 path ends at **$45.71 — a 54% loss —
while the strategy's expectancy is positive.** That is the number to sit with. With a
44-trade losing streak in the sample, a run of four losing months is a normal outcome of
a *working* system, not evidence of a broken one.

### Robustness verdict

- **S1: ROBUST.** 0.6% of paths end below start, ruin risk effectively zero, worst-case
  drawdown −48%. The edge is wide relative to friction (17.5% of gross) and the margin
  over break-even win rate is 6.6 points.
- **S2: BORDERLINE-FRAGILE.** Positive expectancy, but 13.6% of paths lose money over 300
  trades and friction takes 65% of the gross edge. A 3-point error in the win-rate
  assumption — well within sampling noise on a few hundred trades — flips it negative.
  **Do not run this one without cutting fees first.**
- **S3: ROBUST but psychologically brutal.** Low ruin risk and a good right tail, but the
  deepest drawdowns and the longest losing streaks of the three. Its fragility is
  behavioural: most people abandon it during the 44-trade streak, converting a positive-
  expectancy system into a realised loss.

**Sensitivity to risk per trade** (S1 profile, 300 trades) — the most important table here:

| Risk/trade | Median | 5th pct | P(loss) | P(lose half) | Median DD | Worst DD |
|---|---|---|---|---|---|---|
| 0.5% | $153 | $115 | 0.4% | 0.00% | −8% | −27% |
| **1.0%** | **$227** | **$130** | **0.6%** | **0.00%** | **−16%** | **−48%** |
| 2% | $462 | $154 | 0.8% | 0.39% | −30% | −74% |
| 3% | $851 | $166 | 1.1% | 2.76% | −42% | −88% |
| 5% | $2,165 | $148 | 2.3% | 14.93% | −61% | −98% |
| 10% | $4,718 | **$27** | 10.3% | **50.73%** | −89% | −100% |

Full Kelly for this profile is **10.7%** — and at 10% risk, **half of all paths lose half
the account** and the 5th percentile is $27. Kelly maximises the *median* log outcome
while destroying the left tail. **Your 1% choice is roughly one-tenth Kelly, and it is
correct.** Do not let the median column tempt you rightward.

---

## 10. Drawdown analysis

The harness prints per-episode drawdowns with recovery times
(`quant/metrics.py: drawdown_table`) — the deepest episodes, bars to trough, and bars to
recovery, plus the average recovery. Expected profile from the Monte Carlo above:

| | S1 trend | S2 mean-rev | S3 squeeze |
|---|---|---|---|
| Median max drawdown | −15.7% | −11.7% | −20.9% |
| 95th percentile | −26.2% | −21.3% | −36.2% |
| Worst (1% risk) | −47.7% | ~−40% | −74.1% |
| Longest losing streak | 35 trades | 14 | 44 |
| **Typical recovery** | ~1.5–2× the time to trough | fast then cliff-like | slowest — needs a trend to arrive |

**Recovery mathematics, which is not symmetric:** a −20% drawdown needs +25% to recover;
−33% needs +50%; −50% needs +100%. At S1's 0.264R net expectancy and ~180 trades/year
(computed, not estimated):

| Drawdown | Gain needed | Trades to recover | Time at 180 trades/yr |
|---|---|---|---|
| −15% | +17.6% | 62 | 4.1 months |
| −20% | +25.0% | 85 | 5.6 months |
| −25% | +33.3% | **109** | **7.3 months** |
| −33% | +49.3% | 152 | 10.1 months |

And that is the *expected* path — half of all recoveries take longer. This is the real
cost of drawdown, and it is why the three fixes below matter more than any signal
improvement.

### Three ways to reduce drawdowns

1. **Volatility-targeted risk** (the same fix as section 3, and the best one). Set
   `risk% = 1% × (median ATR% ÷ current ATR%)`, clamped [0.4%, 1.5%]. Drawdowns cluster in
   high-volatility regimes; this cuts exposure automatically going *into* them rather than
   after the damage. Expect a 20–30% reduction in max drawdown at similar CAGR.
2. **Equity-curve throttle.** When equity falls below its own 50-trade moving average,
   halve risk until it recovers above it. This is a trend filter applied to your own
   returns; it cannot prevent the first leg of a drawdown but it materially truncates the
   deep ones. Cost: slower recovery, since you re-enter at half size. Worth it only if the
   deep tail is what would make you quit.
3. **Strategy-level correlation cap.** Total open risk across S1+S2+S3 capped at 2R, and
   never S1 and S3 long-volatility in the same direction at once. Most catastrophic
   drawdowns are not one bad trade — they are five correlated positions discovering they
   were the same trade.

### Position sizing improvements

- **Fractional-Kelly framing.** Section 9 puts full Kelly at 10.7% for the S1 profile.
  Trade **1/10 Kelly (~1%)**, which is where you already are. Quarter-Kelly (2.7%) is
  defensible with 500+ trades of evidence; anything above half-Kelly is gambling with a
  spreadsheet.
- **Risk *down* after losses, never up.** After 3 consecutive losses, drop to 0.5% until
  a winner. Never Martingale — the bankroll assumption it requires does not exist.
- **Size on ATR, not on a fixed percent.** Already implemented: the stop is `k × ATR`, so
  position size self-adjusts to volatility. This is the difference between risking 1% and
  *thinking* you are risking 1%.
- **Cap leverage below the venue maximum.** The 3× cap is deliberate. It makes tight-stop
  trades take *less* than 1% risk rather than more leverage — the engine reports these as
  leverage-capped so the reduction is visible, not silent.

---

## 11. Macro-based strategy

Macro drives the regime BTC and Nasdaq trade in — it does not produce 1h entries. The
right architecture is **macro sets the exposure, technicals set the trade.**

### How each factor transmits

**Interest rates (the dominant one).** Real rates are the discount rate on every long-
duration asset. BTC and Nasdaq are the two longest-duration assets most people own, so
both are inversely sensitive to the real 10-year yield and to changes in rate *expectations*
(watch the 2-year — it moves first). Rising real rates compress multiples and drain
speculative liquidity; falling real rates do the reverse. **The tradable signal is the
change in expectations, not the level.**

**Inflation.** Matters through its effect on policy, and the relationship is non-monotonic:
- *Falling inflation + growth holding* = the best regime for both assets (easing without recession). Maximum risk-on.
- *Rising inflation* = hawkish repricing = risk-off for both, despite the "inflation hedge" story. BTC has empirically traded as a high-beta liquidity asset, not as gold.
- *Very high inflation with currency instability* = the debasement regime where BTC decouples upward. Rare, and mostly an emerging-market phenomenon.

**Economic growth.** Growth surprises lift earnings (Nasdaq) and risk appetite (BTC).
The dangerous quadrant is *falling growth + rising inflation* (stagflation): both assets
fall and correlation goes to 1, which is exactly when a BTC+Nasdaq portfolio's
diversification disappears.

### The regime matrix

| Growth | Inflation | Policy path | BTC / Nasdaq stance |
|---|---|---|---|
| Up | Down | Easing | **Maximum long.** Full weights; run S1 aggressively. |
| Up | Up | Hiking | Neutral-to-long equities, reduce crypto. Trade both directions. |
| Down | Down | Cutting | Long duration; buy weakness. Volatile but the turn is near. |
| Down | Up | Stuck (stagflation) | **Minimum exposure.** Cash. Correlations converge; diversification fails. |

### Entry / exit signals

**Entry (risk-on):** 2-year yield falling for 3+ consecutive weeks *and* core inflation
below its 3-month average *and* the credit spread (HYG/LQD ratio) not making new lows.
Trigger the *technical* entry with S1 in the direction macro allows.

**Exit / de-risk:** any of — real 10-year yield up >40bp in a month; DXY up >3% in a
month (dollar strength drains crypto liquidity); a scheduled CPI or FOMC release within 4
hours (flatten short-horizon mean-reversion, keep trend positions with stops widened);
BTC perp funding above ~0.1%/8h for 2+ days (leveraged crowd is one-sided, mean-reversion
risk is high).

**Practical implementation for a $100 account:** macro should change *one* thing — your
risk multiplier. Maintain a simple 0/1/2 macro score (rates falling? inflation cooling?
growth holding?) and scale per-trade risk to 0.5% / 1.0% / 1.5% accordingly. Trying to
trade macro directly at this size is not viable; using it to size technical trades is.

### Example trades

1. **Dovish CPI surprise.** CPI prints below consensus; 2-year yield drops 15bp within
   the hour. *Do not* chase the first 1h candle — that move is liquidity, not information.
   Wait for the S1 Donchian-55 breakout to confirm on the *following* bars with ADX ≥ 20,
   enter at the next open, stop 2×ATR (wider than usual — post-event ATR is elevated, and
   the sizing formula handles it automatically). Macro score 2 → risk 1.5%.
2. **Hawkish repricing.** Fed dots shift higher; DXY breaks a 3-month high. Macro score 0
   → risk 0.5%, and disable S2 long fades entirely (buying dips into a liquidity drain is
   how mean-reversion systems die). Trade S1 short-side only, EMA200 filter already aligned.
3. **Stagflation quadrant.** Growth data missing while inflation re-accelerates. Cut the
   portfolio's BTC sleeve from 15% to 5%, stand down S2 and S3 completely, keep S1 at
   0.5% risk in both directions. The goal in this quadrant is to still be solvent when
   the regime turns, not to make money in it.

---

## 12. Alpha / edge detection — underexploited opportunities in crypto

### Behavioural inefficiencies

1. **Round-number and liquidation-cluster magnetism.** Retail places stops at round
   numbers ($100k, $95k) and exchanges publish liquidation heatmaps. Price is drawn toward
   dense liquidation clusters because filling those orders is *profitable for whoever can
   push it there*. This is not a conspiracy theory, it is order-book mechanics.
2. **Funding-rate crowding.** Perpetual funding is a direct, public measure of crowd
   positioning. Extreme positive funding means longs are paying heavily to stay long — a
   crowded, leveraged, fragile position.
3. **Weekend liquidity holes.** Crypto trades 24/7 but market-maker balance sheets do not.
   Weekend books are thinner, moves overshoot, and Monday often retraces.
4. **Time-zone handoff effects.** The Asia→Europe and Europe→US handovers produce
   repeatable volatility and reversal patterns that no one is paid to arbitrage away.

### Market-structure gaps

1. **Perp/spot basis and funding are mechanically linked**, and the arbitrage is capital-
   intensive — which means it does *not* get fully arbitraged at small size. Small traders
   can take the other side of crowded funding without competing with size.
2. **Cross-exchange latency and fragmentation.** No consolidated tape. Price discovery
   leads on some venues and lags on others, persistently.
3. **Altcoin listing and index-inclusion flow.** Predictable, announced, mechanical flows
   that the institutional world trades routinely in equities and largely ignores in crypto.
4. **Stablecoin flows as a liquidity gauge.** Aggregate stablecoin supply change is a
   genuine measure of dry powder entering the system, published on-chain and free.

### Strategy A — Funding-Extreme Fade with structural confirmation

**The idea.** When perpetual funding is extremely positive, longs pay shorts continuously.
That is a crowded, leveraged, one-sided position — precisely the setup for a long squeeze.
The trade is a short (or exit of longs) triggered by *funding*, confirmed by *price*.

**Why most traders miss it.** Three reasons, all structural: (1) funding data lives in a
different API from price data, so it doesn't appear on a chart by default; (2) fading
extreme funding means shorting an asset that is ripping upward, which is emotionally
almost impossible; (3) it fires rarely — a few times a year — and most traders cannot sit
on a strategy that doesn't trade.

**Execution, step by step**
1. Poll 8h funding (`/fapi/v1/fundingRate`) and compute its 90-day percentile.
2. **Arm** when funding > 95th percentile for ≥ 2 consecutive periods (≥16h of sustained crowding).
3. **Confirm** with price: wait for a 1h close below the prior 24-bar low. *Never* short on funding alone — crowded can stay crowded for weeks.
4. **Enter** at the next bar's open. Stop `2.0 × ATR(14)` above entry. Size at 1% risk.
5. **Target** the 20-bar VWAP or a 3R trail, whichever comes first. Exit immediately if funding normalises below its 50th percentile — the thesis is gone.
6. **Hard rule:** no more than one position at a time, and stand down entirely for 72h after a loss. Squeeze setups cluster, and so do the failures.

**Expected shape:** 6–12 trades/year, ~40% win rate, 2.5–3R average win. The rarity is the
edge — it is uncompetitive precisely because it is unprofitable for anyone who needs to
trade daily.

### Strategy B — Weekend Liquidity-Hole Reversion

**The idea.** From Friday 22:00 UTC to Sunday 22:00 UTC, market-maker inventory capacity
shrinks. The same order flow moves price further, so weekend extremes overshoot, and a
disproportionate share retraces when Monday liquidity returns.

**Why most traders miss it.** (1) It requires being at a desk on a weekend, when almost
all discretionary traders and every institutional desk are off; (2) it looks like "fading
a breakout," which trend-followers are trained never to do; (3) the effect is modest per
trade and only pays through repetition and strict risk control; (4) it is *seasonal in the
week*, so it does not show up in an undifferentiated backtest that treats every hour alike.

**Execution, step by step**
1. Compute the 20-bar Bollinger (2.5σ) and ATR(14) as in S2.
2. **Window filter:** only trade entries between Friday 22:00 and Sunday 20:00 UTC.
3. **Liquidity filter:** current 24h volume < 0.8× the 30-day average (confirm the hole is actually there — a high-volume weekend is a news weekend, and this trade must not be taken).
4. **Entry:** close beyond the 2.5σ band with RSI(2) < 5 (long) or > 95 (short); fill next open.
5. **Stop:** `1.5 × ATR`. **Target:** the 20-bar mean. **Hard exit: Monday 00:00 UTC regardless of P&L** — the edge is the liquidity hole, and it closes when the hole does.
6. **Veto:** skip entirely if a macro release or major unlock lands on the following Monday.

**Expected shape:** 20–40 trades/year, ~60% win rate, ~1:1 payoff. Thin per trade — which
is why step 3's liquidity filter and the low-fee requirement from section 0 are not
optional. At 7bps/side this is marginal; at 2bps/side (maker) it works.

**Both are testable with the harness here.** Strategy B runs as an S2 variant with a
time-window filter today. Strategy A needs the funding endpoint added to `quant/data.py` —
the fetcher is straightforward and the entry logic reuses `Order` unchanged.

---

## What to do next, in order

1. **`python3 tests/test_engine.py`** — never trust a backtester you haven't tested.
2. **Get the data.** `python3 run_all.py` in an environment with network access, or drop a
   CSV in and use `--csv`. Binance serves 1h BTCUSDT back to Aug-2017 — the full window.
3. **Read the walk-forward section first, not the backtest section.** The overfitting tax
   tells you whether anything here is real.
4. **Fix fees before fixing signals.** Section 3 shows a fee-tier change is worth more
   than most strategy improvements, and it carries no risk.
5. **Paper-trade for 50+ trades** and compare realised slippage against the 2bps
   assumption. If real slippage is 10bps, every number in this document changes.
6. **Then, and only then, consider funding it** — at a size where the friction table
   above is survivable.

---

### Limitations, stated plainly

- No backtest numbers on real data are presented anywhere in this document, because no
  market data was reachable from this environment. Sections 3, 7, 9 and 10 are exact
  computations on **stated assumptions**, labelled as such; they are real mathematics
  about hypothetical inputs, not measurements of BTC.
- Section 4 (current regime) and section 8 (live setups) are deliberately unanswered as
  point-in-time calls. The tools that answer them are built and run in one command.
- Historical performance, simulated or real, does not establish future performance. Every
  edge described here can decay, and crypto market structure changes faster than most.
- This is research tooling and analysis, not investment advice. You own the decisions.
