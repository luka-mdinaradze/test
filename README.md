# Quant Research Harness — 1h Crypto Strategies

A backtesting and risk-analysis system for three 1h BTC/USDT strategies, sized for a
small account (defaults: $100, 1% risk per trade).

**[REPORT.md](REPORT.md) is the analysis.** This file is how to run the code.

## Quick start

```bash
pip install -r requirements.txt
python3 tests/test_engine.py     # verify the engine first — always
python3 run_all.py               # full pipeline, live data if reachable
```

`run_all.py` prints every section of the brief: backtest table, friction analysis,
regime read, multi-factor model, walk-forward optimisation, portfolio construction,
trade setups, Monte Carlo, and drawdown episodes.

### Options

```bash
python3 run_all.py --csv BTCUSDT_1h.csv   # your own OHLCV (open_time/open/high/low/close/volume)
python3 run_all.py --synthetic            # force the simulator
python3 run_all.py --quick                # skip walk-forward (the slow part, ~30s)
python3 run_all.py --equity 1000 --risk 0.02 --fee-bps 2 --slip-bps 1
python3 analyze_profiles.py               # assumption-driven risk math, no data needed
```

## Data

`quant/data.py` tries Binance (1h BTCUSDT back to Aug-2017), then falls back to a
regime-switching simulator. **Results computed on simulated bars are stamped
`SYNTHETIC` everywhere they appear** — they validate the plumbing, never a strategy.
Fetched data is cached under `data/`.

If Binance is blocked in your environment, export 1h OHLCV from any source and use `--csv`.

## Layout

| File | Contents |
|---|---|
| `quant/engine.py` | Bar-by-bar execution: next-bar fills, pessimistic intrabar, gaps, fees, slippage, min-notional, risk sizing |
| `quant/strategies.py` | S1 trend breakout, S2 mean reversion, S3 squeeze expansion |
| `quant/indicators.py` | EMA/RSI/ATR/ADX/Donchian/Bollinger/Keltner/squeeze — all strictly causal |
| `quant/metrics.py` | CAGR, Sharpe, Sortino, Calmar, Ulcer, drawdown episodes with recovery times |
| `quant/friction.py` | Cost-in-R arithmetic and break-even win rates (no data required) |
| `quant/montecarlo.py` | Path simulation, ruin probability, losing-streak distribution |
| `quant/optimize.py` | Walk-forward with in-sample vs out-of-sample gap |
| `quant/portfolio.py` | BTC + Nasdaq + cash allocation, fat-tailed drawdown simulation |
| `quant/multifactor.py` | Cross-sectional momentum / value / volatility / trend |
| `quant/regime.py` | Trend / volatility / volume classification and strategy selection |
| `quant/setups.py` | Entry, stop, target, R:R and position size from the latest bar |
| `tests/test_engine.py` | Seven correctness tests on hand-built bars |

## What the engine guarantees

Each of these is enforced in code and verified by a test:

- A signal at bar *t* fills at bar *t+1*'s **open** — never the signal bar's close.
- A bar touching both stop and target books the **stop**.
- A bar that gaps through the stop fills at the **open** (a −2R loss, not −1R).
- Fees and slippage are charged on **both** sides; a flat round trip loses money.
- Orders below the venue's minimum notional are **rejected and counted**.
- A stop-out costs exactly the configured risk fraction.

## Extending it

Add a strategy by writing a builder that returns `(orders, atr, context)` where `orders`
is a Series of `engine.Order` objects, then register it in `strategies.STRATEGIES`. The
engine, metrics, Monte Carlo and walk-forward all work on it unchanged.

## Caveat

Research tooling, not investment advice. Simulated and historical results do not
establish future performance. See the Limitations section at the end of REPORT.md.
