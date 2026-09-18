#!/usr/bin/env python3
"""Behavioural fingerprint of each strategy.

Separates what is RULE-DRIVEN (holding time, exit mix, filter attrition, trade
frequency, streak structure - these carry over to real data because the rules
determine them) from what is MARKET-DRIVEN (win rate, expectancy, equity curve -
these need real prices and are NOT read from here).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import data as D
from quant import indicators as ind
from quant.engine import Costs, ExecConfig, run_backtest
from quant.strategies import (S1Params, S2Params, S3Params, s1_trend_breakout,
                              s2_mean_reversion, s3_squeeze_expansion)

pd.set_option("display.width", 200)
df = D.synthetic_ohlc() if "--synthetic" in sys.argv or True else D.load_btc_1h()
costs, cfg = Costs(), ExecConfig()
BUILDERS = [("S1 trend", s1_trend_breakout, S1Params()),
            ("S2 mean-rev", s2_mean_reversion, S2Params()),
            ("S3 squeeze", s3_squeeze_expansion, S3Params())]

print("=" * 92)
print("FILTER ATTRITION - how many raw signals each gate destroys (RULE-DRIVEN)")
print("=" * 92)

dc = ind.donchian(df, 55); ema200 = ind.ema(df["close"], 200)
adx = ind.adx(df, 14)["adx"]; rv_rank = ind.percentile_rank(ind.realized_vol(df["close"], 24), 500)
raw = (df["close"] > dc["upper"]) | (df["close"] < dc["lower"])
step1 = raw & (((df["close"] > dc["upper"]) & (df["close"] > ema200)) |
               ((df["close"] < dc["lower"]) & (df["close"] < ema200)))
step2 = step1 & (adx >= 20); step3 = step2 & (rv_rank >= 0.35)
n = len(df)
for label, mask in [("raw 55-bar breakouts", raw), ("+ EMA200 alignment", step1),
                    ("+ ADX >= 20", step2), ("+ vol pct >= 0.35", step3)]:
    k = int(mask.sum())
    print(f"  S1 {label:<24} {k:>6,} bars ({k/n*100:5.2f}% of all bars)"
          f"{'' if label == 'raw 55-bar breakouts' else f'  -> keeps {k/int(raw.sum())*100:4.1f}% of raw'}")

bb = ind.bollinger(df["close"], 20, 2.5); rsi2 = ind.rsi(df["close"], 2)
raw2 = (df["close"] < bb["lower"]) | (df["close"] > bb["upper"])
s2a = raw2 & (((df["close"] < bb["lower"]) & (rsi2 < 5)) | ((df["close"] > bb["upper"]) & (rsi2 > 95)))
s2b = s2a & (adx <= 18)
print()
for label, mask in [("raw 2.5-sigma touches", raw2), ("+ RSI(2) extreme", s2a), ("+ ADX <= 18", s2b)]:
    k = int(mask.sum())
    print(f"  S2 {label:<24} {k:>6,} bars ({k/n*100:5.2f}% of all bars)"
          f"{'' if 'raw' in label else f'  -> keeps {k/int(raw2.sum())*100:4.1f}% of raw'}")

print("\n" + "=" * 92)
print("TRADE BEHAVIOUR (RULE-DRIVEN: holding time, exit mix, frequency, exposure)")
print("=" * 92)
rows = []
for name, fn, p in BUILDERS:
    orders, atr_s, _ = fn(df, p)
    res = run_backtest(df, orders, atr_s, costs, cfg)
    t = res.trades
    if t.empty:
        continue
    mix = t["reason"].value_counts(normalize=True) * 100
    exposure = t["bars_held"].sum() / len(df) * 100
    years = len(df) / 8760
    rows.append({
        "strategy": name,
        "trades/yr": round(len(t) / years, 1),
        "median_hold_h": round(t["bars_held"].median(), 1),
        "mean_hold_h": round(t["bars_held"].mean(), 1),
        "p90_hold_h": round(t["bars_held"].quantile(0.9), 1),
        "time_in_market_%": round(exposure, 1),
        "long_%": round((t["side"] == "long").mean() * 100, 1),
        "init_stop_%": round(mix.get("stop", 0) + mix.get("gap_stop", 0), 1),
        "trail_out_%": round(mix.get("trail_stop", 0), 1),
        "target_%": round(mix.get("take_profit", 0), 1),
        "time_exit_%": round(mix.get("time_exit", 0), 1),
        "gap_stop_%": round(mix.get("gap_stop", 0), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 92)
print("R-MULTIPLE SHAPE - how wins and losses are distributed (SHAPE is rule-driven,")
print("                   the win RATE is market-driven and NOT readable here)")
print("=" * 92)
for name, fn, p in BUILDERS:
    orders, atr_s, _ = fn(df, p)
    res = run_backtest(df, orders, atr_s, costs, cfg)
    t = res.trades
    if t.empty:
        continue
    r = t["r_multiple"]
    print(f"\n{name}:  n={len(t)}")
    print(f"  worst {r.min():6.2f}R | p10 {r.quantile(.1):5.2f}R | median {r.median():5.2f}R | "
          f"p90 {r.quantile(.9):5.2f}R | best {r.max():6.2f}R")
    print(f"  skew {r.skew():+.2f}  (positive = a few big winners carry it; "
          f"negative = many small wins, rare big losses)")
    big = (r > 2).mean() * 100
    print(f"  {big:.1f}% of trades exceed +2R | {(r < -1).mean()*100:.1f}% lose more than 1R "
          f"(cost drag on a full stop-out; gaps would make this worse)")
    # Streaks actually realised
    sign = (r > 0).astype(int)
    grp = (sign != sign.shift()).cumsum()
    runs = sign.groupby(grp).agg(["first", "size"])
    lose = runs[runs["first"] == 0]["size"]
    win = runs[runs["first"] == 1]["size"]
    print(f"  longest losing streak {int(lose.max()) if len(lose) else 0} | "
          f"longest winning streak {int(win.max()) if len(win) else 0}")

print("\n" + "=" * 92)
print("NOTE: run on SYNTHETIC bars. Holding times, exit mix, filter attrition and")
print("trade frequency carry over to real data (the rules produce them). Win rate,")
print("expectancy and the equity curve DO NOT - those need real prices.")
print("=" * 92)
