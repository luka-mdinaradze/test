#!/usr/bin/env python3
"""Mechanical verdict on whether the bot is worth trading.

The thresholds below were fixed BEFORE any real data was seen. That ordering is
the whole point: a threshold chosen after looking at results is not a test, it
is a rationalisation. This script only applies them.

    python3 evaluate_bot.py --csv BTCUSDT_1h.csv
    python3 evaluate_bot.py                      # fetches live if reachable
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant import data as D
from quant import montecarlo
from quant.engine import Costs, ExecConfig, run_backtest
from quant.friction import cost_in_R
from quant.metrics import max_drawdown, sharpe, summarize
from quant.optimize import param_grid, walk_forward
from quant.strategies import S1Params, s1_trend_breakout

# --- PRE-REGISTERED THRESHOLDS ------------------------------------------
# Fixed in advance. Do not tune these to make a strategy pass.
CRITERIA = [
    ("sample_trades",   "Trades in sample",            ">=", 200,   "hard"),
    ("sample_years",    "Years of data",               ">=", 3.0,   "hard"),
    ("oos_sharpe",      "Walk-forward OOS Sharpe",     ">=", 0.50,  "hard"),
    ("overfit_tax",     "Overfitting tax (IS - OOS)",  "<=", 0.50,  "hard"),
    ("expectancy_R",    "Net expectancy per trade",    ">=", 0.10,  "hard"),
    ("profit_factor",   "Profit factor",               ">=", 1.20,  "hard"),
    ("max_dd_pct",      "Max drawdown",                "<=", 35.0,  "hard"),
    ("mc_prob_loss",    "MC P(loss over 300 trades)",  "<=", 25.0,  "hard"),
    ("mc_prob_ruin",    "MC P(-50% account)",          "<=", 2.0,   "hard"),
    ("edge_cost_ratio", "Gross edge / cost per trade", ">=", 3.0,   "soft"),
    ("trades_per_year", "Trades per year",             ">=", 20.0,  "soft"),
]


def passes(value: float, op: str, threshold: float) -> bool:
    if value is None or not np.isfinite(value):
        return False
    return value >= threshold if op == ">=" else value <= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--fee-bps", type=float, default=2.0, help="maker fee")
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--equity", type=float, default=100.0)
    a = ap.parse_args()

    if a.csv:
        df = D.load_csv(a.csv)
    elif a.synthetic:
        df = D.synthetic_ohlc()
    else:
        df = D.load_btc_1h()
    synth = bool(df.attrs.get("is_synthetic"))

    costs = Costs(fee_bps=a.fee_bps, slippage_bps=a.slip_bps)
    cfg = ExecConfig(initial_equity=a.equity, risk_pct=a.risk, max_leverage=1.0,
                     allow_short=False)

    print("=" * 84)
    print(f"EVALUATION  |  {df.attrs.get('source')}")
    print(f"{len(df):,} bars  {df.index[0]} -> {df.index[-1]}")
    print(f"costs {costs.fee_bps:g}bps fee + {costs.slippage_bps:g}bps slip per side | "
          f"risk {cfg.risk_pct*100:g}% | long-only spot")
    if synth:
        print("\n*** SYNTHETIC DATA - the verdict below is about the simulator, not BTC ***")
    print("=" * 84)

    # ---- 1. baseline backtest -----------------------------------------
    p = S1Params(allow_short=False)
    orders, atr_s, _ = s1_trend_breakout(df, p)
    res = run_backtest(df, orders, atr_s, costs, cfg)
    s = summarize(res, cfg.bars_per_year, "S1")
    years = len(df) / cfg.bars_per_year

    print("\n[1/3] BACKTEST")
    for k in ("final_equity", "CAGR_pct", "Sharpe", "max_drawdown_pct", "trades",
              "win_rate_pct", "expectancy_R", "payoff_ratio", "profit_factor",
              "total_costs", "rejected_orders"):
        v = s.get(k)
        print(f"  {k:<20} {v:>12.3f}" if isinstance(v, (int, float)) and np.isfinite(v)
              else f"  {k:<20} {v:>12}")

    if res.trades.empty:
        print("\nNo trades. Nothing to evaluate.")
        return 1

    # ---- 2. walk-forward ----------------------------------------------
    print("\n[2/3] WALK-FORWARD (the number that actually matters)")
    cands = param_grid(S1Params(allow_short=False), donchian=[34, 55, 89],
                       stop_atr=[1.5, 2.0, 2.5], trail_atr=[2.5, 3.5],
                       adx_min=[18.0, 25.0])
    wf = walk_forward(df, s1_trend_breakout, cands, n_folds=5, costs=costs, cfg=cfg)
    oos_sharpe = overfit_tax = float("nan")
    if not wf["folds"].empty:
        f = wf["folds"]
        show = f[["fold", "is_sharpe", "oos_sharpe", "oos_return_pct", "oos_maxdd_pct",
                  "oos_trades"]].copy()
        show[show.select_dtypes("number").columns] = show.select_dtypes("number").round(3)
        print(show.to_string(index=False))
        oos_sharpe = sharpe(wf["oos_equity"], cfg.bars_per_year) if len(wf["oos_equity"]) > 10 \
            else wf["mean_oos_sharpe"]
        overfit_tax = wf["gap"]
        print(f"  mean IS {wf['mean_is_sharpe']:+.2f} | mean OOS {wf['mean_oos_sharpe']:+.2f} "
              f"| stitched OOS Sharpe {oos_sharpe:+.2f} | tax {overfit_tax:+.2f}")
    else:
        print("  not enough data for a walk-forward")

    # ---- 3. monte carlo on the REAL trade log --------------------------
    print("\n[3/3] MONTE CARLO (resampling the actual trades)")
    mc = montecarlo.simulate(res.trades["r_multiple"].to_numpy(), n_trades=300,
                             n_sims=20_000, risk_pct=cfg.risk_pct,
                             initial_equity=cfg.initial_equity)
    print(montecarlo.summary_frame(mc).to_string(index=False))
    print(f"  {montecarlo.robustness_verdict(mc)}")

    # ---- verdict --------------------------------------------------------
    med_stop_pct = float((res.trades["entry_price"] - res.trades["stop_price"]).abs()
                         .div(res.trades["entry_price"]).median() * 100)
    cost_R = cost_in_R(med_stop_pct, costs.per_side_bps)
    gross_exp = float(s["expectancy_R"]) + cost_R

    values = {
        "sample_trades": float(s["trades"]),
        "sample_years": years,
        "oos_sharpe": oos_sharpe,
        "overfit_tax": overfit_tax,
        "expectancy_R": float(s["expectancy_R"]),
        "profit_factor": float(s["profit_factor"]) if np.isfinite(s["profit_factor"]) else 0.0,
        "max_dd_pct": abs(float(s["max_drawdown_pct"])),
        "mc_prob_loss": mc["prob_loss"] * 100,
        "mc_prob_ruin": mc["prob_ruin"] * 100,
        "edge_cost_ratio": gross_exp / cost_R if cost_R > 0 else np.inf,
        "trades_per_year": float(s["trades_per_year"]),
    }

    print("\n" + "=" * 84)
    print("VERDICT  (thresholds fixed before the data was seen)")
    print("=" * 84)
    hard_fail, soft_fail = [], []
    for key, label, op, thr, kind in CRITERIA:
        v = values.get(key)
        ok = passes(v, op, thr)
        if not ok:
            (hard_fail if kind == "hard" else soft_fail).append(label)
        vs = f"{v:.3f}" if v is not None and np.isfinite(v) else "n/a"
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<32} {vs:>10}  (need {op} {thr})"
              + ("" if kind == "hard" else "   [soft]"))

    print()
    if hard_fail:
        print(f"RESULT: DO NOT TRADE. Failed {len(hard_fail)} hard criteria:")
        for x in hard_fail:
            print(f"   - {x}")
        print("\nA failure here is information, not a setback: the costs of finding out")
        print("in a backtest are zero and the costs of finding out live are not.")
    elif soft_fail:
        print("RESULT: PROCEED TO PAPER, with reservations. Soft criteria failed:")
        for x in soft_fail:
            print(f"   - {x}")
    else:
        print("RESULT: PROCEED TO PAPER TRADING.")
        print("This is NOT a green light for real money. It says the historical record")
        print("clears the bar; forward performance on unseen bars is still unproven.")
        print("Next: python3 -m bot.run paper   -> 40+ trades -> then reassess.")
    if synth:
        print("\n(Reminder: synthetic data. This verdict means nothing about BTC.)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
