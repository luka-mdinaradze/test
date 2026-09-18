#!/usr/bin/env python3
"""Run the full research pipeline and print every section of the brief.

    python3 run_all.py                 # live data if reachable, else simulator
    python3 run_all.py --csv btc.csv   # your own 1h OHLCV file
    python3 run_all.py --synthetic     # force the simulator
    python3 run_all.py --quick         # skip the walk-forward (slowest part)

Every section prints its data source. Sections computed on SYNTHETIC bars are
labelled as such and must not be read as evidence about Bitcoin.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from quant import data as D
from quant import friction, montecarlo, multifactor, portfolio, regime, setups
from quant.engine import Costs, ExecConfig, run_backtest
from quant.metrics import drawdown_table, summarize
from quant.optimize import param_grid, walk_forward
from quant.strategies import (S1Params, S2Params, S3Params, s1_trend_breakout,
                              s2_mean_reversion, s3_squeeze_expansion)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
BAR = "=" * 96


def head(n: int, title: str):
    print(f"\n{BAR}\n{n}. {title}\n{BAR}")


def load(args) -> pd.DataFrame:
    if args.csv:
        return D.load_csv(args.csv)
    if args.synthetic:
        return D.synthetic_ohlc()
    return D.load_btc_1h(prefer_live=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--equity", type=float, default=100.0)
    ap.add_argument("--risk", type=float, default=0.01)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    args = ap.parse_args()

    df = load(args)
    synth = bool(df.attrs.get("is_synthetic", False))
    src = df.attrs.get("source", "unknown")
    costs = Costs(fee_bps=args.fee_bps, slippage_bps=args.slip_bps)
    cfg = ExecConfig(initial_equity=args.equity, risk_pct=args.risk)

    print(BAR)
    print(f"DATA SOURCE : {src}")
    print(f"BARS        : {len(df):,}  ({df.index[0]} -> {df.index[-1]})")
    print(f"ACCOUNT     : ${cfg.initial_equity:.2f}, {cfg.risk_pct*100:g}% risk/trade, "
          f"max {cfg.max_leverage:g}x, ${cfg.min_notional:g} min notional")
    print(f"COSTS       : {costs.fee_bps:g}bps fee + {costs.slippage_bps:g}bps slippage per side")
    if synth:
        print("\n*** SYNTHETIC DATA. Numbers below validate the PIPELINE, not the strategies. ***")
        print("*** Re-run with --csv <real 1h OHLCV> for results that mean something.        ***")
    print(BAR)

    # ---------------------------------------------------------------- 2 -----
    head(2, "BACKTEST - three strategies, identical costs and sizing")
    builders = [("S1_trend_breakout", s1_trend_breakout, S1Params()),
                ("S2_mean_reversion", s2_mean_reversion, S2Params()),
                ("S3_squeeze_expansion", s3_squeeze_expansion, S3Params())]
    results, rows = {}, []
    for name, fn, p in builders:
        orders, atr_s, _ = fn(df, p)
        res = run_backtest(df, orders, atr_s, costs, cfg)
        results[name] = res
        rows.append(summarize(res, cfg.bars_per_year, name))
    table = pd.DataFrame(rows).set_index("strategy")
    cols = ["final_equity", "CAGR_pct", "Sharpe", "Sortino", "max_drawdown_pct", "Calmar",
            "trades", "win_rate_pct", "expectancy_R", "payoff_ratio", "profit_factor",
            "total_costs", "rejected_orders"]
    print(table[cols].round(3).to_string())

    # ---------------------------------------------------------------- 3 -----
    head(3, "RISK / REWARD - cost structure of risk-based sizing (data-independent)")
    print(f"Per-side cost assumed: {costs.per_side_bps:g}bps. Payoff 2:1.\n")
    print(friction.friction_table(per_side_bps=costs.per_side_bps, equity=cfg.initial_equity,
                                  risk_pct=cfg.risk_pct, min_notional=cfg.min_notional,
                                  max_lev=cfg.max_leverage).to_string(index=False))
    print("\nAnnual friction bill at different trade frequencies:")
    for sp, n in ((0.5, 900), (1.0, 500), (1.5, 300), (3.0, 120)):
        e = friction.edge_required(sp, costs.per_side_bps, n, cfg.risk_pct)
        print(f"  stop {sp:>4.1f}% x {n:>4} trades/yr -> {e['annual_cost_R']:6.1f}R/yr "
              f"= {e['annual_cost_pct_of_equity']:5.1f}% of equity consumed by friction alone")

    # ---------------------------------------------------------------- 4 -----
    head(4, "MARKET REGIME - current state of the loaded series")
    print(regime.render(regime.classify(df)))

    # ---------------------------------------------------------------- 5 -----
    head(5, "MULTI-FACTOR MODEL - momentum / value / volatility / trend")
    w = multifactor.FactorWeights()
    print("Factor weights:", {k: f"{v*100:.0f}%" for k, v in w.normalised().items()})
    uni = pd.DataFrame({"BTC": df["close"].resample("1D").last()}).dropna()
    if len(uni) > 600:
        rng = np.random.default_rng(3)
        # A stand-in universe so the cross-sectional machinery is exercised.
        for nm, drift, vol in (("ALT_A", 0.0004, 0.035), ("ALT_B", 0.0001, 0.02), ("ALT_C", -0.0002, 0.028)):
            uni[nm] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(uni))))
        cfgf = multifactor.FactorConfig(top_n=2)
        sc = multifactor.score(uni, w, cfgf)
        tw = multifactor.target_weights(sc, uni, cfgf)
        eq = multifactor.backtest(uni, tw)
        print(f"\nLatest composite scores:\n{sc.dropna().iloc[-1].round(3).to_string()}")
        print(f"\nLatest target weights:\n{(tw.iloc[-1] * 100).round(1).to_string()}  (% of capital)")
        print(f"\nRebalances: {len(tw)} ({cfgf.rebalance}); factor-portfolio final multiple: {eq.iloc[-1]:.2f}x")
        if synth:
            print("(universe is simulated - structure demo only)")

    # ---------------------------------------------------------------- 6 -----
    if not args.quick:
        head(6, "OPTIMISATION - in-sample grid vs walk-forward out-of-sample")
        cands = param_grid(S1Params(), donchian=[34, 55, 89], stop_atr=[1.5, 2.0, 2.5],
                           trail_atr=[2.5, 3.5], adx_min=[18.0, 25.0])
        print(f"Searching {len(cands)} parameter sets over 5 folds...")
        wf = walk_forward(df, s1_trend_breakout, cands, n_folds=5, costs=costs, cfg=cfg)
        if not wf["folds"].empty:
            f = wf["folds"]
            show = f[["fold", "oos_start", "oos_end", "is_sharpe", "oos_sharpe",
                      "oos_return_pct", "oos_maxdd_pct", "oos_trades"]].copy()
            num = show.select_dtypes("number").columns
            show[num] = show[num].round(3)
            print(show.to_string(index=False))
            print(f"\nMean IN-SAMPLE Sharpe : {wf['mean_is_sharpe']:.2f}")
            print(f"Mean OUT-OF-SAMPLE    : {wf['mean_oos_sharpe']:.2f}")
            print(f"Overfitting tax       : {wf['gap']:.2f} Sharpe points lost out of sample")
            base = summarize(results["S1_trend_breakout"], cfg.bars_per_year, "baseline")
            print(f"\nBEFORE (fixed default params, full sample): Sharpe {base['Sharpe']:.2f}, "
                  f"maxDD {base['max_drawdown_pct']:.1f}%")
            if len(wf["oos_equity"]) > 10:
                from quant.metrics import max_drawdown, sharpe as _sh
                print(f"AFTER  (walk-forward, honest OOS)        : Sharpe "
                      f"{_sh(wf['oos_equity'], cfg.bars_per_year):.2f}, "
                      f"maxDD {max_drawdown(wf['oos_equity']) * 100:.1f}%")

    # ---------------------------------------------------------------- 7 -----
    head(7, "PORTFOLIO - BTC + Nasdaq + cash")
    views = [
        portfolio.AssetView("BTC", 0.25, 0.60, -0.77,
                            "Highest expected return and the only real diversifier vs. equities "
                            "in a debasement scenario; also the largest drawdown source."),
        portfolio.AssetView("Nasdaq 100", 0.10, 0.22, -0.35,
                            "Long-run earnings growth with far deeper liquidity; carries the "
                            "portfolio when crypto is in a multi-year winter."),
        portfolio.AssetView("Cash / T-bills", 0.04, 0.01, -0.01,
                            "Pays a real yield, funds rebalancing into drawdowns, and is the "
                            "only asset that is certainly available at the bottom."),
    ]
    corr = np.array([[1.0, 0.45, 0.0], [0.45, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for profile in ("low", "medium", "high"):
        b = portfolio.build(views, corr, profile, horizon_years=2.0)
        s, sim = b["stats"], b["sim"]
        print(f"\n--- {profile.upper()} risk tolerance, 2-year horizon ---")
        print(b["table"][["asset", "weight_pct", "exp_return_pct", "vol_pct"]].to_string(index=False))
        print(f"  Expected return {s['exp_return']*100:5.1f}%/yr | vol {s['vol']*100:5.1f}% | "
              f"Sharpe {s['sharpe']:.2f}")
        print(f"  Simulated 2y (fat-tailed): median {sim['exp_total_return_p50']:+.1f}%, "
              f"5th pct {sim['total_return_p5']:+.1f}%, P(loss) {sim['prob_negative']:.0f}%")
        print(f"  Drawdown: median {sim['maxdd_median']:.0f}%, 95th pct {sim['maxdd_p95']:.0f}%, "
              f"worst {sim['maxdd_worst']:.0f}%")
    print("\nWhy each asset is included:")
    for v in views:
        print(f"  {v.name:15s} {v.rationale}")

    # ---------------------------------------------------------------- 8 -----
    head(8, "TRADE SETUPS - levels implied by the latest bar")
    print(setups.render(setups.generate(df, cfg.initial_equity, cfg.risk_pct,
                                        cfg.min_notional, cfg.max_leverage), top=3))

    # ---------------------------------------------------------------- 9 -----
    head(9, "MONTE CARLO")
    for name, res in results.items():
        if res.trades.empty or len(res.trades) < 20:
            print(f"\n{name}: only {len(res.trades)} trades - too few to resample.")
            continue
        mc = montecarlo.simulate(res.trades["r_multiple"].to_numpy(), n_trades=300,
                                 n_sims=20_000, risk_pct=cfg.risk_pct,
                                 initial_equity=cfg.initial_equity)
        print(f"\n--- {name} (resampling {len(res.trades)} real trade outcomes, "
              f"300-trade horizon) ---")
        print(montecarlo.summary_frame(mc).to_string(index=False))
        print(f"VERDICT: {montecarlo.robustness_verdict(mc)}")

    print("\n--- Assumption-driven MC (no backtest needed): 45% win rate, +2R wins, -1R losses ---")
    rng = np.random.default_rng(5)
    r = montecarlo.parametric_r(0.45, 2.0, 1.0, 5000, rng)
    mc = montecarlo.simulate(r, n_trades=300, n_sims=20_000, risk_pct=cfg.risk_pct,
                             initial_equity=cfg.initial_equity)
    print(montecarlo.summary_frame(mc).to_string(index=False))
    print(f"VERDICT: {montecarlo.robustness_verdict(mc)}")

    # --------------------------------------------------------------- 10 -----
    head(10, "DRAWDOWN ANALYSIS")
    for name, res in results.items():
        dd = drawdown_table(res.equity, top=3)
        print(f"\n--- {name} ---")
        if dd.empty:
            print("  no drawdown episodes (no trades taken)")
            continue
        view = dd.copy()
        view["depth_pct"] = view["depth_pct"].round(2)
        view["recovery_hours"] = view["recovery_bars"]
        print(view[["start", "trough", "depth_pct", "to_trough_bars", "recovery_hours"]].to_string(index=False))
        rec = dd["recovery_bars"].dropna()
        if not rec.empty:
            print(f"  average recovery: {rec.mean():.0f} hours ({rec.mean()/24:.1f} days)")

    print(f"\n{BAR}\nDone. Data source was: {src}")
    if synth:
        print("REMINDER: synthetic bars. Load real 1h OHLCV before trusting any number above.")
    print(BAR)


if __name__ == "__main__":
    main()
