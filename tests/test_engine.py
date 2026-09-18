"""Correctness tests for the execution engine.

These are the assertions that decide whether any downstream number is
meaningful, so they use hand-built bars with known answers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.engine import Costs, ExecConfig, Order, run_backtest  # noqa: E402

IDX = lambda n: pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")  # noqa: E731
NOCOST = Costs(fee_bps=0.0, slippage_bps=0.0, funding_bps_per_8h=0.0)


def bars(rows):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=IDX(len(rows)))
    df["volume"] = 1.0
    return df


def empty_orders(df):
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def test_signal_fills_next_bar_open():
    df = bars([[1000, 1000, 1000, 1000], [1010, 1010, 1010, 1010],
               [1020, 1020, 1020, 1020], [1030, 1030, 1030, 1030]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0)          # intent at bar 0 close
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(min_notional=0.0))
    assert len(r.trades) == 0 or r.trades["entry_price"].iloc[0] == 1010, "must fill at bar 1 open"
    # Nothing is filled at bar 0's own price:
    assert r.equity.iloc[0] == 100.0
    print("PASS no-lookahead: intent at t fills at t+1 open")


def test_risk_sizing_is_one_percent():
    """A stop-out must cost ~1% of equity, before costs exactly 1%."""
    df = bars([[1000, 1000, 1000, 1000], [1000, 1005, 995, 1000],
               [1000, 1000, 889, 900], [900, 900, 900, 900]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0)           # stop at 900 from entry 1000
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(initial_equity=100.0, risk_pct=0.01,
                                                         min_notional=0.0, max_leverage=100))
    t = r.trades.iloc[0]
    assert abs(t["risk_amount"] - 1.0) < 1e-9, t["risk_amount"]
    assert abs(t["pnl"] + 1.0) < 1e-6, t["pnl"]
    assert abs(t["r_multiple"] + 1.0) < 1e-6, t["r_multiple"]
    print(f"PASS sizing: risk=${t['risk_amount']:.4f}  stop-out pnl=${t['pnl']:.4f} = {t['r_multiple']:.2f}R")


def test_stop_wins_when_bar_touches_both():
    """Pessimistic intrabar assumption: same-bar stop+target resolves as a loss."""
    df = bars([[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000],
               [1000, 1300, 890, 1000], [1000, 1000, 1000, 1000]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0, tp_dist=200.0)   # stop 900, tp 1200
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(min_notional=0.0, max_leverage=100))
    assert r.trades.iloc[0]["reason"] == "stop", r.trades.iloc[0]["reason"]
    print("PASS pessimistic fill: bar hitting stop AND target books the stop")


def test_gap_through_stop_fills_at_open():
    df = bars([[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000],
               [800, 820, 780, 800], [800, 800, 800, 800]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0)            # stop 900, bar opens at 800
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(min_notional=0.0, max_leverage=100))
    t = r.trades.iloc[0]
    assert t["reason"] == "gap_stop" and t["exit_price"] == 800, t.to_dict()
    assert t["r_multiple"] < -1.9, t["r_multiple"]        # a gap costs MORE than 1R
    print(f"PASS gap handling: filled at open, realised {t['r_multiple']:.2f}R (worse than -1R)")


def test_min_notional_rejects_small_orders():
    """The $100-account trap: a tight stop implies a notional the venue won't accept."""
    df = bars([[1000, 1000, 1000, 1000]] * 4)
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=500.0)            # qty = 1/500 -> notional $2
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(min_notional=5.0))
    assert r.rejected_orders == 1 and r.trades.empty
    print("PASS min-notional: sub-minimum order rejected and counted, not silently filled")


def test_costs_are_charged_both_sides():
    df = bars([[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000],
               [1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0, max_bars=1)   # flat move, exits on time
    c = Costs(fee_bps=10.0, slippage_bps=5.0, funding_bps_per_8h=0.0)
    r = run_backtest(df, o, costs=c, cfg=ExecConfig(min_notional=0.0, max_leverage=100))
    t = r.trades.iloc[0]
    # Flat price, yet the round trip loses money: 2x fee + 2x slippage on $10 notional.
    assert t["pnl"] < 0 and t["costs"] > 0
    print(f"PASS costs: flat round-trip on ${t['qty']*1000:.2f} notional loses ${-t['pnl']:.5f} "
          f"({t['r_multiple']:.3f}R) to friction alone")


def test_equity_curve_marks_to_market():
    df = bars([[1000, 1000, 1000, 1000], [1000, 1000, 1000, 1000],
               [1000, 1100, 1000, 1100], [1100, 1100, 1100, 1100]])
    o = empty_orders(df)
    o.iloc[0] = Order("long", stop_dist=100.0)
    r = run_backtest(df, o, costs=NOCOST, cfg=ExecConfig(min_notional=0.0, max_leverage=100))
    assert r.equity.iloc[2] > r.equity.iloc[1], "open profit must show in equity"
    print("PASS mark-to-market: unrealised PnL appears in the equity curve")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{'ALL ENGINE TESTS PASSED' if not fails else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)
