"""Correctness tests for nqlab.engine on hand-built bars with known answers.

    python3 -m pytest tests/test_nqlab_engine.py -q
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import data  # noqa: E402
from nqlab.config import MNQ, NQ_SESSION, Account  # noqa: E402
from nqlab.engine import Market1m, Signals, label, simulate  # noqa: E402

FREE = replace(MNQ, commission_side=0.0, slip_ticks_rth=0.0, slip_ticks_eth=0.0, tp_through_ticks=0.0)
BIG = Account(equity=1e9, risk_pct=0.001, max_risk_pct=0.01, daily_loss_limit_pct=1.0,
              max_consecutive_losses=10**6, compounding=False)


def bars(rows, start="2024-03-12 10:00"):
    """1-minute RTH bars from [open, high, low, close] rows."""
    idx = pd.date_range(start, periods=len(rows), freq="1min", name="ts")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 100.0
    return data.prepare(df, NQ_SESSION)


def one(side, stop, tp=0.0, hold=10_000, at=0):
    return Signals([at], [side], [stop], [tp], [hold])


def run(m1, sig, inst=FREE, acct=BIG, fractional=False):
    return simulate(Market1m.build(m1, inst), sig, inst, acct, fractional=fractional)


def test_fill_is_next_bar_open_not_signal_close():
    m1 = bars([[100, 100, 100, 100], [110, 111, 109, 110], [110, 111, 109, 110]])
    t = run(m1, one(+1, stop=5.0)).trades.iloc[0]
    assert t.entry_i == 1 and t.entry == 110.0


def test_slippage_is_adverse_on_entry_and_stop():
    m1 = bars([[100, 100, 100, 100], [100, 100.5, 99.5, 100], [100, 100, 94, 95]])
    inst = replace(FREE, slip_ticks_rth=2.0)
    t = run(m1, one(+1, stop=5.0), inst=inst).trades.iloc[0]
    assert t.entry == 100.5                       # bought 2 ticks above the open
    assert t.reason == "stop" and t.exit == 95.0  # stop 95.5 filled 2 ticks lower


def test_stop_wins_when_bar_touches_both():
    m1 = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 120, 80, 100]])
    t = run(m1, one(+1, stop=10.0, tp=10.0)).trades.iloc[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_gap_through_stop_fills_at_open():
    m1 = bars([[100, 100, 100, 100], [100, 101, 99, 100], [80, 82, 79, 81]])
    t = run(m1, one(+1, stop=10.0)).trades.iloc[0]
    assert t.reason == "gap_stop" and t.exit == 80.0 and t.r < -1.9


def test_target_needs_trade_through():
    touch = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 110, 99, 105], [105, 105, 105, 105]])
    inst = replace(FREE, tp_through_ticks=1.0)
    t = run(touch, one(+1, stop=5.0, tp=10.0, hold=3), inst=inst).trades.iloc[0]
    assert t.reason == "time"                     # touched 110, did not trade through
    thru = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 110.25, 99, 105]])
    t = run(thru, one(+1, stop=5.0, tp=10.0), inst=inst).trades.iloc[0]
    assert t.reason == "target" and t.exit == 110.0 and abs(t.r - 2.0) < 1e-9


def test_time_exit_at_close_after_hold_bars():
    m1 = bars([[100, 100, 100, 100]] + [[100, 101, 99, 100.5]] * 5)
    t = run(m1, one(+1, stop=5.0, hold=3)).trades.iloc[0]
    assert t.reason == "time" and t.exit_i == 3 and t.exit == 100.5


def test_flatten_before_session_end_never_holds_overnight():
    m1 = bars([[100, 100, 100, 100]] * 30, start="2024-03-12 16:35")
    r = run(m1, one(+1, stop=50.0))
    t = r.trades.iloc[0]
    assert t.reason == "flatten"
    assert m1.index[t.exit_i] == pd.Timestamp("2024-03-12 16:49")


def test_no_entry_inside_last_minutes_or_across_days():
    m1 = bars([[100, 100, 100, 100]] * 30, start="2024-03-12 16:35")
    pos = int(np.flatnonzero(m1.index == pd.Timestamp("2024-03-12 16:42"))[0])
    r = run(m1, one(+1, stop=5.0, at=pos))
    assert len(r.trades) == 0 and r.counts["session"] == 1


def test_overlapping_signal_is_dropped_and_counted():
    m1 = bars([[100, 100, 100, 100]] * 10)
    sig = Signals([0, 2, 6], [1, 1, 1], [5.0, 5.0, 5.0], [0, 0, 0], [4, 4, 4])
    r = run(m1, sig)
    assert len(r.trades) == 2 and r.counts["overlap"] == 1
    assert list(r.trades.entry_i) == [1, 7]


def test_signal_on_exit_bar_may_reenter_next_bar():
    m1 = bars([[100, 100, 100, 100]] * 10)
    sig = Signals([0, 3], [1, 1], [5.0, 5.0], [0, 0], [3, 3])  # first exits at bar 3 close
    r = run(m1, sig)
    assert list(r.trades.entry_i) == [1, 4]


def test_mnq_sizing_respects_risk_budget_and_rejects_too_wide():
    m1 = bars([[100, 100, 100, 100]] * 5)
    acct = Account(equity=5000, risk_pct=0.005, max_risk_pct=0.01)
    # stop 10 pts: (10 + 0.25 slip) * $2 + 2 * $0.62 = $21.74 per contract -> 1 contract
    r = run(m1, one(+1, stop=10.0, hold=2), inst=MNQ, acct=acct)
    t = r.trades.iloc[0]
    assert t.qty == 1 and abs(t.risk - 21.74) < 1e-9 and t.risk <= 25.0
    # stop 12 pts: $25.74 > $25 budget -> rejected, never rounded up
    r = run(m1, one(+1, stop=12.0, hold=2), inst=MNQ, acct=acct)
    assert len(r.trades) == 0 and r.counts["rejected_size"] == 1


def test_clean_stop_out_costs_exactly_one_r_with_costs():
    m1 = bars([[20000, 20000, 20000, 20000], [20000, 20001, 19999, 20000], [20000, 20000, 19980, 19985]])
    acct = Account(equity=5000, risk_pct=0.01)
    t = run(m1, one(+1, stop=8.0), inst=MNQ, acct=acct).trades.iloc[0]
    assert t.reason == "stop" and abs(t.r + 1.0) < 1e-9


def test_daily_loss_limit_blocks_rest_of_day():
    rows = [[100, 100, 100, 100]] + [[100, 100, 90, 95], [95, 95, 95, 95]] * 6
    m1 = bars(rows)
    # $10 budget, 2.5-pt stop on $2/pt = $5/contract -> 2 contracts, each loss exactly $10.
    # Two losses reach the -2% ($20) limit: the remaining four signals are blocked.
    acct = Account(equity=1000, risk_pct=0.01, daily_loss_limit_pct=0.02, max_consecutive_losses=99,
                   compounding=False)
    sig = Signals(np.arange(0, 12, 2), np.ones(6, int), np.full(6, 2.5), np.zeros(6), np.full(6, 50))
    r = run(m1, sig, acct=acct)
    assert len(r.trades) == 2 and r.counts["blocked_risk"] == 4
    assert abs(r.trades.pnl.sum() + 20.0) < 1e-9


def test_consecutive_loss_pause():
    rows = [[100, 100, 100, 100]] + [[100, 100, 90, 95], [95, 95, 95, 95]] * 6
    m1 = bars(rows)
    acct = Account(equity=1000, risk_pct=0.01, daily_loss_limit_pct=1.0, max_consecutive_losses=3)
    sig = Signals(np.arange(0, 12, 2), np.ones(6, int), np.full(6, 2.0), np.zeros(6), np.full(6, 50))
    r = run(m1, sig, acct=acct)
    assert len(r.trades) == 3 and r.counts["blocked_risk"] == 3


def test_label_matches_simulation_for_isolated_signal():
    m1 = bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 112, 99, 111], [111, 111, 111, 111]])
    sig = one(+1, stop=5.0, tp=10.0)
    lab = label(Market1m.build(m1, FREE), sig, FREE)
    t = run(m1, sig).trades.iloc[0]
    assert bool(lab.hit_tp.iloc[0]) and abs(lab.r.iloc[0] - t.r) < 1e-12


def _random_walk(n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range("2024-01-02", periods=n_days):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=390, freq="1min")
        steps = rng.choice([-0.5, -0.25, 0.25, 0.5], size=390)
        c = 20000 + np.cumsum(steps)
        o = np.r_[20000, c[:-1]]
        hi = np.maximum(o, c) + rng.choice([0, 0.25], size=390)
        lo = np.minimum(o, c) - rng.choice([0, 0.25], size=390)
        frames.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1.0}, index=idx))
    df = pd.concat(frames)
    df.index.name = "ts"
    return data.prepare(df, NQ_SESSION)


@pytest.mark.parametrize("inst,expect", [(FREE, "zero"), (MNQ, "negative")])
def test_random_entries_on_random_walk_have_no_edge(inst, expect):
    m1 = _random_walk()
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(np.flatnonzero(m1.entry_ok.to_numpy()), 4000, replace=False))
    sig = Signals(idx, rng.choice([-1, 1], len(idx)), np.full(len(idx), 4.0),
                  np.full(len(idx), 8.0), np.full(len(idx), 60))
    lab = label(Market1m.build(m1, inst), sig, inst)
    r = lab.r[lab.valid]
    se = r.std() / np.sqrt(len(r))
    if expect == "zero":
        assert abs(r.mean()) < 3 * se, (r.mean(), se)
    else:
        assert r.mean() < -3 * se, (r.mean(), se)
