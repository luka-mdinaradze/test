"""Tests for the live bot: risk gates, state durability, reconciliation, and the
invariant that an open position is never left without a stop order.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.broker import Order, PaperBroker  # noqa: E402
from bot.config import TradingConfig  # noqa: E402
from bot.risk import can_open, effective_risk_pct, promotion_ready  # noqa: E402
from bot.state import BotState, PendingEntry, Position  # noqa: E402
from bot.trader import Trader  # noqa: E402
from quant.data import synthetic_ohlc  # noqa: E402

CFG = TradingConfig()
PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f": {detail}" if detail and not cond else
                                                    (f" — {detail}" if detail else "")))


def fresh_state(**kw):
    st = BotState(**kw)
    st.day_key = "2024-01-01"
    return st


# ---------------------------------------------------------------- state ----
def test_state_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = str(Path(d) / "s.json")
        st = fresh_state(equity=123.45)
        st.position = Position("long", 0.001, 50000, 49000, 49000, "t", 3, 1.0, "c1", "s1")
        st.pending_entry = None
        st.record_trade({"phase": "PAPER", "pnl": -1.0, "r_multiple": -1.0})
        st.save(p)
        back = BotState.load(p)
        ok = (back.equity == 123.45 and back.position is not None
              and back.position.entry_price == 50000 and len(back.trades) == 1
              and back.consecutive_losses == 1)
        check("state survives a save/load round trip", ok)
        # An atomic write leaves no partial files behind.
        leftovers = [f for f in Path(d).iterdir() if f.suffix == ".tmp"]
        check("atomic write leaves no temp files", not leftovers, str(leftovers))


# ----------------------------------------------------------------- risk ----
def test_risk_blocks_when_halted():
    st = fresh_state(halted=True, halt_reason="test")
    d = can_open(CFG, st, 50000, 49000, 100)
    check("halted bot cannot open", not d.allowed, d.reason)


def test_risk_blocks_on_drawdown():
    st = fresh_state(equity=79.0, peak_equity=100.0)        # -21%, limit 20%
    d = can_open(CFG, st, 50000, 49000, 79)
    check("max-drawdown breach blocks entry", not d.allowed, d.reason)


def test_risk_blocks_on_daily_loss():
    st = fresh_state()
    st.day_realised_R = -3.5
    d = can_open(CFG, st, 50000, 49000, 100)
    check("daily loss limit blocks entry", not d.allowed, d.reason)


def test_risk_blocks_double_position():
    st = fresh_state()
    st.position = Position("long", 0.001, 50000, 49000, 49000, "t", 1, 1.0, "c1")
    d = can_open(CFG, st, 50000, 49000, 100)
    check("cannot open a second position", not d.allowed, d.reason)
    st.position = None
    st.pending_entry = PendingEntry("c2", "long", 50000, 0.001, 1000, 1, "t")
    d = can_open(CFG, st, 50000, 49000, 100)
    check("cannot stack an entry on a working order", not d.allowed, d.reason)


def test_sizing_risks_one_percent():
    st = fresh_state()
    d = can_open(CFG, st, 50000.0, 49000.0, 100.0)      # $1000 stop distance
    # 1% of $100 = $1 risk; $1 / $1000 = 0.001 BTC; notional $50 < $100 cap.
    check("sizing risks 1% of equity", d.allowed and abs(d.risk_amount - 1.0) < 0.02,
          f"risk=${d.risk_amount:.4f} qty={d.qty:.6f} notional=${d.notional:.2f}")


def test_leverage_cap_shrinks_size_on_spot():
    st = fresh_state()
    # A 0.5% stop implies $200 notional at $1 risk — more cash than we have.
    d = can_open(CFG, st, 50000.0, 49750.0, 100.0)
    ok = d.allowed and d.notional <= 100.0 + 1e-6 and d.risk_amount < 1.0
    check("spot leverage cap shrinks size (risk falls, never borrows)", ok,
          f"notional=${d.notional:.2f} risk=${d.risk_amount:.4f}")


def test_min_notional_blocks():
    cfg = TradingConfig(min_notional=50.0)
    st = fresh_state()
    d = can_open(cfg, st, 50000.0, 40000.0, 100.0)   # huge stop -> tiny notional
    check("sub-minimum notional is rejected", not d.allowed, d.reason)


def test_probation_and_derisk():
    st = fresh_state(phase="LIVE", live_trades_done=0)
    check("probation halves risk for the first live trades",
          effective_risk_pct(CFG, st) == CFG.probation_risk_pct)
    st2 = fresh_state(phase="PAPER")
    st2.consecutive_losses = 3
    check("loss streak halves risk", effective_risk_pct(CFG, st2) == CFG.risk_pct * 0.5)


def test_promotion_gate():
    st = fresh_state()
    ok, why = promotion_ready(CFG, st)
    check("promotion blocked with no record", not ok, why)
    for _ in range(CFG.min_paper_trades):
        st.trades.append({"phase": "PAPER", "r_multiple": -0.2, "pnl": -1})
    ok, why = promotion_ready(CFG, st)
    check("promotion blocked on negative expectancy", not ok, why)
    st.trades = [{"phase": "PAPER", "r_multiple": 0.3, "pnl": 1}
                 for _ in range(CFG.min_paper_trades)]
    ok, why = promotion_ready(CFG, st)
    check("promotion allowed on a good record", ok, why)


# --------------------------------------------------------------- broker ----
def test_client_order_id_is_idempotent():
    bars = synthetic_ohlc(bars=100, seed=1)
    b = PaperBroker(bars, warmup=10)
    o1 = b.place_limit_maker("BTCUSDT", "buy", 0.001, 100.0, "same-id")
    o2 = b.place_limit_maker("BTCUSDT", "buy", 0.001, 100.0, "same-id")
    check("resubmitting a client_order_id does not duplicate the order",
          o1 is o2 and len(b.orders) == 1)


# -------------------------------------------------- reconcile & invariant --
def test_reconcile_detects_offline_stopout():
    bars = synthetic_ohlc(bars=800, seed=2)
    b = PaperBroker(bars, warmup=600)
    st = fresh_state()
    st.position = Position("long", 0.001, float(bars["close"].iloc[600]),
                           float(bars["close"].iloc[600]) * 0.98,
                           float(bars["close"].iloc[600]) * 0.98, "t", 1, 1.0, "c1", "s1")
    with tempfile.TemporaryDirectory() as d:
        cfg = TradingConfig(state_path=str(Path(d) / "s.json"))
        t = Trader(b, cfg, st)
        b.base = 0.0                     # venue says we hold nothing
        t.reconcile()
        check("reconcile books a stop-out that happened while offline",
              st.position is None and len(st.trades) == 1,
              f"pos={st.position} trades={len(st.trades)}")


def test_reconcile_replaces_missing_stop():
    bars = synthetic_ohlc(bars=800, seed=3)
    b = PaperBroker(bars, warmup=600)
    px = float(bars["close"].iloc[600])
    st = fresh_state()
    st.position = Position("long", 0.001, px, px * 0.98, px * 0.98, "t", 1, 1.0, "c1", "missing")
    b.base = 0.001                       # we DO hold it, but no stop order exists
    with tempfile.TemporaryDirectory() as d:
        cfg = TradingConfig(state_path=str(Path(d) / "s.json"))
        t = Trader(b, cfg, st)
        t.reconcile()
        has_stop = any(o.type == "stop_loss" and o.status == "new"
                       for o in b.orders.values())
        check("reconcile re-places a missing stop on an open position",
              has_stop and st.position is not None)


def test_full_paper_run_invariants():
    """Drive the bot across a long simulated series and assert the invariants
    hold on EVERY bar, not just at the end."""
    bars = synthetic_ohlc(bars=9000, seed=5)
    b = PaperBroker(bars, start_equity=100.0, warmup=700)
    with tempfile.TemporaryDirectory() as d:
        cfg = TradingConfig(state_path=str(Path(d) / "s.json"))
        st = fresh_state()
        t = Trader(b, cfg, st)

        naked_bars = 0
        worst_dd = 0.0
        over_risk = 0
        while b.advance():
            df = b.klines(cfg.symbol, cfg.interval, 900)
            t.on_bar(df)
            if st.position is not None:
                live_stop = any(o.type == "stop_loss" and o.status == "new"
                                for o in b.orders.values())
                if not live_stop:
                    naked_bars += 1
                if st.position.risk_amount > st.equity * cfg.risk_pct * 1.5:
                    over_risk += 1
            worst_dd = min(worst_dd, st.drawdown)

        check("INVARIANT: an open position always has a live stop order",
              naked_bars == 0, f"{naked_bars} bars with an unprotected position")
        check("INVARIANT: position risk never exceeds the configured fraction",
              over_risk == 0, f"{over_risk} oversized positions")
        check("INVARIANT: drawdown never breaches the halt limit by much",
              worst_dd >= -(cfg.max_drawdown_pct + 0.10),
              f"worst drawdown {worst_dd*100:.1f}%")
        check("bot actually traded (the test is meaningful)", len(st.trades) > 5,
              f"{len(st.trades)} trades")
        print(f"     -> {len(st.trades)} trades, final equity ${st.equity:.2f}, "
              f"worst drawdown {worst_dd*100:.1f}%, halted={st.halted}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", ", ".join(FAILED))
    sys.exit(1 if FAILED else 0)
