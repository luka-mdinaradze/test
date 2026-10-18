"""Tests for the tradebot platform: fills, sizing, risk governor, sessions, feeds,
the run loop, state recovery, and the live adapters' order logic (with fakes).

    python3 -m pytest tests/test_tradebot.py -q
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tradebot import config as cfgmod  # noqa: E402
from tradebot.brokers.base import BrokerError  # noqa: E402
from tradebot.brokers.ccxt_broker import CCXTBroker  # noqa: E402
from tradebot.brokers.paper import PaperBroker  # noqa: E402
from tradebot.config import AccountSection, Config, InstrumentSection, SessionSection  # noqa: E402
from tradebot.feeds import CCXTFeed, load_csv  # noqa: E402
from tradebot.models import LONG, SHORT, Bar, Exit, Signal  # noqa: E402
from tradebot.risk import Governor, GovernorState, size_position  # noqa: E402
from tradebot.runner import Runner  # noqa: E402
from tradebot.session import Session  # noqa: E402
from tradebot.strategy import Strategy, load_strategy  # noqa: E402

FREE = InstrumentSection(point_value=1.0, tick_size=0.25, qty_step=1, min_qty=1, max_qty=100,
                         commission_per_unit=0.0, fee_bps=0.0, slippage_ticks=0, slippage_bps=0.0)
MNQ = InstrumentSection(point_value=2.0, tick_size=0.25, qty_step=1, min_qty=1, max_qty=5,
                        commission_per_unit=0.62, fee_bps=0.0, slippage_ticks=1, slippage_bps=0.0)


def bar(i, o, h, l, c):
    return Bar(pd.Timestamp("2025-01-06 10:00", tz="UTC") + pd.Timedelta(minutes=15 * i), o, h, l, c, 1.0)


# --------------------------------------------------------------------------- paper fills

def test_paper_fills_at_next_bar_open_not_signal_close():
    b = PaperBroker(FREE, 10_000)
    assert b.open_bracket(LONG, 1, 5.0, None, "x", ref_price=100, ts="t") == []
    ev = b.on_bar(bar(1, 103, 104, 102, 103))
    assert ev[0].kind == "opened" and ev[0].price == 103 and ev[0].stop_price == 98


def test_paper_stop_wins_when_bar_touches_both():
    b = PaperBroker(FREE, 10_000)
    b.open_bracket(LONG, 1, 5.0, 5.0, "x", 100, "t")
    b.on_bar(bar(1, 100, 100, 100, 100))
    ev = b.on_bar(bar(2, 100, 106, 94, 100))
    assert ev[0].reason == "stop" and ev[0].pnl == -5.0


def test_paper_gap_through_stop_fills_at_open():
    b = PaperBroker(FREE, 10_000)
    b.open_bracket(LONG, 1, 5.0, None, "x", 100, "t")
    b.on_bar(bar(1, 100, 101, 99, 100))
    ev = b.on_bar(bar(2, 90, 91, 89, 90))
    assert ev[0].reason == "gap_stop" and ev[0].price == 90 and ev[0].pnl == -10.0


def test_paper_target_needs_trade_through():
    inst = replace(FREE, target_through_ticks=1)
    b = PaperBroker(inst, 10_000)
    b.open_bracket(SHORT, 1, 5.0, 10.0, "x", 100, "t")
    b.on_bar(bar(1, 100, 100, 100, 100))
    assert b.on_bar(bar(2, 100, 100, 90, 92)) == []              # touched 90, not through
    ev = b.on_bar(bar(3, 92, 92, 89.75, 90))
    assert ev[0].reason == "target" and ev[0].price == 90 and ev[0].pnl == 10.0


def test_paper_costs_both_sides():
    b = PaperBroker(MNQ, 5_000)
    b.open_bracket(LONG, 1, 10.0, None, "x", 20000, "t")
    b.on_bar(bar(1, 20000, 20000, 20000, 20000))                  # fill 20000.25 (1 tick slippage)
    b.close_position("test", "y", 20000, "t")
    ev = b.on_bar(bar(2, 20010, 20010, 20010, 20010))             # exit 20009.75
    assert abs(ev[0].pnl - ((20009.75 - 20000.25) * 2 - 2 * 0.62)) < 1e-9


# --------------------------------------------------------------------------- sizing

def test_mnq_whole_contracts_and_refusal():
    acct = AccountSection(risk_pct=0.5)
    s = size_position(5000, 10.0, 20000, MNQ, acct)              # (10+0.25)*2 + 1.24 = 21.74
    assert s.qty == 1 and abs(s.risk_amount - 21.74) < 1e-9
    s = size_position(5000, 12.0, 20000, MNQ, acct)              # 25.74 > 25
    assert s.qty == 0 and "refused" in s.note


def test_fractional_sizing_never_exceeds_budget():
    inst = InstrumentSection(qty_step=0.001, min_qty=0.001, max_qty=10, fee_bps=5, slippage_bps=1)
    for stop in (50, 300, 1200):
        s = size_position(5000, stop, 60000, inst, AccountSection(risk_pct=0.5))
        assert 0 < s.risk_amount <= 25.0 + 1e-9


# --------------------------------------------------------------------------- governor

def test_governor_daily_loss_and_consecutive_losses_reset_next_day():
    g = Governor(AccountSection(daily_loss_limit_pct=2.0, max_consecutive_losses=3))
    g.roll_day("d1", 5000)
    g.record_trade(-60)
    assert g.allows_entry()[0]
    g.record_trade(-50)                                            # -110 > 100 (2%)
    ok, why = g.allows_entry()
    assert not ok and "daily loss" in why
    g.roll_day("d2", 4890)
    assert g.allows_entry()[0]
    for _ in range(3):
        g.record_trade(-1)
    ok, why = g.allows_entry()
    assert not ok and "consecutive" in why


def test_governor_drawdown_hard_halt_survives_new_day(tmp_path):
    g = Governor(AccountSection(max_drawdown_pct=10), kill_file=tmp_path / "STOP")
    g.roll_day("d1", 5000)
    g.roll_day("d2", 4400)
    assert "HALTED" in g.allows_entry()[1]
    g.roll_day("d3", 4500)
    assert "HALTED" in g.allows_entry()[1]


def test_kill_switch(tmp_path):
    g = Governor(AccountSection(), kill_file=tmp_path / "STOP")
    g.roll_day("d1", 5000)
    assert g.allows_entry()[0]
    (tmp_path / "STOP").write_text("x")
    assert "kill switch" in g.allows_entry()[1]


# --------------------------------------------------------------------------- session

def test_session_window_wraps_midnight_and_flatten_rule():
    s = Session(SessionSection(timezone="America/New_York", trade_start="18:15", trade_end="16:40",
                               flatten_at="16:50", weekdays=[0, 1, 2, 3, 4, 6]), 15)
    et = lambda t: pd.Timestamp(t, tz="America/New_York").tz_convert("UTC")  # noqa: E731
    assert s.entry_allowed(et("2025-01-06 19:00"))[0]              # evening session
    assert s.entry_allowed(et("2025-01-07 10:00"))[0]
    assert not s.entry_allowed(et("2025-01-07 16:30"))[0]          # ends 16:45, < 1 bar to flatten
    assert not s.entry_allowed(et("2025-01-07 17:15"))[0]          # halt
    assert s.must_flatten(et("2025-01-07 16:30"))                  # next bar ends 17:00 > 16:50
    assert not s.must_flatten(et("2025-01-07 16:15"))
    assert not s.entry_allowed(et("2025-01-11 10:00"))[0]          # Saturday


# --------------------------------------------------------------------------- feeds

def test_csv_resamples_finer_data(tmp_path):
    idx = pd.date_range("2025-01-01", periods=60, freq="1min", tz="UTC")
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": np.arange(60.0),
                       "high": np.arange(60.0) + 1, "low": np.arange(60.0) - 1, "close": np.arange(60.0) + 0.5,
                       "volume": 1.0})
    df.to_csv(tmp_path / "m1.csv", index=False)
    out = load_csv(str(tmp_path / "m1.csv"), 15)
    assert len(out) == 4 and out.iloc[0]["open"] == 0 and out.iloc[0]["close"] == 14.5 and out.iloc[0]["high"] == 15


class FakeExchange:
    """Just enough of a ccxt exchange for the adapter and feed tests."""

    def __init__(self, now_ms=None):
        self.orders, self.cancels, self.pos, self.bal = [], 0, [], 10_000.0
        self.fail_stop = False
        self.now_ms = now_ms

    def load_markets(self):
        pass

    def market(self, s):
        return {"swap": True, "contractSize": 1.0, "settle": "USDT"}

    def amount_to_precision(self, s, a):
        return f"{np.floor(a * 1000) / 1000:.3f}"

    def price_to_precision(self, s, p):
        return f"{round(p, 1):.1f}"

    def fetch_balance(self):
        return {"total": {"USDT": self.bal}}

    def fetch_positions(self, syms):
        return self.pos

    def create_order(self, sym, typ, side, amt, price=None, params=None):
        self.orders.append(("order", typ, side, amt, dict(params or {})))
        if not (params or {}).get("reduceOnly"):
            self.pos = [{"symbol": sym, "contracts": amt, "side": "long" if side == "buy" else "short", "entryPrice": 100.0}]
        else:
            self.pos = []
        return {"id": str(len(self.orders)), "average": 100.0}

    def create_stop_loss_order(self, sym, typ, side, amt, price, stop, params):
        if self.fail_stop:
            raise RuntimeError("rejected")
        self.orders.append(("stop", typ, side, amt, stop, dict(params)))

    def create_take_profit_order(self, sym, typ, side, amt, price, tp, params):
        self.orders.append(("target", typ, side, amt, tp, dict(params)))

    def cancel_all_orders(self, sym):
        self.cancels += 1

    def fetch_ohlcv(self, sym, tf, limit=None):
        base = pd.Timestamp("2025-01-01 00:00", tz="UTC")
        rows = [[int((base + pd.Timedelta(minutes=15 * i)).timestamp() * 1000), 1, 2, 0, 1.5, 10] for i in range(4)]
        return rows


def _ccxt_cfg():
    c = Config()
    c.market.broker, c.market.feed, c.market.symbol, c.bot.mode = "ccxt", "ccxt", "BTC/USDT:USDT", "live"
    return c


def test_ccxt_bracket_places_reduce_only_stop_and_target():
    ex = FakeExchange()
    b = CCXTBroker(_ccxt_cfg(), ex)
    b.connect()
    ev = b.open_bracket(LONG, 0.05, 2.0, 4.0, "cid1", 100.0, "t")
    kinds = [o[0] for o in ex.orders]
    assert kinds == ["order", "stop", "target"]
    assert ex.orders[1][4] == 98.0 and ex.orders[2][4] == 104.0
    assert ex.orders[1][5]["reduceOnly"] and ex.orders[2][5]["reduceOnly"]
    assert ev[0].kind == "opened" and ev[0].stop_price == 98.0


def test_ccxt_stop_rejection_closes_position_and_raises():
    ex = FakeExchange()
    ex.fail_stop = True
    b = CCXTBroker(_ccxt_cfg(), ex)
    b.connect()
    with pytest.raises(BrokerError):
        b.open_bracket(LONG, 0.05, 2.0, None, "cid1", 100.0, "t")
    assert ex.orders[-1][4].get("reduceOnly") and ex.pos == []        # flattened at once


def test_ccxt_detects_venue_exit_and_cancels_other_leg():
    ex = FakeExchange()
    b = CCXTBroker(_ccxt_cfg(), ex)
    b.connect()
    b.open_bracket(LONG, 0.05, 2.0, 4.0, "cid1", 100.0, "t")
    ex.pos, ex.bal = [], 9_990.0                                      # stop hit on the venue
    ev = b.on_bar(bar(1, 98, 98, 97, 97))
    assert ev[0].kind == "closed" and ev[0].pnl == -10.0 and ex.cancels >= 1


def test_ccxt_feed_drops_the_forming_candle(monkeypatch):
    ex = FakeExchange()
    feed = CCXTFeed("BTC/USDT:USDT", "15m", 15, exchange=ex)
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2025-01-01 00:50", tz="UTC")))
    h = feed.history(10)
    assert list(h.index.strftime("%H:%M")) == ["00:00", "00:15", "00:30"]   # 00:45 bar still forming


def test_ibkr_bracket_structure():
    pytest.importorskip("ib_async")
    from tradebot.brokers.ibkr import bracket_orders
    ids = iter(range(100, 110))
    o = bracket_orders(lambda: next(ids), SHORT, 2, 20100.0, 19900.0, "ref")
    assert [type(x).__name__ for x in o] == ["MarketOrder", "LimitOrder", "StopOrder"]
    assert o[0].action == "SELL" and o[1].action == o[2].action == "BUY"
    assert o[1].parentId == o[2].parentId == o[0].orderId
    assert [x.transmit for x in o] == [False, False, True] and o[2].tif == "GTC"


# --------------------------------------------------------------------------- run loop

class Scripted(Strategy):
    """Goes long on bar index `at`, with stop and target distances given."""
    warmup = 1
    defaults = {"at": 3, "stop": 5.0, "target": 10.0, "max_bars": None, "exit_at": None}

    def on_bar(self, ctx):
        i = len(ctx.bars) - 1
        if ctx.position is None and i == self.params["at"]:
            return Signal.long(self.params["stop"], self.params["target"], max_bars=self.params["max_bars"], reason="test")
        if ctx.position is not None and i == self.params["exit_at"]:
            return Exit("scripted exit")
        return None


def _bt_cfg(tmp_path, closes, highs=None, lows=None, session=None):
    idx = pd.date_range("2025-01-06 00:00", periods=len(closes), freq="15min", tz="UTC")
    c = np.array(closes, float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": o,
                       "high": highs if highs is not None else np.maximum(o, c),
                       "low": lows if lows is not None else np.minimum(o, c), "close": c, "volume": 1.0})
    df.to_csv(tmp_path / "bars.csv", index=False)
    cfg = Config()
    cfg.bot.mode, cfg.bot.state_dir, cfg.bot.history_bars = "backtest", str(tmp_path / "run"), 50
    cfg.market.csv_path, cfg.market.timeframe = str(tmp_path / "bars.csv"), "15m"
    cfg.instrument = FREE
    cfg.account = AccountSection(starting_equity=10_000, risk_pct=0.5)
    if session:
        cfg.session = session
    return cfg


def test_backtest_long_hits_target_with_expected_r(tmp_path):
    closes = [100] * 5 + [103, 106, 111, 112, 112]
    cfg = _bt_cfg(tmp_path, closes)
    r = Runner(cfg, strategy=Scripted(), printer=lambda *_: None)
    s = r.backtest()
    t = r.trades[0]
    assert s["trades"] == 1 and t["exit_reason"] == "target"
    assert t["entry"] == 100 and t["exit"] == 110 and t["qty"] == 10       # $50 budget / $5 per unit
    assert abs(t["r"] - 2.0) < 1e-9


def test_template_strategy_never_trades(tmp_path):
    cfg = _bt_cfg(tmp_path, list(100 + np.sin(np.arange(200) / 5) * 10))
    s = Runner(cfg, strategy=load_strategy("template"), printer=lambda *_: None).backtest()
    assert s["trades"] == 0 and s["signals_total"] == 0


def test_time_exit_and_strategy_exit(tmp_path):
    cfg = _bt_cfg(tmp_path, [100] * 20)
    r = Runner(cfg, strategy=Scripted({"max_bars": 3}), printer=lambda *_: None)
    r.backtest()
    assert r.trades[0]["exit_reason"].startswith("time exit")
    r2 = Runner(_bt_cfg(tmp_path, [100] * 20), strategy=Scripted({"exit_at": 6}), printer=lambda *_: None)
    r2.backtest()
    assert r2.trades[0]["exit_reason"] == "scripted exit"


def test_session_flatten_closes_before_cutoff(tmp_path):
    sess = SessionSection(timezone="UTC", trade_start="00:00", trade_end="23:59", flatten_at="01:20")
    cfg = _bt_cfg(tmp_path, [100] * 20, session=sess)
    r = Runner(cfg, strategy=Scripted({"at": 2}), printer=lambda *_: None)
    r.backtest()
    t = r.trades[0]
    assert t["exit_reason"] == "session flatten" and pd.Timestamp(t["closed_at"]) <= pd.Timestamp("2025-01-06 01:20", tz="UTC")


def test_oversized_signal_is_rejected_not_rounded(tmp_path):
    cfg = _bt_cfg(tmp_path, [100] * 10)
    cfg.account = AccountSection(starting_equity=100, risk_pct=0.5)         # $0.50 budget, 1 unit risks $5
    r = Runner(cfg, strategy=Scripted(), printer=lambda *_: None)
    s = r.backtest()
    assert s["trades"] == 0 and s["signals_rejected_size"] == 1


def test_state_survives_restart_and_reconcile_halts_on_unknown_position(tmp_path):
    cfg = _bt_cfg(tmp_path, [100] * 10)
    cfg.bot.mode, cfg.market.broker = "paper", "paper"
    r = Runner(cfg, strategy=Scripted({"at": 3}), printer=lambda *_: None)
    r.run()                                                    # replays the CSV through the live loop
    d = r.store.load()
    assert d["last_bar"] and "governor" in d
    # a fresh runner sees the broker holding a position the bot never recorded
    r2 = Runner(cfg, strategy=Scripted(), printer=lambda *_: None)
    r2.restore()
    r2.position = None
    r2.broker.pos = {"side": 1, "qty": 1, "entry": 100, "stop": 95, "target": None, "cid": "x", "opened": "t"}
    msgs = r2.reconcile()
    assert msgs and "did not" in r2.gov.state.halted_reason


# --------------------------------------------------------------------------- config & CLI

def test_config_rejects_unknown_keys_and_unsafe_settings(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[bot]\nmode = "live"\n[market]\nbroker = "paper"\nfeed = "csv"\ncsv_path = "x.csv"\n')
    with pytest.raises(ValueError, match="live mode needs a real broker"):
        cfgmod.load(p)
    p.write_text('[account]\nrisk_pct = 3\nmax_risk_pct = 3\n[market]\ncsv_path="x"\n')
    with pytest.raises(ValueError, match="2%"):
        cfgmod.load(p)
    p.write_text('[bot]\nmodee = "paper"\n')
    with pytest.raises(ValueError, match="unknown keys"):
        cfgmod.load(p)


def test_live_run_refuses_without_explicit_flag(tmp_path, capsys):
    from tradebot.__main__ import main
    p = tmp_path / "live.toml"
    p.write_text('[bot]\nmode = "live"\nstate_dir = "r"\n[market]\nbroker = "ccxt"\nfeed = "ccxt"\n')
    assert main(["run", "--config", str(p)]) == 2
    assert "--i-understand-live" in capsys.readouterr().out


def test_new_strategy_scaffold_loads(tmp_path, monkeypatch):
    from tradebot import __main__ as cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "template.py").write_text(
        (Path(cli.__file__).parent / "strategies" / "template.py").read_text())
    assert cli.main(["new-strategy", "my_idea"]) == 0
    src = (tmp_path / "strategies" / "my_idea.py").read_text()
    assert "class MyIdeaStrategy" in src and 'name = "my_idea"' in src


def test_live_loop_through_ccxt_adapter_records_venue_stop(tmp_path):
    """The live path: entry fills immediately, the stop fills on the venue, the bot
    notices on the next bar, cancels the other leg and books the account's P&L."""
    ex = FakeExchange()
    cfg = _ccxt_cfg()
    cfg.bot.state_dir = str(tmp_path / "live")
    cfg.instrument = InstrumentSection(qty_step=0.001, min_qty=0.001, max_qty=5, fee_bps=0, slippage_bps=0)
    broker = CCXTBroker(cfg, ex)
    closes = [100.0] * 8
    idx = pd.date_range("2025-01-06", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0}, index=idx)

    class ListFeed:
        def history(self, n):
            return df.iloc[:3]

        def stream(self):
            for i in range(3, 8):
                if i == 6:                       # venue stop filled between bars
                    ex.pos, ex.bal = [], ex.bal - 25.0
                yield Bar(idx[i], *df.iloc[i][["open", "high", "low", "close", "volume"]])

    r = Runner(cfg, strategy=Scripted({"at": 3, "stop": 2.0, "target": 4.0}), broker=broker, feed=ListFeed(),
               printer=lambda *_: None)
    r.run()
    assert [o[0] for o in ex.orders[:3]] == ["order", "stop", "target"]
    assert len(r.trades) == 1 and r.trades[0]["pnl"] == -25.0 and "venue" in r.trades[0]["exit_reason"]
    assert ex.cancels >= 1 and r.position is None
