"""Tests for the control panel and the process controls behind its buttons.

    python3 -m pytest tests/test_dashboard.py -q
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tradebot import config as cfgmod  # noqa: E402
from tradebot import control  # noqa: E402
from tradebot.runner import Runner  # noqa: E402

st_testing = pytest.importorskip("streamlit.testing.v1")
SAMPLE = ROOT / "examples" / "data" / "btcusd_15m_2025H1.csv"
APP = str(ROOT / "tradebot" / "dashboard.py")

COMMON = """
[instrument]
point_value = 1.0
tick_size = 0.1
qty_step = 0.001
min_qty = 0.001
max_qty = 1.0
fee_bps = 5.0
slippage_bps = 1.0

[account]
starting_equity = 5000
risk_pct = 0.5

[session]
timezone = "UTC"

[strategy]
name = "example_ema_cross"
"""


def write_config(d: Path, name: str, mode: str, csv: Path, broker: str = "paper", feed: str = "csv") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.toml"
    p.write_text(f'[bot]\nname = "{name}"\nmode = "{mode}"\nstate_dir = "runs/{name}"\nhistory_bars = 300\n\n'
                 f'[market]\nbroker = "{broker}"\nfeed = "{feed}"\nsymbol = "BTC/USDT:USDT"\ntimeframe = "15m"\n'
                 f'csv_path = "{csv}"\n' + COMMON)
    return p


@pytest.fixture
def small_csv(tmp_path):
    p = tmp_path / "btc_small.csv"
    pd.read_csv(SAMPLE, nrows=2500).to_csv(p, index=False)
    return p


def app(monkeypatch, config_dir: Path):
    monkeypatch.setenv("TRADEBOT_CONFIG_DIR", str(config_dir))
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_backtest_button_matches_the_cli_backtest(tmp_path, small_csv, monkeypatch):
    cfg_path = write_config(tmp_path / "configs", "bt", "backtest", small_csv)
    at = app(monkeypatch, tmp_path / "configs")
    at.button(key="run_backtest").click().run()
    assert not at.exception, at.exception
    expected = Runner(cfgmod.load(cfg_path), state_dir=tmp_path / "cli", printer=lambda *_: None).backtest()
    shown = {m.label: m.value for m in at.metric}
    assert shown["გარიგებები"] == str(expected["trades"])
    assert expected["trades"] > 0
    assert shown["Max drawdown"] == f"{expected['max_drawdown_pct']:.1f}%"


def test_dashboard_bot_tab_buttons_follow_mode(tmp_path, small_csv, monkeypatch):
    d = tmp_path / "configs"
    write_config(d, "a_paper", "paper", small_csv)
    write_config(d, "b_live", "live", small_csv, broker="ccxt", feed="ccxt")
    at = app(monkeypatch, d)                                   # first config alphabetically: paper
    assert not at.button(key="bot-start").disabled
    assert at.button(key="bot-stop").disabled and at.button(key="bot-kill").disabled
    at.selectbox(key="config").select(d / "b_live.toml").run()
    assert not at.exception, at.exception
    assert at.button(key="bot-start").disabled                 # live needs the confirmation box first
    at.checkbox(key="live-confirm").check().run()
    assert not at.button(key="bot-start").disabled


def test_dashboard_renders_with_a_broken_strategy_file(tmp_path, small_csv, monkeypatch):
    strat_dir = ROOT / "tradebot" / "strategies"
    broken = strat_dir / "zz_broken_for_test.py"
    broken.write_text("this is not python\n")
    try:
        write_config(tmp_path / "configs", "bt", "backtest", small_csv)
        at = app(monkeypatch, tmp_path / "configs")
        at.selectbox(key="bt-strategy").select("zz_broken_for_test").run()
        assert not at.exception, at.exception
        assert any("zz_broken_for_test" in e.value for e in at.error)
    finally:
        broken.unlink()
        sys.modules.pop("tradebot.strategies.zz_broken_for_test", None)


def wait_for(cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_control_starts_and_stops_a_paper_bot_process(tmp_path):
    cfg = cfgmod.load(write_config(tmp_path / "configs", "proc", "paper", SAMPLE))
    pid = control.start(cfg)
    try:
        assert wait_for(lambda: control.running_pid(cfg) == pid, 10)
        with pytest.raises(RuntimeError, match="already running"):
            control.start(cfg)
        assert wait_for(lambda: (cfg.state_dir / "state.json").exists())
        assert control.stop_force(cfg)
        assert wait_for(lambda: control.running_pid(cfg) is None)
    finally:
        if control.running_pid(cfg):
            os.kill(pid, 9)
    assert not control.pid_file(cfg).exists()
    assert "stopped by user" in (cfg.state_dir / "console.log").read_text()


def test_control_refuses_backtest_configs_and_unconfirmed_live(tmp_path, small_csv):
    d = tmp_path / "configs"
    with pytest.raises(RuntimeError, match="Backtest"):
        control.start(cfgmod.load(write_config(d, "bt", "backtest", small_csv)))
    with pytest.raises(RuntimeError, match="confirmation"):
        control.start(cfgmod.load(write_config(d, "lv", "live", small_csv, broker="ccxt", feed="ccxt")))


def test_graceful_stop_writes_the_kill_switch(tmp_path, small_csv):
    cfg = cfgmod.load(write_config(tmp_path / "configs", "ks", "paper", small_csv))
    control.stop_graceful(cfg)
    assert (cfg.state_dir / "STOP").exists()
    assert control.running_pid(cfg) is None and not control.stop_force(cfg)
