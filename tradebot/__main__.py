"""Command line.

    python3 -m tradebot backtest  --config configs/backtest_btc.toml
    python3 -m tradebot run       --config configs/paper_btc_binance.toml
    python3 -m tradebot run       --config configs/live_btc_binance_testnet.toml --i-understand-live
    python3 -m tradebot check     --config ...      # validate config + strategy (+ connect, read-only, for live)
    python3 -m tradebot status    --config ...
    python3 -m tradebot stop      --config ...      # kill switch: no new trades, flatten, exit
    python3 -m tradebot resume    --config ...      # remove the kill switch
    python3 -m tradebot reset-halt --config ...     # clear a HARD halt after a human has looked
    python3 -m tradebot new-strategy my_idea        # copy the template to tradebot/strategies/my_idea.py
    python3 -m tradebot dashboard                   # control panel in the browser
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import config as config_mod

ROOT = Path(__file__).resolve().parent


def _logging(cfg) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(cfg.state_dir / "bot.log")])


def cmd_backtest(a) -> int:
    from .runner import Runner
    cfg = config_mod.load(a.config)
    if cfg.bot.mode != "backtest" or cfg.market.broker != "paper":
        cfg.bot.mode, cfg.market.broker, cfg.market.feed = "backtest", "paper", "csv"
    if a.csv:
        cfg.market.csv_path = a.csv
    if not cfg.market.csv_path:
        print("a backtest needs data: set market.csv_path or pass --csv FILE")
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    r = Runner(cfg, printer=(print if a.verbose else (lambda *_: None)))
    s = r.backtest()
    print(json.dumps(s, indent=2, default=str))
    print("\nBACKTEST, not live results. Trades: " + s["output_dir"] + "/trades.csv")
    return 0


def cmd_run(a) -> int:
    from .runner import Runner
    cfg = config_mod.load(a.config)
    if cfg.bot.mode == "backtest":
        print("this config is a backtest — use:  python3 -m tradebot backtest --config ...")
        return 2
    if cfg.bot.mode == "live":
        if not a.i_understand_live:
            print("LIVE mode places real orders. Re-run with --i-understand-live once you have:\n"
                  "  1. backtested the strategy\n  2. paper-traded it on live data\n"
                  "  3. run this exact config against a testnet / paper account")
            return 2
        sandbox = (cfg.market.broker == "ccxt" and cfg.ccxt.sandbox) or \
                  (cfg.market.broker == "ibkr" and cfg.ibkr.port in (7497, 4002))
        print(f"\n*** LIVE ORDERS on {cfg.market.broker} ({'TESTNET / PAPER account' if sandbox else 'REAL MONEY'}) "
              f"— starting in 5 s, Ctrl+C to abort ***\n")
        time.sleep(5)
    _logging(cfg)
    Runner(cfg).run(a.max_bars)
    return 0


def cmd_check(a) -> int:
    from .brokers import make_broker
    from .strategy import load_strategy
    cfg = config_mod.load(a.config)
    s = load_strategy(cfg.strategy.name, cfg.strategy.params)
    print(f"config OK: {cfg.bot.name} | mode {cfg.bot.mode} | {cfg.market.broker}/{cfg.market.feed} "
          f"{cfg.market.symbol} {cfg.market.timeframe}")
    print(f"strategy OK: {type(s).__name__} params={s.params} warmup={s.warmup}")
    if cfg.market.broker != "paper":
        b = make_broker(cfg)
        b.connect()
        print(f"broker OK: equity {b.equity():,.2f} | position {b.position()}")
        b.disconnect()
    return 0


def cmd_status(a) -> int:
    from .state import StateStore
    cfg = config_mod.load(a.config)
    d = StateStore(cfg.state_dir).load()
    print(json.dumps({k: d.get(k) for k in ("last_bar", "position", "governor", "pending")}, indent=2, default=str))
    print("kill switch:", "ON" if (cfg.state_dir / "STOP").exists() else "off")
    return 0


def cmd_stop(a) -> int:
    cfg = config_mod.load(a.config)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "STOP").write_text("created by `tradebot stop`\n")
    print(f"kill switch ON ({cfg.state_dir / 'STOP'}): no new entries; an open position is closed at the next bar.")
    return 0


def cmd_resume(a) -> int:
    cfg = config_mod.load(a.config)
    (cfg.state_dir / "STOP").unlink(missing_ok=True)
    print("kill switch off")
    return 0


def cmd_reset_halt(a) -> int:
    from .state import StateStore
    cfg = config_mod.load(a.config)
    st = StateStore(cfg.state_dir)
    d = st.load()
    g = d.get("governor") or {}
    print("clearing:", g.get("halted_reason") or "(no halt)")
    g["halted_reason"] = ""
    g["peak_equity"] = 0.0               # drawdown is measured from the next peak
    d["governor"] = g
    st.path.write_text(json.dumps(d, indent=2, default=str))
    return 0


def cmd_new_strategy(a) -> int:
    from .strategy import create_strategy_file
    try:
        dst = create_strategy_file(a.name, ROOT / "strategies")
    except ValueError as e:
        print(e)
        return 2
    print(f"created {dst}\nthen set   [strategy] name = '{a.name}'   in your config")
    return 0


def cmd_dashboard(a) -> int:
    import subprocess
    app = ROOT / "dashboard.py"
    # localhost only: the panel can start a live bot, so it must not be reachable from the network
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
                            "--server.address", "localhost",
                            "--browser.gatherUsageStats", "false", "--server.headless", "true" if a.no_browser else "false",
                            "--theme.light.primaryColor", "#2a78d6", "--theme.dark.primaryColor", "#3987e5",
                            "--client.toolbarMode", "minimal"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--config", required=True)
    b.add_argument("--csv", help="override market.csv_path")
    b.add_argument("--verbose", action="store_true", help="print every signal")
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--i-understand-live", action="store_true")
    r.add_argument("--max-bars", type=int)
    for n in ("check", "status", "stop", "resume", "reset-halt"):
        sub.add_parser(n).add_argument("--config", required=True)
    ns = sub.add_parser("new-strategy")
    ns.add_argument("name")
    d = sub.add_parser("dashboard", help="open the control panel in the browser")
    d.add_argument("--port", type=int, default=8501)
    d.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    try:
        return _dispatch(a)
    except KeyboardInterrupt:
        print("\nstopped by user (state is saved; any stop order stays on the venue)")
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        if logging.getLogger().handlers:            # `run` logs the full traceback to bot.log
            logging.getLogger("tradebot").exception("fatal")
        print(f"error: {e}")
        return 1


def _dispatch(a) -> int:
    return {"backtest": cmd_backtest, "run": cmd_run, "check": cmd_check, "status": cmd_status, "stop": cmd_stop,
            "resume": cmd_resume, "reset-halt": cmd_reset_halt, "new-strategy": cmd_new_strategy,
            "dashboard": cmd_dashboard}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
