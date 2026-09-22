#!/usr/bin/env python3
"""Bot CLI.

    python3 -m bot.run backtest              # replay the strategy on history
    python3 -m bot.run paper                 # forward-test on live prices, simulated fills
    python3 -m bot.run paper --synthetic     # offline rehearsal
    python3 -m bot.run status                # where the bot stands
    python3 -m bot.run promote               # human moves PAPER -> LIVE (gated)
    python3 -m bot.run reset-halt            # human clears a hard halt
    python3 -m bot.run live --i-understand-the-risk

The bot cannot promote itself. That is deliberate: a system that decides on its
own that it is ready is a system with no gate at all.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.broker import BinanceSpotBroker, PaperBroker
from bot.config import TradingConfig
from bot.risk import promotion_ready
from bot.state import BotState
from bot.trader import Trader
from quant import data as D


def setup_logging(path: str, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler(sys.stdout)],
    )


def load_bars(cfg: TradingConfig, synthetic: bool):
    if synthetic:
        return D.synthetic_ohlc(bars=20000, seed=17)
    try:
        return D.fetch_binance_klines(cfg.symbol, cfg.interval)
    except Exception as e:
        print(f"[data] live fetch failed ({type(e).__name__}: {e})")
        print("[data] falling back to the simulator - results are NOT about BTC")
        return D.synthetic_ohlc(bars=20000, seed=17)


def cmd_paper(args, cfg: TradingConfig) -> int:
    bars = load_bars(cfg, args.synthetic)
    synth = bool(bars.attrs.get("is_synthetic"))
    st = BotState.load(cfg.state_path)
    broker = PaperBroker(bars, start_equity=st.equity,
                         maker_fee_bps=cfg.maker_fee_bps, taker_fee_bps=cfg.taker_fee_bps,
                         slippage_bps=cfg.slippage_bps, warmup=cfg.vol_pct_window + 250)
    trader = Trader(broker, cfg, st, dry_run=False)
    print(f"PAPER on {bars.attrs.get('source')} | {len(bars):,} bars | start ${st.equity:.2f}")
    if synth:
        print("*** SYNTHETIC BARS - this rehearses the machinery, it does not test the edge ***")

    n = 0
    while broker.advance():
        trader.on_bar(broker.klines(cfg.symbol, cfg.interval, 900))
        n += 1
        if args.limit and n >= args.limit:
            break
    print(json.dumps(trader.status(), indent=2))
    return 0


def cmd_status(args, cfg: TradingConfig) -> int:
    st = BotState.load(cfg.state_path)
    broker = PaperBroker(D.synthetic_ohlc(bars=10, seed=1), start_equity=st.equity)
    print(json.dumps(Trader(broker, cfg, st).status(), indent=2))
    if st.trades:
        rs = [t["r_multiple"] for t in st.trades]
        wins = [r for r in rs if r > 0]
        print(f"\ntrades {len(rs)} | win rate {len(wins)/len(rs)*100:.1f}% | "
              f"expectancy {sum(rs)/len(rs):+.3f}R | total {sum(rs):+.2f}R")
    return 0


def cmd_promote(args, cfg: TradingConfig) -> int:
    st = BotState.load(cfg.state_path)
    ok, why = promotion_ready(cfg, st)
    if not ok:
        print(f"REFUSED: {why}")
        return 1
    if st.halted:
        print(f"REFUSED: bot is halted ({st.halt_reason}). Clear it first.")
        return 1
    st.phase = "LIVE"
    st.live_trades_done = 0            # probation restarts at half risk
    st.save(cfg.state_path)
    print(f"Promoted to LIVE. {why}")
    print(f"First {cfg.probation_trades} trades run at {cfg.probation_risk_pct*100:g}% risk.")
    return 0


def cmd_reset_halt(args, cfg: TradingConfig) -> int:
    st = BotState.load(cfg.state_path)
    if not st.halted:
        print("not halted")
        return 0
    print(f"clearing halt: {st.halt_reason}")
    st.halted = False
    st.halt_reason = ""
    st.consecutive_losses = 0
    st.api_error_count = 0
    st.peak_equity = st.equity          # reset the high-water mark with the human's consent
    st.save(cfg.state_path)
    print("halt cleared; drawdown baseline reset to current equity")
    return 0


def cmd_live(args, cfg: TradingConfig) -> int:
    import os
    st = BotState.load(cfg.state_path)
    if st.phase != "LIVE":
        print("REFUSED: bot is in PAPER. Run `promote` first.")
        return 1
    if st.halted:
        print(f"REFUSED: halted ({st.halt_reason})")
        return 1
    key, sec = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
    if not key or not sec:
        print("REFUSED: set BINANCE_API_KEY and BINANCE_API_SECRET")
        return 1
    print("!! The live adapter has NOT been verified against a real exchange in the")
    print("!! environment where it was written. Run it on testnet.binance.vision first.")
    broker = BinanceSpotBroker(key, sec, base=args.base, unlocked=True)
    trader = Trader(broker, cfg, st, dry_run=args.dry_run)
    print(f"LIVE on {cfg.symbol} {cfg.interval} | equity ${st.equity:.2f} | dry_run={args.dry_run}")
    while True:
        try:
            trader.reconcile()
            df = broker.klines(cfg.symbol, cfg.interval, 900)
            trader.on_bar(df)
            if st.halted:
                print(f"halted: {st.halt_reason}")
                return 1
        except KeyboardInterrupt:
            print("\nstopped by user; state saved")
            return 0
        except Exception as e:
            logging.exception("loop error: %s", e)
            st.api_error_count += 1
            st.save(cfg.state_path)
        time.sleep(cfg.poll_seconds)


def cmd_backtest(args, cfg: TradingConfig) -> int:
    import pandas as pd
    from quant.engine import Costs, ExecConfig, run_backtest
    from quant.metrics import summarize
    from quant.strategies import S1Params, s1_trend_breakout

    bars = load_bars(cfg, args.synthetic)
    p = S1Params(donchian=cfg.donchian, ema_trend=cfg.ema_trend, adx_min=cfg.adx_min,
                 stop_atr=cfg.stop_atr, trail_atr=cfg.trail_atr,
                 vol_pct_min=cfg.vol_pct_min, max_bars=cfg.max_bars_in_trade,
                 allow_short=cfg.allow_short)
    orders, atr_s, _ = s1_trend_breakout(bars, p)
    res = run_backtest(bars, orders, atr_s,
                       Costs(fee_bps=cfg.maker_fee_bps, slippage_bps=cfg.slippage_bps),
                       ExecConfig(initial_equity=100.0, risk_pct=cfg.risk_pct,
                                  max_leverage=cfg.max_leverage,
                                  min_notional=cfg.min_notional, allow_short=cfg.allow_short))
    s = summarize(res, cfg.bars_per_year, "bot S1")
    print(f"source: {bars.attrs.get('source')}")
    if bars.attrs.get("is_synthetic"):
        print("*** SYNTHETIC - validates the pipeline, says nothing about BTC ***")
    print(pd.Series(s).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="$100 trend bot")
    ap.add_argument("command", choices=["paper", "live", "status", "promote",
                                        "reset-halt", "backtest"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-understand-the-risk", action="store_true")
    ap.add_argument("--base", default="https://api.binance.com")
    ap.add_argument("--state", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = TradingConfig()
    if args.state:
        cfg.state_path = args.state
    setup_logging(cfg.log_path, args.verbose)

    if args.command == "live" and not args.i_understand_the_risk:
        print("REFUSED: live trading needs --i-understand-the-risk")
        return 1

    return {"paper": cmd_paper, "status": cmd_status, "promote": cmd_promote,
            "reset-halt": cmd_reset_halt, "live": cmd_live,
            "backtest": cmd_backtest}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
