"""The run loop — identical for backtest, paper and live.

Per closed bar:
  1. broker.on_bar        fills (paper) or venue-side stop/target exits (live)
  2. governor             day roll-over, drawdown halt
  3. strategy.on_bar      the only strategy-specific step
  4. exits                session flatten, time limit, Exit(...), kill switch
  5. entries              governor + session window + sizing, then a bracket order
  6. state + journal      saved after every bar (live/paper)
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .brokers import make_broker
from .brokers.base import BrokerError
from .config import Config
from .feeds import COLS, CSVFeed, to_bars
from .models import SIDE_NAME, Bar, Exit, Position, Signal
from .risk import Governor, GovernorState, size_position
from .session import Session
from .state import Journal, StateStore
from .strategy import Context, load_strategy

log = logging.getLogger("tradebot")


class Runner:
    def __init__(self, cfg: Config, strategy=None, broker=None, feed=None, state_dir: Path | None = None,
                 printer=print):
        self.cfg = cfg
        self.strategy = strategy or load_strategy(cfg.strategy.name, cfg.strategy.params)
        self.broker = broker or make_broker(cfg)
        self.feed = feed
        self.dir = Path(state_dir or cfg.state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = Session(cfg.session, cfg.tf_minutes)
        self.store = StateStore(self.dir)
        self.journal = Journal(self.dir)
        self.kill_file = self.dir / "STOP"
        self.gov = Governor(cfg.account, GovernorState(), self.kill_file)
        self.position: Position | None = None
        self.pending: dict | None = None
        self.last_bar: str | None = None
        self.trades: list[dict] = []
        self.equity_curve: list[tuple] = []
        self.counts = {"total": 0, "sent": 0, "blocked": 0, "rejected_size": 0, "broker_errors": 0}
        self.halted_at: str | None = None
        self.printer = printer
        self.persist = True

    # --- ids, state ---------------------------------------------------------------------
    def _cid(self) -> str:
        return "tb" + uuid.uuid4().hex[:12]

    def restore(self) -> None:
        d = self.store.load()
        self.position = StateStore.position_from(d.get("position"))
        self.gov.state = StateStore.governor_from(d.get("governor"))
        self.broker.restore(d.get("broker", {}))
        self.last_bar = d.get("last_bar")
        self.pending = d.get("pending")

    def save(self) -> None:
        if self.persist:
            self.store.save(self.position, self.gov.state, self.broker.state(), self.last_bar,
                            {"pending": self.pending, "strategy": self.cfg.strategy.name})

    def reconcile(self) -> list[str]:
        """The broker is the source of truth; disagreements stop new trading."""
        bp = self.broker.position()
        msgs = []
        if bp is None and self.position is not None:
            msgs.append(f"saved position ({SIDE_NAME[self.position.side]} {self.position.qty}) is no longer open at "
                        "the broker: recorded as closed while the bot was offline")
            self.journal.trade({"opened_at": self.position.opened_at, "side": SIDE_NAME[self.position.side],
                                "qty": self.position.qty, "entry": self.position.entry_price,
                                "exit_reason": "closed while bot offline (see broker statement)",
                                "client_id": self.position.client_id})
            self.position = None
        elif bp is not None and self.position is None:
            self.gov.state.halted_reason = (f"broker shows a {SIDE_NAME[bp.side]} {bp.qty} position this bot did not "
                                            "open — handle it manually, then run reset-halt")
            msgs.append(self.gov.state.halted_reason)
        elif bp is not None and (bp.side != self.position.side or abs(bp.qty - self.position.qty) > 1e-9):
            self.gov.state.halted_reason = (f"position mismatch: broker {SIDE_NAME[bp.side]} {bp.qty}, "
                                            f"bot {SIDE_NAME[self.position.side]} {self.position.qty}")
            msgs.append(self.gov.state.halted_reason)
        return msgs

    # --- events ---------------------------------------------------------------------------
    def handle(self, events) -> None:
        for e in events:
            if e.kind == "opened":
                p = self.pending or {}
                self.position = Position(e.side, e.qty, e.price, e.stop_price, e.target_price, e.ts,
                                         p.get("risk", 0.0), p.get("reason", ""), e.client_id, 0, p.get("max_bars"))
                self.pending = None
                log.info("OPENED %s %g @ %.4f stop %.4f target %s", SIDE_NAME[e.side], e.qty, e.price,
                         e.stop_price, f"{e.target_price:.4f}" if e.target_price else "-")
            elif e.kind == "closed":
                pos = self.position
                risk = pos.risk_amount if pos else 0.0
                row = {"opened_at": pos.opened_at if pos else "", "closed_at": e.ts, "side": SIDE_NAME[e.side],
                       "qty": e.qty, "entry": pos.entry_price if pos else "", "exit": e.price if e.price is not None else "",
                       "stop": pos.stop_price if pos else "", "target": pos.target_price if pos and pos.target_price else "",
                       "risk": round(risk, 2), "pnl": round(e.pnl or 0.0, 2),
                       "r": round((e.pnl or 0.0) / risk, 3) if risk else "", "bars": pos.bars_held if pos else "",
                       "entry_reason": pos.reason if pos else "", "exit_reason": e.reason, "client_id": e.client_id}
                self.journal.trade(row)
                self.trades.append(row)
                self.gov.record_trade(e.pnl or 0.0)
                self.position = None
                log.info("CLOSED %s %g @ %s (%s) pnl %.2f", SIDE_NAME[e.side], e.qty, e.price, e.reason, e.pnl or 0.0)

    # --- one bar --------------------------------------------------------------------------
    def step(self, bar: Bar, hist: pd.DataFrame) -> None:
        self.handle(self.broker.on_bar(bar))
        self.pending = None
        eq = self.broker.equity()
        self.gov.roll_day(self.session.trading_day(bar.ts), eq)
        if self.gov.state.halted_reason and self.halted_at is None:
            self.halted_at = str(bar.ts)
            log.error("HALTED at %s: %s", bar.ts, self.gov.state.halted_reason)
        self.last_bar = str(bar.ts)
        if self.position is not None:
            self.position.bars_held += 1
        if len(hist) >= self.strategy.warmup:
            ctx = Context(hist, self.position, eq, self.cfg.market.symbol, self.cfg.market.timeframe)
            out = self.strategy.on_bar(ctx)
            if self.position is not None:
                reason = None
                if self.session.must_flatten(bar.ts):
                    reason = "session flatten"
                elif self.position.max_bars and self.position.bars_held >= self.position.max_bars:
                    reason = f"time exit ({self.position.max_bars} bars)"
                elif isinstance(out, Exit):
                    reason = out.reason
                elif self.gov.killed():
                    reason = "kill switch"
                if reason:
                    self.handle(self.broker.close_position(reason, self._cid(), bar.close, str(bar.ts)))
            elif isinstance(out, Signal):
                self.on_signal(out, bar, eq, ctx)
        self.equity_curve.append((bar.ts, self.broker.equity(), self.position is not None))
        self.save()

    def on_signal(self, sig: Signal, bar: Bar, eq: float, ctx: Context) -> None:
        self.counts["total"] += 1
        inst, acct = self.cfg.instrument, self.cfg.account
        ok, why = self.gov.allows_entry()
        if ok:
            ok, why = self.session.entry_allowed(bar.ts)
        if ok and sig.side < 0 and not inst.allow_short:
            ok, why = False, "shorting is disabled for this instrument"
        sz = size_position(eq, sig.stop, bar.close, inst, acct)
        status = "sent"
        if not ok:
            status = "blocked"
        elif sz.qty <= 0:
            status, why = "rejected_size", sz.note
        self.counts[status] += 1
        self.journal.signal({"ts": str(bar.ts), "side": SIDE_NAME[sig.side], "price": bar.close,
                             "stop_dist": round(sig.stop, 6), "target_dist": round(sig.target, 6) if sig.target else "",
                             "qty": sz.qty if status == "sent" else 0, "status": status, "detail": why or sig.reason})
        self.printer(self.format_signal(sig, bar, sz, status, why, ctx))
        if status != "sent":
            return
        cid = self._cid()
        self.pending = {"risk": sz.risk_amount, "reason": sig.reason, "max_bars": sig.max_bars, "cid": cid}
        try:
            self.handle(self.broker.open_bracket(sig.side, sz.qty, sig.stop, sig.target, cid, bar.close, str(bar.ts)))
        except BrokerError as e:
            self.counts["broker_errors"] += 1
            self.pending = None
            log.error("broker refused the entry: %s", e)
            self.journal.signal({"ts": str(bar.ts), "side": SIDE_NAME[sig.side], "status": "broker_error", "detail": str(e)})

    def format_signal(self, sig: Signal, bar: Bar, sz, status: str, why: str, ctx: Context) -> str:
        inst, side = self.cfg.instrument, SIDE_NAME[sig.side]
        entry = bar.close
        stop = entry - sig.side * sig.stop
        tgt = entry + sig.side * sig.target if sig.target else None
        rr = sig.target / sig.stop if sig.target else None
        flat = self.cfg.session.flatten_at
        if status == "sent":
            signal = side
            size = f"{sz.qty:g} units, risk ${sz.risk_amount:,.2f} of ${sz.budget:,.2f} budget"
        else:
            signal = f"NO TRADE ({side} signal {status.replace('_', ' ')}: {why})"
            size = f"0 — {why}" if status == "rejected_size" else "0"
        if sig.probability is not None and rr:
            p = sig.probability
            ev = f"{p * rr - (1 - p):+.3f} R before costs (from the strategy's own probability — not verified here)"
            prob = f"{100 * p:.1f}% (supplied by the strategy; only meaningful if it was calibrated out of sample)"
        else:
            ev = "unavailable — the strategy supplies no calibrated probability"
            prob = "unavailable — no calibrated, validated probability model"
        lines = [
            ("INSTRUMENT", self.cfg.market.symbol),
            ("SIGNAL", signal),
            ("TIMEFRAME", f"{self.cfg.market.timeframe} (bar opened {bar.ts}, now closed)"),
            ("ENTRY PRICE", f"~{entry:.4f} — market order at the next bar's open"),
            ("STOP-LOSS", f"{stop:.4f} ({sig.stop:.4f} from entry)"),
            ("TAKE-PROFIT", f"{tgt:.4f} ({sig.target:.4f} from entry)" if tgt else "none"),
            ("RISK-TO-REWARD", f"1:{rr:.2f}" if rr else "n/a"),
            ("ESTIMATED WIN PROBABILITY", prob),
            ("EXPECTED VALUE", ev),
            ("POSITION SIZE", size),
            ("MARKET REGIME", "; ".join(ctx.notes) if ctx.notes else "n/a (strategy reported none)"),
            ("SIGNAL CONFIDENCE", "unavailable — not validated"),
            ("REASON FOR SIGNAL", sig.reason or "(strategy gave no reason)"),
            ("INVALIDATION CONDITION", f"stop {stop:.4f} trades"
             + (f"; time exit after {sig.max_bars} bars" if sig.max_bars else "")
             + (f"; session flatten {flat}" if flat else "")),
        ]
        return "\n".join(f"{k}: {v}" for k, v in lines)

    # --- modes ----------------------------------------------------------------------------
    def backtest(self, csv_path: str | None = None) -> dict:
        m = self.cfg.market
        df = CSVFeed(csv_path or m.csv_path, self.cfg.tf_minutes, m.csv_timezone, m.start, m.end).df
        out = self.dir / "backtest"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        self.journal = Journal(out)
        self.persist = False
        L = self.cfg.bot.history_bars
        for i, b in enumerate(to_bars(df)):
            self.step(b, df.iloc[max(0, i - L + 1): i + 1])
        if self.position is not None and hasattr(self.broker, "pos") and self.broker.pos is not None:
            last = df.iloc[-1]
            self.handle([self.broker._close(float(last["close"]), "end of data", df.index[-1])])
        summary = summarize(self.trades, self.equity_curve, self.cfg.account.starting_equity, self.counts)
        summary["halted"] = ({"at": self.halted_at, "reason": self.gov.state.halted_reason}
                             if self.gov.state.halted_reason else None)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        pd.DataFrame(self.equity_curve, columns=["ts", "equity", "in_position"]).to_csv(out / "equity.csv", index=False)
        summary["output_dir"] = str(out)
        return summary

    def run(self, max_bars: int | None = None) -> None:
        from .feeds import CCXTFeed, IBKRFeed
        self.broker.connect()
        self.restore()
        for msg in self.reconcile():
            log.warning("RECONCILE: %s", msg)
        self.save()
        cfg, m = self.cfg, self.cfg.market
        feed = self.feed
        if feed is None:
            if m.feed == "ccxt":
                ex = getattr(self.broker, "ex", None)
                feed = CCXTFeed(m.symbol, m.timeframe, cfg.tf_minutes, cfg.ccxt.exchange, ex, cfg.bot.poll_seconds)
            elif m.feed == "ibkr":
                from .brokers.ibkr import IBKRBroker
                src = self.broker if isinstance(self.broker, IBKRBroker) else IBKRBroker(cfg)
                if src is not self.broker:
                    src.connect()
                feed = IBKRFeed(src.ib, src.contract, m.timeframe, cfg.tf_minutes, cfg.ibkr.use_rth, cfg.bot.poll_seconds)
            else:
                feed = ReplayFeed(CSVFeed(m.csv_path, cfg.tf_minutes, m.csv_timezone, m.start, m.end).df)
        hist = feed.history(cfg.bot.history_bars)
        log.info("START %s | mode=%s broker=%s feed=%s %s %s | equity %.2f | strategy=%s | risk %.2f%%",
                 cfg.bot.name, cfg.bot.mode, m.broker, m.feed, m.symbol, m.timeframe, self.broker.equity(),
                 cfg.strategy.name, cfg.account.risk_pct)
        n = 0
        for bar in feed.stream():
            if self.last_bar and pd.Timestamp(bar.ts) <= pd.Timestamp(self.last_bar):
                continue
            row = pd.DataFrame([[bar.open, bar.high, bar.low, bar.close, bar.volume]], columns=COLS,
                               index=pd.DatetimeIndex([bar.ts], name="ts"))
            hist = pd.concat([hist, row]).tail(cfg.bot.history_bars)
            self.step(bar, hist)
            n += 1
            if self.gov.killed() and self.position is None:
                log.warning("kill switch present and flat: stopping")
                break
            if max_bars and n >= max_bars:
                break
        self.broker.disconnect()


class ReplayFeed:
    """Plays a CSV through the live loop (offline paper-mode rehearsal)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cut = 0

    def history(self, n: int) -> pd.DataFrame:
        self.cut = min(n, len(self.df) - 1)
        return self.df.iloc[: self.cut]

    def stream(self):
        yield from to_bars(self.df.iloc[self.cut:])


def summarize(trades: list[dict], curve: list[tuple], eq0: float, counts: dict) -> dict:
    eq = np.array([c[1] for c in curve]) if curve else np.array([eq0])
    peak = np.maximum.accumulate(np.r_[eq0, eq])
    dd = float(np.max(1 - np.r_[eq0, eq] / peak)) if len(eq) else 0.0
    pnl = np.array([t["pnl"] for t in trades], float)
    r = np.array([t["r"] for t in trades if t["r"] != ""], float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    out = {
        "trades": len(trades),
        "net_pnl": round(float(pnl.sum()), 2),
        "return_pct": round(100 * float(eq[-1] / eq0 - 1), 2),
        "win_rate_pct": round(100 * len(wins) / len(pnl), 1) if len(pnl) else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if len(losses) and losses.sum() < 0 else None,
        "expectancy_r": round(float(r.mean()), 4) if len(r) else None,
        "max_drawdown_pct": round(100 * dd, 2),
        "time_in_market_pct": round(100 * float(np.mean([c[2] for c in curve])), 1) if curve else 0.0,
        "longs": sum(1 for t in trades if t["side"] == "LONG"),
        "shorts": sum(1 for t in trades if t["side"] == "SHORT"),
        "exit_reasons": pd.Series([t["exit_reason"] for t in trades]).value_counts().to_dict() if trades else {},
        **{f"signals_{k}": v for k, v in counts.items()},
    }
    if len(r) > 1:
        se = r.std(ddof=1) / math.sqrt(len(r))
        out["expectancy_r_95ci"] = [round(float(r.mean() - 1.96 * se), 4), round(float(r.mean() + 1.96 * se), 4)]
    return out
