"""The bot's control loop.

Order of operations each cycle, and the reasoning behind it:

 1. RECONCILE against the venue. Local state is a cache; the exchange is truth.
    A bot that restarts and trusts its own file will re-enter a position it
    already holds.
 2. Roll the UTC day, so the daily loss limit means something.
 3. Check hard halts BEFORE anything else.
 4. Resolve a working entry order (filled -> protect it; stale -> cancel).
 5. Manage an open position (stop hit? time out? trail the stop?).
 6. Only then look for a new entry.
 7. Persist state after every mutation.

The invariant that matters most: AN OPEN POSITION ALWAYS HAS A LIVE STOP ORDER
ON THE EXCHANGE. If the process dies, the stop still protects the account. The
trailing update therefore places the new stop before it trusts the cancel, and
screams if it ever ends up naked.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .broker import Broker, BrokerError, Order
from .config import TradingConfig
from .risk import can_open, check_halts, effective_risk_pct, promotion_ready
from .state import BotState, PendingEntry, Position, utcnow
from .strategy import Signal, evaluate, trail_stop

log = logging.getLogger("bot")


def _coid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class Trader:
    def __init__(self, broker: Broker, cfg: TradingConfig, state: BotState,
                 dry_run: bool = False):
        self.b, self.cfg, self.st, self.dry = broker, cfg, state, dry_run
        self.bar_index = 0

    # ------------------------------------------------------------------
    def halt(self, reason: str) -> None:
        if not self.st.halted:
            self.st.halted = True
            self.st.halt_reason = reason
            log.critical("HALTED: %s", reason)
            self._flatten_protectively()
        self.save()

    def _flatten_protectively(self) -> None:
        """On a hard halt, cancel working entries. The STOP stays in place —
        removing a stop on a live position is the opposite of safe."""
        try:
            if self.st.pending_entry:
                self.b.cancel(self.cfg.symbol, self.st.pending_entry.client_order_id)
                self.st.pending_entry = None
        except BrokerError as e:
            log.error("could not cancel pending entry during halt: %s", e)

    def save(self) -> None:
        self.st.save(self.cfg.state_path)

    # ------------------------------------------------------------------
    def reconcile(self) -> None:
        """Make local state agree with the venue. Venue wins, always."""
        sym = self.cfg.symbol
        try:
            base = self.b.base_qty(sym)
            open_orders = {o.client_order_id: o for o in self.b.open_orders(sym)}
            self.st.api_error_count = 0
        except BrokerError as e:
            self.st.api_error_count += 1
            log.error("reconcile failed (%d/%d): %s", self.st.api_error_count,
                      self.cfg.max_api_errors, e)
            return

        # A pending entry that the venue no longer knows about is resolved.
        pe = self.st.pending_entry
        if pe and pe.client_order_id not in open_orders:
            od = self.b.get_order(sym, pe.client_order_id)
            if od and od.status == "filled":
                log.info("entry filled on reconcile: %s", pe.client_order_id)
                self._open_position_from_fill(od, pe)
            else:
                log.info("pending entry gone (%s); clearing",
                         od.status if od else "unknown")
                self.st.pending_entry = None

        pos = self.st.position
        if pos:
            if base < pos.qty * 0.5:
                # The base asset is gone: the stop filled while we were away.
                log.warning("position not on venue (base=%.8f, expected %.8f) - "
                            "treating as stopped out", base, pos.qty)
                px = pos.stop_price
                od = self.b.get_order(sym, pos.stop_order_id) if pos.stop_order_id else None
                if od and od.status == "filled" and od.avg_price > 0:
                    px = od.avg_price
                self._close_position(px, "stop_offline")
            elif pos.stop_order_id and pos.stop_order_id not in open_orders:
                od = self.b.get_order(sym, pos.stop_order_id)
                if not od or od.status != "filled":
                    # We hold the asset with no stop. Fix immediately.
                    log.critical("OPEN POSITION WITH NO STOP ORDER - replacing now")
                    self._place_stop(pos, pos.stop_price)
        elif base * self._price() > self.cfg.min_notional:
            log.warning("venue holds %.8f base with no local position. Not adopting it "
                        "automatically - halting for human review.", base)
            self.halt("unrecognised base-asset balance on venue")
        self.save()

    def _price(self) -> float:
        try:
            return float(self.b.klines(self.cfg.symbol, self.cfg.interval, 2)["close"].iloc[-1])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _place_stop(self, pos: Position, stop_price: float) -> Optional[str]:
        stop_price = round(stop_price / self.cfg.price_tick) * self.cfg.price_tick
        coid = _coid("stp")
        if self.dry:
            log.info("[dry] would place stop %s qty=%.8f @ %.2f", coid, pos.qty, stop_price)
            pos.stop_order_id = coid
            pos.stop_price = stop_price
            return coid
        try:
            side = "sell" if pos.side == "long" else "buy"
            self.b.place_stop_loss(self.cfg.symbol, side, pos.qty, stop_price, coid)
            pos.stop_order_id = coid
            pos.stop_price = stop_price
            log.info("stop placed %s @ %.2f", coid, stop_price)
            return coid
        except BrokerError as e:
            log.critical("FAILED TO PLACE STOP: %s - position is unprotected", e)
            self.st.api_error_count += 1
            return None

    def _move_stop(self, pos: Position, new_stop: float) -> None:
        """Replace the resting stop. Place-then-cancel is not possible on spot
        (the balance is committed), so cancel first and re-place immediately,
        then verify. Any failure here is a critical event, not a warning."""
        old = pos.stop_order_id
        if not self.dry and old:
            try:
                self.b.cancel(self.cfg.symbol, old)
            except BrokerError as e:
                log.error("could not cancel old stop %s: %s - leaving it in place", old, e)
                return
        if self._place_stop(pos, new_stop) is None:
            self.halt("position left without a stop after a trail update")

    # ------------------------------------------------------------------
    def _open_position_from_fill(self, od: Order, pe: PendingEntry) -> None:
        entry = od.avg_price or pe.limit_price
        stop = entry - pe.stop_distance if pe.side == "long" else entry + pe.stop_distance
        pos = Position(
            side=pe.side, qty=od.filled_qty or pe.qty, entry_price=entry,
            stop_price=stop, initial_stop=stop, entry_time=utcnow(),
            entry_bar_index=self.bar_index, risk_amount=(od.filled_qty or pe.qty) * pe.stop_distance,
            client_order_id=pe.client_order_id,
        )
        self.st.position = pos
        self.st.pending_entry = None
        slip_bps = abs(entry - pe.limit_price) / pe.limit_price * 10_000 if pe.limit_price else 0.0
        pos_slip = slip_bps
        log.info("OPENED %s qty=%.8f @ %.2f stop=%.2f risk=$%.2f (slip %.1fbps)",
                 pos.side, pos.qty, entry, stop, pos.risk_amount, pos_slip)
        self._place_stop(pos, stop)
        self._last_slip_bps = pos_slip
        self.save()

    def _close_position(self, exit_price: float, reason: str) -> None:
        pos = self.st.position
        if not pos:
            return
        gross = ((exit_price - pos.entry_price) if pos.side == "long"
                 else (pos.entry_price - exit_price)) * pos.qty
        fee_rate = self.cfg.taker_fee_bps / 10_000.0
        costs = (pos.entry_price * pos.qty * self.cfg.maker_fee_bps / 10_000.0
                 + exit_price * pos.qty * fee_rate)
        pnl = gross - costs
        r = pnl / pos.risk_amount if pos.risk_amount > 0 else 0.0
        trade = {
            "phase": self.st.phase, "side": pos.side, "qty": pos.qty,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "entry_time": pos.entry_time, "exit_time": utcnow(),
            "stop_price": pos.stop_price, "initial_stop": pos.initial_stop,
            "risk_amount": pos.risk_amount, "pnl": pnl, "costs": costs,
            "r_multiple": r, "reason": reason,
            "slippage_bps": getattr(self, "_last_slip_bps", 0.0),
            "bars_held": self.bar_index - pos.entry_bar_index,
        }
        self.st.record_trade(trade)
        self.st.position = None
        log.info("CLOSED %s @ %.2f (%s) pnl=$%.4f = %+.2fR", pos.side, exit_price, reason, pnl, r)

        try:
            self.st.mark_equity(self.b.equity())
        except BrokerError:
            self.st.mark_equity(self.st.equity + pnl)

        halt = check_halts(self.cfg, self.st)
        if halt:
            self.halt(halt)
        self.save()

    # ------------------------------------------------------------------
    def on_bar(self, df: pd.DataFrame) -> None:
        """Process one CLOSED bar. This is the whole bot."""
        if df.empty:
            return
        bar_time = df.index[-1]
        if str(bar_time) == self.st.last_processed_bar:
            return                                    # idempotent per bar
        self.bar_index += 1
        self.st.roll_day(bar_time.strftime("%Y-%m-%d"))

        halt = check_halts(self.cfg, self.st)
        if halt and not self.st.halted:
            self.halt(halt)
        if self.st.halted:
            self.st.last_processed_bar = str(bar_time)
            self.save()
            return

        close = float(df["close"].iloc[-1])
        from quant import indicators as ind
        atr_v = float(ind.atr(df, self.cfg.atr_period).iloc[-1])

        # --- 1. resolve a working entry ---------------------------------
        pe = self.st.pending_entry
        if pe:
            od = self.b.get_order(self.cfg.symbol, pe.client_order_id)
            if od and od.status == "filled":
                self._open_position_from_fill(od, pe)
            elif self.bar_index - pe.placed_bar_index >= self.cfg.entry_timeout_bars:
                log.info("entry %s unfilled after %d bars - cancelling",
                         pe.client_order_id, self.cfg.entry_timeout_bars)
                try:
                    self.b.cancel(self.cfg.symbol, pe.client_order_id)
                except BrokerError as e:
                    log.error("cancel failed: %s", e)
                self.st.pending_entry = None

        # --- 2. manage an open position ---------------------------------
        pos = self.st.position
        if pos:
            od = self.b.get_order(self.cfg.symbol, pos.stop_order_id) if pos.stop_order_id else None
            if od and od.status == "filled":
                self._close_position(od.avg_price or pos.stop_price, "stop")
            else:
                held = self.bar_index - pos.entry_bar_index
                if held >= self.cfg.max_bars_in_trade:
                    log.info("time exit after %d bars", held)
                    self._exit_at_market(close, "time_exit")
                else:
                    new_stop = trail_stop(self.cfg, pos.side, pos.stop_price, close, atr_v)
                    # Only move it if the improvement is worth the exposure window.
                    if abs(new_stop - pos.stop_price) > close * 0.001:
                        log.info("trailing stop %.2f -> %.2f", pos.stop_price, new_stop)
                        self._move_stop(pos, new_stop)

        # --- 3. look for a new entry ------------------------------------
        if self.st.position is None and self.st.pending_entry is None:
            sig = evaluate(df, self.cfg)
            if sig.action == "enter":
                self._try_enter(sig, close)
            else:
                log.debug("no signal: %s", sig.reason)

        self.st.last_processed_bar = str(bar_time)
        try:
            self.st.mark_equity(self.b.equity())
        except BrokerError:
            pass
        self.save()

    def _exit_at_market(self, price: float, reason: str) -> None:
        pos = self.st.position
        if not pos:
            return
        if pos.stop_order_id and not self.dry:
            try:
                self.b.cancel(self.cfg.symbol, pos.stop_order_id)
            except BrokerError as e:
                log.error("could not cancel stop before market exit: %s", e)
                return                       # never exit while a stop may also fire
        if self.dry:
            self._close_position(price, reason)
            return
        try:
            side = "sell" if pos.side == "long" else "buy"
            od = self.b.place_market(self.cfg.symbol, side, pos.qty, _coid("exit"))
            self._close_position(od.avg_price or price, reason)
        except BrokerError as e:
            log.critical("MARKET EXIT FAILED: %s - re-placing stop", e)
            self._place_stop(pos, pos.stop_price)

    def _try_enter(self, sig: Signal, close: float) -> None:
        tick = self.cfg.price_tick
        # Post-only: sit just inside the book so the order is a maker.
        limit = close - self.cfg.entry_offset_ticks * tick if sig.side == "long" \
            else close + self.cfg.entry_offset_ticks * tick
        limit = round(limit / tick) * tick
        stop = limit - sig.stop_distance if sig.side == "long" else limit + sig.stop_distance

        try:
            equity = self.b.equity()
        except BrokerError as e:
            log.error("cannot read equity: %s", e)
            self.st.api_error_count += 1
            return

        d = can_open(self.cfg, self.st, limit, stop, equity)
        if not d.allowed:
            log.info("entry rejected: %s", d.reason)
            return

        coid = _coid("ent")
        log.info("ENTRY %s qty=%.8f limit=%.2f stop=%.2f notional=$%.2f risk=$%.2f (%s)",
                 sig.side, d.qty, limit, stop, d.notional, d.risk_amount, sig.reason)
        if self.dry:
            log.info("[dry] no order sent")
            return
        try:
            side = "buy" if sig.side == "long" else "sell"
            if self.cfg.maker_only:
                self.b.place_limit_maker(self.cfg.symbol, side, d.qty, limit, coid)
            else:
                self.b.place_market(self.cfg.symbol, side, d.qty, coid)
            self.st.pending_entry = PendingEntry(coid, sig.side, limit, d.qty,
                                                 sig.stop_distance, self.bar_index, utcnow())
            self.save()
        except BrokerError as e:
            log.error("entry order failed: %s", e)
            self.st.api_error_count += 1

    # ------------------------------------------------------------------
    def status(self) -> dict:
        ok, why = promotion_ready(self.cfg, self.st)
        return {
            "phase": self.st.phase, "halted": self.st.halted,
            "halt_reason": self.st.halt_reason, "equity": round(self.st.equity, 4),
            "peak": round(self.st.peak_equity, 4),
            "drawdown_pct": round(self.st.drawdown * 100, 2),
            "trades": len(self.st.trades),
            "consecutive_losses": self.st.consecutive_losses,
            "day_R": round(self.st.day_realised_R, 3),
            "effective_risk_pct": round(effective_risk_pct(self.cfg, self.st) * 100, 3),
            "in_position": self.st.position is not None,
            "promotion_ready": ok, "promotion_note": why,
        }
