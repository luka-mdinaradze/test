"""Simulated broker with conservative fills (backtest and paper trading).

Rules (the same as the research engine in nqlab/engine.py):
  1. A market order sent at a bar's close fills at the NEXT bar's open, plus
     adverse slippage. Never at the close it was decided on.
  2. If one bar touches both the stop and the target, the STOP is booked.
  3. A bar that opens through the stop fills at that open (a gap loses > 1R).
  4. A target is a resting limit: it fills only if price trades through it by
     `target_through_ticks`; it fills at the target price.
  5. Commission per unit and fee bps are charged on both sides.
Resolution is the bar itself: on 15m bars, the order inside a bar is unknown,
which is why rule 2 is pessimistic.
"""
from __future__ import annotations

from ..config import InstrumentSection
from ..models import LONG, Bar, BrokerPosition, Event
from ..risk import adverse
from .base import Broker, BrokerError


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, inst: InstrumentSection, starting_equity: float):
        self.inst = inst
        self.cash = float(starting_equity)
        self.pos: dict | None = None
        self.pending: dict | None = None

    # --- interface -------------------------------------------------------------------
    def equity(self) -> float:
        return self.cash

    def position(self) -> BrokerPosition | None:
        if self.pos is None:
            return None
        return BrokerPosition(self.pos["side"], self.pos["qty"], self.pos["entry"])

    def open_bracket(self, side, qty, stop_dist, target_dist, client_id, ref_price, ts) -> list[Event]:
        if self.pos is not None or self.pending is not None:
            raise BrokerError("paper broker: a position or order is already working")
        if side != LONG and not self.inst.allow_short:
            raise BrokerError("shorting disabled for this instrument")
        self.pending = {"kind": "open", "side": side, "qty": qty, "stop": stop_dist,
                        "target": target_dist, "cid": client_id}
        return []

    def close_position(self, reason, client_id, ref_price, ts) -> list[Event]:
        if self.pos is not None:
            self.pending = {"kind": "close", "reason": reason, "cid": client_id}
        elif self.pending and self.pending["kind"] == "open":
            self.pending = None
        return []

    def on_bar(self, bar: Bar) -> list[Event]:
        ev: list[Event] = []
        p, self.pending = self.pending, None
        entered = False
        if p and p["kind"] == "open" and self.pos is None:
            side = p["side"]
            fill = bar.open + side * adverse(self.inst, bar.open)
            self.pos = {"side": side, "qty": p["qty"], "entry": fill,
                        "stop": fill - side * p["stop"],
                        "target": fill + side * p["target"] if p["target"] else None,
                        "cid": p["cid"], "opened": str(bar.ts)}
            ev.append(Event("opened", side, p["qty"], fill, str(bar.ts), stop_price=self.pos["stop"],
                            target_price=self.pos["target"], client_id=p["cid"]))
            entered = True
        elif p and p["kind"] == "close" and self.pos is not None:
            side = self.pos["side"]
            ev.append(self._close(bar.open - side * adverse(self.inst, bar.open), p["reason"], bar.ts))
        if self.pos is not None:
            e = self._exits(bar, entered)
            if e:
                ev.append(e)
        return ev

    # --- internals -------------------------------------------------------------------
    def _exits(self, bar: Bar, entered: bool) -> Event | None:
        side, stop, tgt = self.pos["side"], self.pos["stop"], self.pos["target"]
        adv = adverse(self.inst, bar.open)
        if not entered:
            if (side > 0 and bar.open <= stop) or (side < 0 and bar.open >= stop):
                return self._close(bar.open - side * adv, "gap_stop", bar.ts)
        through = self.inst.target_through_ticks * self.inst.tick_size
        hit_sl = bar.low <= stop if side > 0 else bar.high >= stop
        hit_tp = tgt is not None and (bar.high >= tgt + through if side > 0 else bar.low <= tgt - through)
        if hit_sl:                                   # also wins any tie with the target
            return self._close(stop - side * adverse(self.inst, stop), "stop", bar.ts)
        if hit_tp:
            return self._close(tgt, "target", bar.ts)
        return None

    def _close(self, px: float, reason: str, ts) -> Event:
        p = self.pos
        inst = self.inst
        costs = p["qty"] * (2 * inst.commission_per_unit + inst.fee_bps * 1e-4 * (p["entry"] + px) * inst.point_value)
        pnl = p["side"] * (px - p["entry"]) * inst.point_value * p["qty"] - costs
        self.cash += pnl
        self.pos = None
        return Event("closed", p["side"], p["qty"], px, str(ts), reason=reason, pnl=pnl, client_id=p["cid"],
                     extra={"costs": costs})

    def state(self) -> dict:
        return {"cash": self.cash, "pos": self.pos, "pending": self.pending}

    def restore(self, d: dict) -> None:
        if d:
            self.cash, self.pos, self.pending = d.get("cash", self.cash), d.get("pos"), d.get("pending")
