"""CME futures (MNQ, NQ, ES, ...) through Interactive Brokers TWS / IB Gateway (ib_async).

Runs on macOS: install TWS or IB Gateway, enable the API (Configure -> API ->
Settings: "Enable ActiveX and Socket Clients"), and point [ibkr] at it. Ports:
7497 TWS paper, 4002 Gateway paper (7496 / 4001 are LIVE).

Entry, stop and target go up as ONE bracket: a market parent with a stop and a
limit child attached, transmitted together, so the stop exists from the instant
the entry fills. Because the children are priced before the fill, their levels
are anchored to the last close (`ref_price`), not the fill — a difference of the
entry slippage.

Written against ib_async 2.x; tested here only with fake objects (no gateway is
reachable from the environment it was written in). Use a PAPER account first.
"""
from __future__ import annotations

import logging

from ..config import Config
from ..models import LONG, SHORT, BrokerPosition, Event
from .base import Broker, BrokerError

log = logging.getLogger(__name__)
BAR_SIZE = {"1m": "1 min", "3m": "3 mins", "5m": "5 mins", "15m": "15 mins", "30m": "30 mins",
            "1h": "1 hour", "2h": "2 hours", "4h": "4 hours", "1d": "1 day"}


def round_tick(px: float, tick: float) -> float:
    return round(round(px / tick) * tick, 10)


def bracket_orders(next_id, side: int, qty: float, stop_price: float, target_price: float | None, ref: str):
    """Parent market order + attached stop (+ target). The LAST order transmits all."""
    from ib_async import LimitOrder, MarketOrder, StopOrder
    action, rev = ("BUY", "SELL") if side > 0 else ("SELL", "BUY")
    parent = MarketOrder(action, qty, orderId=next_id(), transmit=False, orderRef=ref)
    out = [parent]
    if target_price is not None:
        out.append(LimitOrder(rev, qty, target_price, orderId=next_id(), parentId=parent.orderId,
                              transmit=False, tif="GTC", orderRef=ref + "-tp"))
    out.append(StopOrder(rev, qty, stop_price, orderId=next_id(), parentId=parent.orderId,
                         transmit=True, tif="GTC", orderRef=ref + "-sl"))
    return out


class IBKRBroker(Broker):
    name = "ibkr"
    live = True

    def __init__(self, cfg: Config, ib=None):
        self.cfg = cfg
        self.ib = ib
        self.contract = None
        self.multiplier = cfg.instrument.point_value
        self._open: dict | None = None

    def connect(self) -> None:
        from ib_async import IB, ContFuture, Future
        c = self.cfg.ibkr
        if self.ib is None:
            self.ib = IB()
        if not self.ib.isConnected():
            self.ib.connect(c.host, c.port, clientId=c.client_id)
        cf = ContFuture(c.symbol, c.exchange, currency=c.currency)
        if not self.ib.qualifyContracts(cf):
            raise BrokerError(f"could not resolve {c.symbol} on {c.exchange}")
        self.contract = Future(conId=cf.conId, exchange=c.exchange)
        self.ib.qualifyContracts(self.contract)
        self.multiplier = float(self.contract.multiplier or self.cfg.instrument.point_value)
        if abs(self.multiplier - self.cfg.instrument.point_value) > 1e-9:
            raise BrokerError(f"IB says the multiplier is {self.multiplier}, config says "
                              f"{self.cfg.instrument.point_value}: fix [instrument].point_value")
        log.info("IB connected: %s %s", self.contract.localSymbol, self.contract.lastTradeDateOrContractMonth)

    def equity(self) -> float:
        for v in self.ib.accountSummary():
            if v.tag == "NetLiquidation" and v.currency in (self.cfg.ibkr.currency, "BASE"):
                return float(v.value)
        raise BrokerError("NetLiquidation not available from IB")

    def position(self) -> BrokerPosition | None:
        for p in self.ib.positions():
            if p.contract.conId == self.contract.conId and p.position != 0:
                side = LONG if p.position > 0 else SHORT
                return BrokerPosition(side, abs(float(p.position)), float(p.avgCost) / self.multiplier)
        return None

    def _wait_fill(self, trade, seconds: float = 10.0) -> float | None:
        waited = 0.0
        while not trade.isDone() and waited < seconds:
            self.ib.sleep(0.25)
            waited += 0.25
        px = trade.orderStatus.avgFillPrice
        return float(px) if px else None

    def open_bracket(self, side, qty, stop_dist, target_dist, client_id, ref_price, ts) -> list[Event]:
        if self.position() is not None:
            raise BrokerError("IB already shows a position in this contract")
        tick = self.cfg.instrument.tick_size
        stop = round_tick(ref_price - side * stop_dist, tick)
        target = round_tick(ref_price + side * target_dist, tick) if target_dist else None
        eq0 = self.equity()
        orders = bracket_orders(self.ib.client.getReqId, side, int(qty), stop, target, client_id)
        trades = [self.ib.placeOrder(self.contract, o) for o in orders]
        fill = self._wait_fill(trades[0])
        if fill is None:
            for t in trades:
                self.ib.cancelOrder(t.order)
            raise BrokerError("entry did not fill within 10 s; bracket cancelled")
        self._open = {"equity_before": eq0, "side": side, "qty": int(qty), "cid": client_id}
        return [Event("opened", side, int(qty), fill, ts, stop_price=stop, target_price=target, client_id=client_id)]

    def cancel_all(self) -> None:
        for t in self.ib.openTrades():
            if t.contract.conId == self.contract.conId:
                self.ib.cancelOrder(t.order)

    def close_position(self, reason, client_id, ref_price, ts) -> list[Event]:
        from ib_async import MarketOrder
        self.cancel_all()
        pos = self.position()
        if pos is None:
            return self._venue_closed(ts, reason) if self._open else []
        t = self.ib.placeOrder(self.contract, MarketOrder("SELL" if pos.side > 0 else "BUY", int(pos.qty),
                                                          orderRef=client_id))
        px = self._wait_fill(t)
        return self._venue_closed(ts, reason, px)

    def on_bar(self, bar) -> list[Event]:
        if self._open and self.position() is None:
            self.cancel_all()
            return self._venue_closed(str(bar.ts), "stop/target filled at IB", None)
        return []

    def _venue_closed(self, ts, reason, price=None) -> list[Event]:
        o, self._open = self._open, None
        if o is None:
            return []
        self.ib.sleep(1.0)                         # let the account summary catch up
        return [Event("closed", o["side"], o["qty"], price, ts, reason=reason,
                      pnl=self.equity() - o["equity_before"], client_id=o["cid"])]

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()

    def state(self) -> dict:
        return {"open": self._open}

    def restore(self, d: dict) -> None:
        self._open = (d or {}).get("open")
