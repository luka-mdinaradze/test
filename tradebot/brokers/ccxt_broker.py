"""Crypto derivatives through ccxt (Binance USD-M, Bybit, OKX, ... — 100+ venues).

Written against the ccxt 4.x unified API and exercised in tests with a fake
exchange. It has NOT been run against a real venue from the environment it was
written in: run it on the venue's testnet / demo account (sandbox = true, the
default) before any real key goes near it.

Supports perpetual swaps and futures only. Spot cannot hold a reduce-only stop
and target the same way, and cannot short.
"""
from __future__ import annotations

import logging
import time

from ..config import Config, secret
from ..feeds import retry
from ..models import LONG, SHORT, BrokerPosition, Event
from .base import Broker, BrokerError

log = logging.getLogger(__name__)


class CCXTBroker(Broker):
    name = "ccxt"
    live = True

    def __init__(self, cfg: Config, exchange=None):
        self.cfg = cfg
        self.symbol = cfg.market.symbol
        self.ex = exchange
        self.csize = 1.0
        self.quote = "USDT"
        self._open: dict | None = None

    def connect(self) -> None:
        if self.ex is None:
            import ccxt
            c = self.cfg.ccxt
            klass = getattr(ccxt, c.exchange)
            self.ex = klass({"apiKey": secret(c.api_key_env), "secret": secret(c.api_secret_env),
                             "enableRateLimit": True, "options": dict(c.options)})
            if c.sandbox:
                if hasattr(self.ex, "enable_demo_trading"):
                    try:
                        self.ex.enable_demo_trading(True)
                    except Exception:  # venue has a classic testnet instead of demo trading
                        self.ex.set_sandbox_mode(True)
                else:
                    self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        m = self.ex.market(self.symbol)
        if not (m.get("swap") or m.get("future")):
            raise BrokerError(f"{self.symbol} is not a swap/future market; this adapter trades derivatives only")
        self.csize = float(m.get("contractSize") or 1.0)
        self.quote = m.get("settle") or m.get("quote") or "USDT"

    def equity(self) -> float:
        bal = retry(self.ex.fetch_balance, what="fetch_balance")
        return float((bal.get("total") or {}).get(self.quote) or 0.0)

    def position(self) -> BrokerPosition | None:
        for p in retry(self.ex.fetch_positions, [self.symbol], what="fetch_positions"):
            if p.get("symbol") != self.symbol:
                continue
            contracts = float(p.get("contracts") or 0.0)
            if contracts > 0:
                side = LONG if p.get("side") == "long" else SHORT
                return BrokerPosition(side, contracts * self.csize, float(p.get("entryPrice") or 0.0))
        return None

    def _amount(self, qty: float) -> float:
        return float(self.ex.amount_to_precision(self.symbol, qty / self.csize))

    def _price(self, px: float) -> float:
        return float(self.ex.price_to_precision(self.symbol, px))

    def open_bracket(self, side, qty, stop_dist, target_dist, client_id, ref_price, ts) -> list[Event]:
        if self.position() is not None:
            raise BrokerError("venue already shows an open position")
        eq0 = self.equity()
        amt = self._amount(qty)
        if amt <= 0:
            raise BrokerError(f"quantity {qty} rounds to zero contracts")
        buy, sell = ("buy", "sell") if side > 0 else ("sell", "buy")
        o = self.ex.create_order(self.symbol, "market", buy, amt, None, {"clientOrderId": client_id})
        fill = o.get("average")
        if not fill:
            try:
                fill = self.ex.fetch_order(o["id"], self.symbol).get("average")
            except Exception:
                fill = None
        fill = float(fill or ref_price)
        stop = self._price(fill - side * stop_dist)
        try:
            self.ex.create_stop_loss_order(self.symbol, "market", sell, amt, None, stop,
                                           {"reduceOnly": True, "clientOrderId": client_id + "s"})
        except Exception as e:
            log.error("stop rejected (%s) — closing the position at market", e)
            self._flatten(sell, amt)
            raise BrokerError(f"stop order rejected, position closed: {e}") from e
        target = None
        if target_dist:
            target = self._price(fill + side * target_dist)
            try:
                self.ex.create_take_profit_order(self.symbol, "market", sell, amt, None, target,
                                                 {"reduceOnly": True, "clientOrderId": client_id + "t"})
            except Exception as e:      # the stop is in place; the trade stays protected
                log.warning("target order rejected (%s); position is protected by its stop only", e)
                target = None
        self._open = {"equity_before": eq0, "side": side, "qty": amt * self.csize, "cid": client_id}
        return [Event("opened", side, amt * self.csize, fill, ts, stop_price=stop, target_price=target,
                      client_id=client_id)]

    def _flatten(self, exit_side: str, amt: float) -> None:
        self.ex.create_order(self.symbol, "market", exit_side, amt, None, {"reduceOnly": True})

    def close_position(self, reason, client_id, ref_price, ts) -> list[Event]:
        self.cancel_all()
        pos = self.position()
        if pos is None:
            return self._venue_closed(ts, reason) if self._open else []
        exit_side = "sell" if pos.side > 0 else "buy"
        o = self.ex.create_order(self.symbol, "market", exit_side, self._amount(pos.qty), None,
                                 {"reduceOnly": True, "clientOrderId": client_id})
        time.sleep(0.5)
        return self._venue_closed(ts, reason, float(o.get("average") or ref_price))

    def on_bar(self, bar) -> list[Event]:
        if self._open and self.position() is None:
            self.cancel_all()                       # the other leg of the bracket
            return self._venue_closed(str(bar.ts), "stop/target filled on venue", None)
        return []

    def _venue_closed(self, ts, reason, price=None) -> list[Event]:
        o, self._open = self._open, None
        if o is None:
            return []
        pnl = self.equity() - o["equity_before"]       # includes fees and funding: what the account really did
        return [Event("closed", o["side"], o["qty"], price, ts, reason=reason, pnl=pnl, client_id=o["cid"])]

    def cancel_all(self) -> None:
        try:
            self.ex.cancel_all_orders(self.symbol)
        except Exception as e:
            log.warning("cancel_all_orders failed: %s", e)

    def state(self) -> dict:
        return {"open": self._open}

    def restore(self, d: dict) -> None:
        self._open = (d or {}).get("open")
