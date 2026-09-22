"""Broker adapters.

`Broker` is the interface the trader speaks. Two implementations:

  PaperBroker  - fully simulated, driven bar-by-bar with the SAME pessimistic
                 fill rules as the backtest engine, so paper results and
                 backtest results are directly comparable. Fully testable.

  BinanceSpotBroker - real REST adapter. It is written but UNVERIFIED in this
                 environment (no network access here), so it refuses to place
                 live orders unless explicitly unlocked. Treat it as a draft to
                 test on a testnet key before it touches real money.

Every order carries a client_order_id. That is what makes retries safe: if a
request times out, resubmitting the SAME id cannot create a second order.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

Side = Literal["buy", "sell"]
OrderStatus = Literal["new", "filled", "canceled", "rejected", "expired"]


@dataclass
class Order:
    client_order_id: str
    side: Side
    type: str                      # limit_maker | market | stop_loss
    qty: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = "new"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    exchange_id: Optional[str] = None
    created_bar: int = 0


class BrokerError(RuntimeError):
    pass


class Broker:
    """Interface. Anything the trader needs from a venue lives here."""

    def klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame: ...
    def equity(self) -> float: ...
    def place_limit_maker(self, symbol: str, side: Side, qty: float, price: float, coid: str) -> Order: ...
    def place_market(self, symbol: str, side: Side, qty: float, coid: str) -> Order: ...
    def place_stop_loss(self, symbol: str, side: Side, qty: float, stop_price: float, coid: str) -> Order: ...
    def get_order(self, symbol: str, coid: str) -> Optional[Order]: ...
    def cancel(self, symbol: str, coid: str) -> None: ...
    def open_orders(self, symbol: str) -> list[Order]: ...
    def base_qty(self, symbol: str) -> float: ...


# --------------------------------------------------------------------------
class PaperBroker(Broker):
    """Bar-driven simulation.

    Fill rules match quant/engine.py deliberately:
      - a resting limit fills only if the bar's range reaches it
      - a stop that gaps fills at the bar's open, not the stop price
      - fees charged on both sides
    """

    def __init__(self, bars: pd.DataFrame, start_equity: float = 100.0,
                 maker_fee_bps: float = 2.0, taker_fee_bps: float = 10.0,
                 slippage_bps: float = 2.0, warmup: int = 500):
        self.bars = bars
        self.i = min(warmup, len(bars) - 1)
        self.cash = start_equity
        self.base = 0.0
        self.maker_fee = maker_fee_bps / 10_000.0
        self.taker_fee = taker_fee_bps / 10_000.0
        self.slip = slippage_bps / 10_000.0
        self.orders: dict[str, Order] = {}
        self.fills: list[dict] = []

    # -- feed ------------------------------------------------------------
    def klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        lo = max(0, self.i - limit + 1)
        return self.bars.iloc[lo:self.i + 1].copy()

    @property
    def price(self) -> float:
        return float(self.bars["close"].iloc[self.i])

    def has_next(self) -> bool:
        return self.i + 1 < len(self.bars)

    def advance(self) -> bool:
        """Move to the next bar and resolve resting orders against its range."""
        if not self.has_next():
            return False
        self.i += 1
        bar = self.bars.iloc[self.i]
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])

        for od in list(self.orders.values()):
            if od.status != "new":
                continue
            if od.type == "limit_maker":
                # A buy limit below the market fills if the bar trades down to it.
                hit = (od.side == "buy" and l <= od.price) or (od.side == "sell" and h >= od.price)
                if hit:
                    self._fill(od, od.price, self.maker_fee)
            elif od.type == "stop_loss":
                if od.side == "sell":
                    if o <= od.stop_price:
                        self._fill(od, o * (1 - self.slip), self.taker_fee)   # gap
                    elif l <= od.stop_price:
                        self._fill(od, od.stop_price * (1 - self.slip), self.taker_fee)
                else:
                    if o >= od.stop_price:
                        self._fill(od, o * (1 + self.slip), self.taker_fee)
                    elif h >= od.stop_price:
                        self._fill(od, od.stop_price * (1 + self.slip), self.taker_fee)
        return True

    def _fill(self, od: Order, price: float, fee_rate: float) -> None:
        notional = od.qty * price
        fee = notional * fee_rate
        if od.side == "buy":
            self.cash -= notional + fee
            self.base += od.qty
        else:
            self.cash += notional - fee
            self.base -= od.qty
        od.status = "filled"
        od.filled_qty = od.qty
        od.avg_price = price
        od.fee = fee
        self.fills.append({"coid": od.client_order_id, "side": od.side, "qty": od.qty,
                           "price": price, "fee": fee, "bar": self.i,
                           "time": str(self.bars.index[self.i])})

    # -- account ---------------------------------------------------------
    def equity(self) -> float:
        return self.cash + self.base * self.price

    def base_qty(self, symbol: str) -> float:
        return self.base

    # -- orders ----------------------------------------------------------
    def _register(self, od: Order) -> Order:
        if od.client_order_id in self.orders:
            return self.orders[od.client_order_id]     # idempotent by construction
        od.created_bar = self.i
        self.orders[od.client_order_id] = od
        return od

    def place_limit_maker(self, symbol, side, qty, price, coid) -> Order:
        return self._register(Order(coid, side, "limit_maker", qty, price=price))

    def place_market(self, symbol, side, qty, coid) -> Order:
        od = self._register(Order(coid, side, "market", qty))
        if od.status == "new":
            px = self.price * (1 + self.slip if side == "buy" else 1 - self.slip)
            self._fill(od, px, self.taker_fee)
        return od

    def place_stop_loss(self, symbol, side, qty, stop_price, coid) -> Order:
        return self._register(Order(coid, side, "stop_loss", qty, stop_price=stop_price))

    def get_order(self, symbol, coid) -> Optional[Order]:
        return self.orders.get(coid)

    def cancel(self, symbol, coid) -> None:
        od = self.orders.get(coid)
        if od and od.status == "new":
            od.status = "canceled"

    def open_orders(self, symbol) -> list[Order]:
        return [o for o in self.orders.values() if o.status == "new"]


# --------------------------------------------------------------------------
class BinanceSpotBroker(Broker):
    """Live Binance spot adapter — UNVERIFIED HERE (no network in this env).

    Refuses to place orders unless `unlocked=True` is passed explicitly. Run it
    against https://testnet.binance.vision first and confirm every path.
    """

    def __init__(self, api_key: str, api_secret: str,
                 base: str = "https://api.binance.com", unlocked: bool = False,
                 recv_window: int = 5000):
        self.key, self.secret = api_key, api_secret.encode()
        self.base, self.unlocked, self.recv_window = base.rstrip("/"), unlocked, recv_window

    # -- transport -------------------------------------------------------
    def _request(self, method: str, path: str, params: dict, signed: bool = False,
                 retries: int = 3) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window
            q = urllib.parse.urlencode(params)
            params["signature"] = hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
        query = urllib.parse.urlencode(params)
        url = f"{self.base}{path}" + (f"?{query}" if query else "")
        req = urllib.request.Request(url, method=method,
                                     headers={"X-MBX-APIKEY": self.key} if signed else {})
        last = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    return json.loads(r.read())
            except Exception as e:
                last = e
                # Exponential backoff. A 429 or 418 means back off hard.
                time.sleep(min(2 ** attempt, 8))
        raise BrokerError(f"{method} {path} failed after {retries}: {last}")

    def _guard(self) -> None:
        if not self.unlocked:
            raise BrokerError(
                "Live broker is locked. This adapter has NOT been verified against a real "
                "exchange in this environment. Test on testnet, then pass unlocked=True.")

    # -- read-only -------------------------------------------------------
    def klines(self, symbol, interval, limit) -> pd.DataFrame:
        raw = self._request("GET", "/api/v3/klines",
                            {"symbol": symbol, "interval": interval, "limit": limit})
        df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume",
                                        "close_time", "qv", "n", "tb", "tq", "ig"])
        df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        out = df[["open", "high", "low", "close", "volume"]].astype(float)
        # Drop the still-forming bar: the bot acts only on CLOSED bars.
        now_ms = int(time.time() * 1000)
        if len(df) and int(df["close_time"].iloc[-1]) > now_ms:
            out = out.iloc[:-1]
        return out

    def equity(self) -> float:
        acc = self._request("GET", "/api/v3/account", {}, signed=True)
        total = 0.0
        for b in acc.get("balances", []):
            free, locked = float(b["free"]), float(b["locked"])
            if free + locked <= 0:
                continue
            if b["asset"] in ("USDT", "BUSD", "USDC"):
                total += free + locked
            elif b["asset"] == "BTC":
                px = float(self._request("GET", "/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])
                total += (free + locked) * px
        return total

    def base_qty(self, symbol: str) -> float:
        asset = symbol.replace("USDT", "").replace("BUSD", "")
        acc = self._request("GET", "/api/v3/account", {}, signed=True)
        for b in acc.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"]) + float(b["locked"])
        return 0.0

    def get_order(self, symbol, coid) -> Optional[Order]:
        try:
            r = self._request("GET", "/api/v3/order",
                              {"symbol": symbol, "origClientOrderId": coid}, signed=True)
        except BrokerError:
            return None
        status_map = {"NEW": "new", "FILLED": "filled", "CANCELED": "canceled",
                      "REJECTED": "rejected", "EXPIRED": "expired",
                      "PARTIALLY_FILLED": "new"}
        executed = float(r.get("executedQty", 0))
        cummulative = float(r.get("cummulativeQuoteQty", 0))
        return Order(
            client_order_id=coid, side=r["side"].lower(), type=r["type"].lower(),
            qty=float(r["origQty"]), price=float(r.get("price") or 0) or None,
            stop_price=float(r.get("stopPrice") or 0) or None,
            status=status_map.get(r["status"], "new"), filled_qty=executed,
            avg_price=(cummulative / executed) if executed else 0.0,
            exchange_id=str(r.get("orderId")),
        )

    def open_orders(self, symbol) -> list[Order]:
        rows = self._request("GET", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        return [o for o in (self.get_order(symbol, r["clientOrderId"]) for r in rows) if o]

    # -- mutating --------------------------------------------------------
    def place_limit_maker(self, symbol, side, qty, price, coid) -> Order:
        self._guard()
        r = self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": side.upper(), "type": "LIMIT_MAKER",
            "quantity": qty, "price": price, "newClientOrderId": coid}, signed=True)
        return Order(coid, side, "limit_maker", qty, price=price, exchange_id=str(r.get("orderId")))

    def place_market(self, symbol, side, qty, coid) -> Order:
        self._guard()
        r = self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": side.upper(), "type": "MARKET",
            "quantity": qty, "newClientOrderId": coid}, signed=True)
        executed = float(r.get("executedQty", 0))
        cq = float(r.get("cummulativeQuoteQty", 0))
        return Order(coid, side, "market", qty, status="filled", filled_qty=executed,
                     avg_price=(cq / executed) if executed else 0.0,
                     exchange_id=str(r.get("orderId")))

    def place_stop_loss(self, symbol, side, qty, stop_price, coid) -> Order:
        self._guard()
        r = self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": side.upper(), "type": "STOP_LOSS_LIMIT",
            "quantity": qty, "stopPrice": stop_price,
            "price": stop_price * (0.995 if side == "sell" else 1.005),  # limit through the stop
            "timeInForce": "GTC", "newClientOrderId": coid}, signed=True)
        return Order(coid, side, "stop_loss", qty, stop_price=stop_price,
                     exchange_id=str(r.get("orderId")))

    def cancel(self, symbol, coid) -> None:
        self._guard()
        try:
            self._request("DELETE", "/api/v3/order",
                          {"symbol": symbol, "origClientOrderId": coid}, signed=True)
        except BrokerError:
            pass        # already gone is a success for our purposes
