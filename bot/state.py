"""Durable bot state with atomic writes and crash recovery.

A trading bot that forgets what it was doing is more dangerous than no bot at
all: it re-enters a position it already holds, or abandons a stop. So state is
written atomically (temp file + os.replace, which is atomic on POSIX) after
every mutation, and the exchange is always treated as the source of truth on
startup — local state is a cache, never an authority.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    side: str
    qty: float
    entry_price: float
    stop_price: float
    initial_stop: float
    entry_time: str
    entry_bar_index: int
    risk_amount: float
    client_order_id: str
    stop_order_id: Optional[str] = None

    def unrealised(self, price: float) -> float:
        d = price - self.entry_price if self.side == "long" else self.entry_price - price
        return d * self.qty


@dataclass
class PendingEntry:
    client_order_id: str
    side: str
    limit_price: float
    qty: float
    stop_distance: float
    placed_bar_index: int
    placed_at: str


@dataclass
class BotState:
    phase: str = "PAPER"                 # PAPER -> LIVE, never the reverse automatically
    equity: float = 100.0
    peak_equity: float = 100.0
    halted: bool = False
    halt_reason: str = ""
    consecutive_losses: int = 0
    live_trades_done: int = 0
    api_error_count: int = 0
    day_key: str = ""
    day_realised_R: float = 0.0
    last_processed_bar: str = ""
    position: Optional[Position] = None
    pending_entry: Optional[PendingEntry] = None
    trades: list = field(default_factory=list)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    # ---- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        self.updated_at = utcnow()
        d = asdict(self)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(d, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)        # atomic: never a half-written state file
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str) -> "BotState":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open() as f:
            d = json.load(f)
        pos = d.pop("position", None)
        pend = d.pop("pending_entry", None)
        st = cls(**d)
        st.position = Position(**pos) if pos else None
        st.pending_entry = PendingEntry(**pend) if pend else None
        return st

    # ---- derived -------------------------------------------------------
    @property
    def drawdown(self) -> float:
        return self.equity / self.peak_equity - 1.0 if self.peak_equity > 0 else 0.0

    def mark_equity(self, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def record_trade(self, trade: dict) -> None:
        self.trades.append(trade)
        r = trade.get("r_multiple", 0.0)
        self.day_realised_R += r
        if trade.get("pnl", 0.0) <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.phase == "LIVE":
            self.live_trades_done += 1

    def roll_day(self, day_key: str) -> None:
        if day_key != self.day_key:
            self.day_key = day_key
            self.day_realised_R = 0.0
