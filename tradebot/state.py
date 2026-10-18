"""Durable bot state and the trade journal.

State is written atomically (temp file + os.replace), so a crash can never leave
a half-written file. On restart the broker is asked what is really open; the
saved state is only the bot's memory of why.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import Position
from .risk import GovernorState


class StateStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def save(self, position: Position | None, governor: GovernorState, broker_state: dict,
             last_bar: str | None, extra: dict | None = None) -> None:
        d = {"position": position.to_dict() if position else None, "governor": governor.to_dict(),
             "broker": broker_state, "last_bar": last_bar, **(extra or {})}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, default=str))
        os.replace(tmp, self.path)

    @staticmethod
    def position_from(d: dict | None) -> Position | None:
        return Position(**d) if d else None

    @staticmethod
    def governor_from(d: dict | None) -> GovernorState:
        return GovernorState(**d) if d else GovernorState()


TRADE_FIELDS = ["opened_at", "closed_at", "side", "qty", "entry", "exit", "stop", "target", "risk", "pnl", "r",
                "bars", "entry_reason", "exit_reason", "client_id"]
SIGNAL_FIELDS = ["ts", "side", "price", "stop_dist", "target_dist", "qty", "status", "detail"]


class Journal:
    """Append-only CSV records: every trade and every signal (taken or not)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.trades = self.dir / "trades.csv"
        self.signals = self.dir / "signals.csv"

    def _append(self, path: Path, fields: list[str], row: dict) -> None:
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    def trade(self, row: dict) -> None:
        self._append(self.trades, TRADE_FIELDS, row)

    def signal(self, row: dict) -> None:
        self._append(self.signals, SIGNAL_FIELDS, row)
