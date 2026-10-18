"""The interface between the run loop and a venue.

Contract every adapter honours:
  * open_bracket opens a position WITH its stop (and target, if any) resting at
    the venue. If the stop cannot be placed, the position is closed at once and
    BrokerError is raised — a position is never left unprotected.
  * stop/target are DISTANCES from the actual fill price.
  * client_id is attached to every order so a retried request cannot duplicate it.
  * the venue is the source of truth: position() and equity() ask it directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Bar, BrokerPosition, Event


class BrokerError(RuntimeError):
    pass


class Broker(ABC):
    name = "base"
    live = False

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def position(self) -> BrokerPosition | None: ...

    @abstractmethod
    def open_bracket(self, side: int, qty: float, stop_dist: float, target_dist: float | None,
                     client_id: str, ref_price: float, ts: str) -> list[Event]:
        """Live adapters return the 'opened' event; the paper broker fills at the next bar."""

    @abstractmethod
    def close_position(self, reason: str, client_id: str, ref_price: float, ts: str) -> list[Event]: ...

    def on_bar(self, bar: Bar) -> list[Event]:
        """Called once per closed bar. Paper: simulate fills. Live: detect venue-side exits."""
        return []

    def cancel_all(self) -> None:
        pass

    def state(self) -> dict:
        return {}

    def restore(self, d: dict) -> None:
        pass
