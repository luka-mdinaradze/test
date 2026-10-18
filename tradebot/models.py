"""The small set of objects the platform passes around."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

LONG, SHORT = 1, -1
SIDE_NAME = {LONG: "LONG", SHORT: "SHORT"}


@dataclass(frozen=True)
class Bar:
    """One CLOSED bar. `ts` is the bar's OPEN time in UTC."""
    ts: object
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Signal:
    """What a strategy asks for. Distances are in price units, measured from the fill.

    stop         stop-loss distance (> 0). Required: the platform never opens a
                 position without a stop.
    target       take-profit distance, or None for no target
    max_bars     close after this many bars in the trade, or None
    probability  P(target before stop) — ONLY if your model was calibrated and
                 validated out of sample. Leave None otherwise; the platform
                 then prints 'unavailable' instead of inventing a number.
    """
    side: int
    stop: float
    target: Optional[float] = None
    max_bars: Optional[int] = None
    reason: str = ""
    probability: Optional[float] = None

    def __post_init__(self):
        if self.side not in (LONG, SHORT):
            raise ValueError("side must be LONG (+1) or SHORT (-1)")
        if not (self.stop and self.stop > 0):
            raise ValueError("every signal needs a stop distance > 0")
        if self.target is not None and self.target <= 0:
            raise ValueError("target distance must be > 0 or None")

    @classmethod
    def long(cls, stop: float, target: Optional[float] = None, **kw) -> "Signal":
        return cls(LONG, stop, target, **kw)

    @classmethod
    def short(cls, stop: float, target: Optional[float] = None, **kw) -> "Signal":
        return cls(SHORT, stop, target, **kw)


@dataclass(frozen=True)
class Exit:
    """A strategy's request to close the open position at market."""
    reason: str = "strategy exit"


@dataclass
class Position:
    """The platform's record of the open position (mirrors the broker)."""
    side: int
    qty: float
    entry_price: float
    stop_price: float
    target_price: Optional[float]
    opened_at: str
    risk_amount: float
    reason: str = ""
    client_id: str = ""
    bars_held: int = 0
    max_bars: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BrokerPosition:
    """What the venue says we hold."""
    side: int
    qty: float
    avg_price: float


@dataclass
class Event:
    """Something that happened at the broker: a position opened or closed."""
    kind: str                         # "opened" | "closed"
    side: int
    qty: float
    price: Optional[float]
    ts: str
    reason: str = ""
    pnl: Optional[float] = None       # closed only, after costs
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    client_id: str = ""
    extra: dict = field(default_factory=dict)
