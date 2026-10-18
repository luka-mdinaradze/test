"""When the bot may open a position, and when it must be flat.

All decisions are made at a bar's CLOSE; the resulting order executes at the
next bar's open (paper) or immediately (live). Times are exchange-local.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from .config import SessionSection


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Session:
    def __init__(self, s: SessionSection, tf_minutes: int):
        self.tz = ZoneInfo(s.timezone)
        self.start = _minutes(s.trade_start)
        self.end = _minutes(s.trade_end)
        self.flatten = _minutes(s.flatten_at) if s.flatten_at else None
        self.weekdays = set(s.weekdays)
        self.tf = tf_minutes

    def local(self, ts) -> datetime:
        t = pd.Timestamp(ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.tz_convert(self.tz).to_pydatetime()

    def bar_end(self, bar_ts) -> datetime:
        return self.local(bar_ts) + timedelta(minutes=self.tf)

    def trading_day(self, bar_ts) -> str:
        return self.bar_end(bar_ts).strftime("%Y-%m-%d")

    def _in_window(self, m: int) -> bool:
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= m < self.end
        return m >= self.start or m < self.end           # window wraps midnight (e.g. CME 18:00 -> 16:40)

    def _to_flatten(self, t: datetime) -> int | None:
        if self.flatten is None:
            return None
        return (self.flatten - (t.hour * 60 + t.minute)) % 1440

    def entry_allowed(self, bar_ts) -> tuple[bool, str]:
        """An entry decided at this bar's close executes at its end time."""
        t = self.bar_end(bar_ts)
        if t.weekday() not in self.weekdays:
            return False, f"weekday {t:%a} not traded"
        if not self._in_window(t.hour * 60 + t.minute):
            return False, f"{t:%H:%M} outside the entry window"
        left = self._to_flatten(t)
        if left is not None and left <= self.tf:
            return False, f"less than one bar before the {self.flatten // 60:02d}:{self.flatten % 60:02d} flatten"
        return True, ""

    def must_flatten(self, bar_ts) -> bool:
        """True if the NEXT bar would end after the flatten time: close now."""
        left = self._to_flatten(self.bar_end(bar_ts))
        return left is not None and left < self.tf


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
