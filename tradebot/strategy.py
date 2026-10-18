"""The strategy interface — the only part you write.

    class MyStrategy(Strategy):
        warmup = 50                          # bars needed before on_bar is called
        defaults = {"fast": 10, "slow": 30}  # overridable from [strategy.params]

        def on_bar(self, ctx):
            ...
            return Signal.long(stop=..., target=...)   # or Signal.short(...), Exit(...), None

`on_bar` is called once per CLOSED bar. `ctx.bars` holds closed bars only, the
newest last, so a strategy cannot see the future. While a position is open the
platform keeps calling `on_bar`; return Exit(...) to close early, anything else
is ignored until the position is flat. The platform handles sizing, stops,
targets, the time limit, the session flatten and every risk limit.
"""
from __future__ import annotations

import importlib
import inspect

import pandas as pd

from .models import Exit, Position, Signal


class Context:
    def __init__(self, bars: pd.DataFrame, position: Position | None, equity: float,
                 symbol: str, timeframe: str):
        self.bars = bars
        self.position = position
        self.equity = equity
        self.symbol = symbol
        self.timeframe = timeframe
        self.notes: list[str] = []

    @property
    def open(self) -> pd.Series:
        return self.bars["open"]

    @property
    def high(self) -> pd.Series:
        return self.bars["high"]

    @property
    def low(self) -> pd.Series:
        return self.bars["low"]

    @property
    def close(self) -> pd.Series:
        return self.bars["close"]

    @property
    def volume(self) -> pd.Series:
        return self.bars["volume"]

    @property
    def last(self) -> pd.Series:
        return self.bars.iloc[-1]

    @property
    def is_flat(self) -> bool:
        return self.position is None

    def note(self, text: str) -> None:
        """Free-text context for the signal printout (e.g. the regime you detected)."""
        self.notes.append(text)


class Strategy:
    name = "base"
    warmup = 50
    defaults: dict = {}

    def __init__(self, params: dict | None = None):
        unknown = set(params or {}) - set(self.defaults)
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown params {sorted(unknown)}; known: {sorted(self.defaults)}")
        self.params = {**self.defaults, **(params or {})}

    def on_bar(self, ctx: Context) -> Signal | Exit | None:
        return None


def load_strategy(name: str, params: dict | None = None) -> Strategy:
    """`name` is a module in tradebot/strategies ("template") or "package.module:ClassName"."""
    if ":" in name:
        mod_name, cls_name = name.split(":", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)
    else:
        mod = importlib.import_module(f"tradebot.strategies.{name}")
        found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
                 if issubclass(c, Strategy) and c is not Strategy and c.__module__ == mod.__name__]
        if len(found) != 1:
            raise ValueError(f"tradebot/strategies/{name}.py must define exactly one Strategy subclass")
        cls = found[0]
    return cls(params)
