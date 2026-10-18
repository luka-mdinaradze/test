"""Bar sources. Every feed yields CLOSED bars only, timestamps = bar open, UTC.

Exchange APIs return the bar that is still forming as their last row; trading on
it means trading on prices that have not finished happening. All live feeds here
drop it.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Iterator

import numpy as np
import pandas as pd

from .models import Bar

log = logging.getLogger(__name__)
COLS = ["open", "high", "low", "close", "volume"]


def retry(fn, *args, attempts: int = 6, base: float = 2.0, what: str = "request", **kw):
    """Retry a READ-ONLY call with exponential backoff (2, 4, 8, ... s). Never used
    for order placement, where a blind retry could duplicate an order."""
    for k in range(attempts):
        try:
            return fn(*args, **kw)
        except Exception as e:                      # network hiccups, rate limits, venue maintenance
            if k == attempts - 1:
                raise
            wait = base * 2 ** k
            log.warning("%s failed (%s: %s) — retrying in %.0f s", what, type(e).__name__, e, wait)
            time.sleep(wait)


def to_bars(df: pd.DataFrame) -> list[Bar]:
    return [Bar(ts, *row) for ts, row in zip(df.index, df[COLS].to_numpy(float))]


def _closed(df: pd.DataFrame, tf_minutes: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=tf_minutes) <= now]


# --------------------------------------------------------------------------- CSV

def load_csv(path: str, tf_minutes: int, tz: str = "UTC", start: str = "", end: str = "") -> pd.DataFrame:
    """Read OHLCV from CSV (first column = timestamp, ISO text or unix s/ms) and
    aggregate to the bot's timeframe if the file is finer (e.g. 1m file, 15m bot)."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    tcol = raw.columns[0]
    t = raw[tcol]
    if pd.api.types.is_numeric_dtype(t):
        idx = pd.to_datetime(t, unit="ms" if t.max() > 1e11 else "s", utc=True)
    else:
        idx = pd.to_datetime(t)
        idx = idx.dt.tz_localize(tz) if idx.dt.tz is None else idx
        idx = idx.dt.tz_convert("UTC")
    missing = {"open", "high", "low", "close"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = raw.reindex(columns=COLS).astype(float)
    df["volume"] = df["volume"].fillna(0.0)
    df.index = pd.DatetimeIndex(idx, name="ts")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if len(df) < 3:
        raise ValueError(f"{path}: fewer than 3 bars in the selected window")
    step = df.index.to_series().diff().median() / pd.Timedelta(minutes=1)
    if step > tf_minutes + 1e-9:
        raise ValueError(f"{path}: bars are {step:g} min apart, coarser than the {tf_minutes} min timeframe")
    if step < tf_minutes - 1e-9:
        df = df.resample(f"{tf_minutes}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return df


class CSVFeed:
    def __init__(self, path: str, tf_minutes: int, tz: str = "UTC", start: str = "", end: str = ""):
        self.df = load_csv(path, tf_minutes, tz, start, end)

    def bars(self) -> list[Bar]:
        return to_bars(self.df)


# --------------------------------------------------------------------------- ccxt

class CCXTFeed:
    """Polls an exchange for new closed candles. Public data needs no API key.

    When the broker is also ccxt, pass its exchange object so prices come from the
    same venue (a testnet's prices differ from the main market's)."""

    def __init__(self, symbol: str, timeframe: str, tf_minutes: int, exchange_id: str = "binanceusdm",
                 exchange=None, poll_seconds: int = 10):
        if exchange is None:
            import ccxt
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        self.ex, self.symbol, self.tf, self.tfm, self.poll = exchange, symbol, timeframe, tf_minutes, poll_seconds
        self.last_ts = None

    def _fetch(self, limit: int) -> pd.DataFrame:
        rows = retry(self.ex.fetch_ohlcv, self.symbol, self.tf, limit=limit, what="fetch_ohlcv")
        df = pd.DataFrame(rows, columns=["ts"] + COLS)
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return _closed(df.astype(float), self.tfm)

    def history(self, n: int) -> pd.DataFrame:
        df = self._fetch(min(n + 2, 1500)).tail(n)
        self.last_ts = df.index[-1] if len(df) else None
        return df

    def stream(self) -> Iterator[Bar]:
        while True:
            df = self._fetch(5)
            new = df[df.index > self.last_ts] if self.last_ts is not None else df.tail(1)
            for b in to_bars(new):
                self.last_ts = b.ts
                yield b
            if new.empty:
                time.sleep(self.poll)


# --------------------------------------------------------------------------- IBKR

class IBKRFeed:
    """Bars from IB with keepUpToDate=True; the last (forming) bar is withheld."""

    def __init__(self, ib, contract, timeframe: str, tf_minutes: int, use_rth: bool = False, poll_seconds: int = 5):
        from .brokers.ibkr import BAR_SIZE
        self.ib, self.contract, self.bar_size, self.tfm = ib, contract, BAR_SIZE[timeframe], tf_minutes
        self.use_rth, self.poll = use_rth, poll_seconds
        self.live = None
        self.last_ts = None

    def _frame(self, bars) -> pd.DataFrame:
        df = pd.DataFrame([{"ts": b.date, "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                            "volume": b.volume} for b in bars])
        idx = pd.to_datetime(df.pop("ts"), utc=True)
        df.index = pd.DatetimeIndex(idx, name="ts")
        return df.astype(float)

    def history(self, n: int) -> pd.DataFrame:
        days = max(2, math.ceil(n * self.tfm / 1380) + 2)
        self.live = self.ib.reqHistoricalData(self.contract, "", f"{days} D", self.bar_size, "TRADES",
                                              useRTH=self.use_rth, formatDate=2, keepUpToDate=True)
        df = self._frame(self.live[:-1]).tail(n)
        self.last_ts = df.index[-1] if len(df) else None
        return df

    def stream(self) -> Iterator[Bar]:
        while True:
            self.ib.sleep(self.poll)
            done = self._frame(self.live[:-1]) if len(self.live) > 1 else pd.DataFrame(columns=COLS)
            new = done[done.index > self.last_ts] if self.last_ts is not None else done.tail(1)
            for b in to_bars(new):
                self.last_ts = b.ts
                yield b
