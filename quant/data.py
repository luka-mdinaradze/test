"""Data loading.

Three sources, in order of preference:
  1. A local CSV you already have (`load_csv`).
  2. A live fetch from Binance / Yahoo (`fetch_binance_klines`, `fetch_yahoo_daily`).
  3. `synthetic_ohlc`, a regime-switching simulator.

IMPORTANT: synthetic data validates the PLUMBING, never the strategy. Any
performance number produced on synthetic bars describes the simulator's
parameters, not Bitcoin. Results carry a `source` attribute so reports can say
which one they came from, and `is_synthetic` is checked by the report writer.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data"
BINANCE_LIMIT = 1000


def _tag(df: pd.DataFrame, source: str, synthetic: bool) -> pd.DataFrame:
    df.attrs["source"] = source
    df.attrs["is_synthetic"] = synthetic
    return df


def load_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """CSV with columns: open_time|timestamp|date, open, high, low, close, volume."""
    df = pd.read_csv(path)
    tcol = next((c for c in ("open_time", "timestamp", "date", "time") if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"no time column found in {path}")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce", format="mixed")
    if ts.isna().all():                      # epoch millis
        ts = pd.to_datetime(df[tcol], unit="ms", utc=True)
    df.index = ts.dt.tz_convert(tz)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    out = df[keep].astype(float).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return _tag(out, f"csv:{Path(path).name}", False)


def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1h",
                         start: str = "2017-08-17", end: str | None = None,
                         cache: bool = True, base: str = "https://api.binance.com") -> pd.DataFrame:
    """Paginated spot klines. Needs outbound access to api.binance.com.

    Binance serves 1h BTCUSDT back to Aug-2017, i.e. the full 5-10 year window.
    """
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"{symbol}_{interval}.csv"
    if cache and cache_file.exists():
        return _tag(load_csv(cache_file), f"binance-cache:{symbol}:{interval}", False)

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow().tz_localize("UTC")).timestamp() * 1000)
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        q = urllib.parse.urlencode(
            {"symbol": symbol, "interval": interval, "startTime": cur, "limit": BINANCE_LIMIT}
        )
        with urllib.request.urlopen(f"{base}/api/v3/klines?{q}", timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)                      # stay under the weight limit

    if not rows:
        raise RuntimeError("Binance returned no data")
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if cache:
        out.to_csv(cache_file, index_label="open_time")
    return _tag(out, f"binance:{symbol}:{interval}", False)


def fetch_yahoo_daily(symbol: str = "^NDX", years: int = 10, cache: bool = True) -> pd.DataFrame:
    """Daily bars from Yahoo's public chart endpoint (used for the Nasdaq leg)."""
    CACHE.mkdir(exist_ok=True)
    safe = symbol.replace("^", "").replace("/", "_")
    cache_file = CACHE / f"{safe}_1d.csv"
    if cache and cache_file.exists():
        return _tag(load_csv(cache_file), f"yahoo-cache:{symbol}", False)

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?range={years}y&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.loads(r.read())
    res = payload["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    out = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
         "volume": q["volume"]},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    ).dropna().sort_index()
    if cache:
        out.to_csv(cache_file, index_label="date")
    return _tag(out, f"yahoo:{symbol}", False)


def synthetic_ohlc(bars: int = 8760 * 6, seed: int = 7, start: str = "2019-01-01",
                   freq: str = "h", s0: float = 5000.0) -> pd.DataFrame:
    """Regime-switching price simulator: bull / bear / chop with vol clustering.

    SYNTHETIC. Use it to exercise the engine, not to judge a strategy.
    """
    rng = np.random.default_rng(seed)
    # Per-bar drift and vol by regime, and a sticky transition matrix.
    drift = np.array([0.00012, -0.00010, 0.0])
    vol = np.array([0.0060, 0.0085, 0.0035])
    P = np.array([[0.9990, 0.0004, 0.0006],
                  [0.0006, 0.9988, 0.0006],
                  [0.0005, 0.0005, 0.9990]])

    state = 2
    states = np.empty(bars, dtype=int)
    for i in range(bars):
        state = rng.choice(3, p=P[state])
        states[i] = state

    # GARCH-ish vol clustering around the regime level.
    shock = rng.standard_normal(bars)
    sig = vol[states].copy()
    for i in range(1, bars):
        sig[i] = 0.92 * sig[i - 1] + 0.08 * vol[states[i]] + 0.03 * abs(shock[i - 1]) * vol[states[i]]
    ret = drift[states] + sig * shock
    close = s0 * np.exp(np.cumsum(ret))

    idx = pd.date_range(start, periods=bars, freq=freq, tz="UTC")
    open_ = np.concatenate([[s0], close[:-1]])
    wick = sig * close * rng.uniform(0.3, 1.4, bars)
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = np.exp(rng.normal(0, 0.4, bars)) * 1000 * (1 + 4 * sig / vol.mean())

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": volume}, index=idx)
    df["regime"] = states
    return _tag(df, f"SYNTHETIC(seed={seed})", True)


def load_btc_1h(prefer_live: bool = True, **kw) -> pd.DataFrame:
    """Best available BTC/USDT 1h data, falling back to the simulator."""
    try:
        if prefer_live:
            return fetch_binance_klines(**kw)
    except Exception as e:                     # offline / blocked / rate-limited
        print(f"[data] live fetch unavailable ({type(e).__name__}: {e}); using SYNTHETIC bars")
    return synthetic_ohlc()
