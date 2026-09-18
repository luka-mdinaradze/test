"""Indicator library.

Every function takes/returns pandas objects indexed by bar timestamp and is
strictly causal: the value at bar t uses only data up to and including bar t.
Nothing here peeks forward, which is the property the backtester relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows -> RSI 100; all-loss windows -> RSI 0.
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI. Returns columns adx, plus_di, minus_di."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr_ = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "adx": dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(),
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def donchian(df: pd.DataFrame, window: int = 55) -> pd.DataFrame:
    """Donchian channel EXCLUDING the current bar.

    Shifting by one is what makes "close breaks above the prior N-bar high" a
    tradable statement instead of a tautology (the current bar's high is always
    inside the unshifted channel).
    """
    upper = df["high"].rolling(window, min_periods=window).max().shift(1)
    lower = df["low"].rolling(window, min_periods=window).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd, "sd": sd})


def keltner(df: pd.DataFrame, window: int = 20, mult: float = 1.5) -> pd.DataFrame:
    mid = ema(df["close"], window)
    rng = atr(df, window)
    return pd.DataFrame({"mid": mid, "upper": mid + mult * rng, "lower": mid - mult * rng})


def realized_vol(close: pd.Series, window: int = 24, bars_per_year: int = 8760) -> pd.Series:
    """Annualised realised volatility of log returns."""
    r = np.log(close).diff()
    return r.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(bars_per_year)


def zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / sd.replace(0.0, np.nan)


def percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the latest value within its own history (0..1)."""
    return s.rolling(window, min_periods=window).rank(pct=True)


def squeeze_on(df: pd.DataFrame, bb_window: int = 20, bb_std: float = 2.0,
               kc_window: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """True when Bollinger bands sit inside Keltner channels (volatility compression)."""
    bb = bollinger(df["close"], bb_window, bb_std)
    kc = keltner(df, kc_window, kc_mult)
    return (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
