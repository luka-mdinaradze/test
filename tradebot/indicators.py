"""A few standard, strictly causal indicators for strategies to use.

Each takes pandas Series and returns a Series aligned to the input; the value at
row t uses rows <= t only. Add your own here or compute them inside a strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's average true range."""
    h, lo, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    prev = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev), np.abs(lo - prev)))   # fmax skips the first bar's NaN
    return pd.Series(tr, index=high.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def zscore(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n, min_periods=n).mean()) / x.rolling(n, min_periods=n).std()


def highest(x: pd.Series, n: int, exclude_current: bool = True) -> pd.Series:
    """Highest value of the previous n bars (excluding the current bar by default)."""
    s = x.shift(1) if exclude_current else x
    return s.rolling(n, min_periods=n).max()


def lowest(x: pd.Series, n: int, exclude_current: bool = True) -> pd.Series:
    s = x.shift(1) if exclude_current else x
    return s.rolling(n, min_periods=n).min()


def crossed_above(a: pd.Series, b) -> bool:
    """True if `a` crossed above `b` on the LAST bar (b may be a Series or a number)."""
    b_now = b.iloc[-1] if isinstance(b, pd.Series) else b
    b_prev = b.iloc[-2] if isinstance(b, pd.Series) else b
    return len(a) >= 2 and a.iloc[-1] > b_now and a.iloc[-2] <= b_prev


def crossed_below(a: pd.Series, b) -> bool:
    b_now = b.iloc[-1] if isinstance(b, pd.Series) else b
    b_prev = b.iloc[-2] if isinstance(b, pd.Series) else b
    return len(a) >= 2 and a.iloc[-1] < b_now and a.iloc[-2] >= b_prev
