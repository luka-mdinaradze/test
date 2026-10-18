"""Causal features on timeframe bars.

Every feature at bar t uses only bars <= t. Inputs are the gap-removed prices
(po/ph/pl/pc from data.prepare), so rolls, weekend gaps and the daily halt
never enter a feature, and the back-adjustment of NQ-A is irrelevant: every
quantity is a price DIFFERENCE, or a ratio of differences.

Statistics that drive entries are standardised by their exact null
distribution under a driftless random walk with the forecast volatility, so a
threshold of 2.0 means the same thing at every timeframe and window:

  z_mom(W) = sum of the last W changes / sqrt(sum of their forecast variances)
  z_reg(W) = OLS slope of the last W closes / its null std under a random walk

Fitted inputs (the intraday volatility and volume profiles) are estimated on the
training period only and passed in explicitly — see `fit_profiles`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass
class Profiles:
    """Time-of-day profiles fitted on TRAINING bars only."""
    tf: int
    vol: np.ndarray          # mean |change| per time-of-day slot, normalised to mean 1
    volume: np.ndarray       # median volume per slot
    slots: int

    def slot(self, dmin: np.ndarray) -> np.ndarray:
        return np.minimum(dmin // self.tf, self.slots - 1)


def fit_profiles(tf_df: pd.DataFrame, tf: int, mask: np.ndarray) -> Profiles:
    slots = 24 * 60 // tf + 1
    d = np.abs(np.diff(tf_df["pc"].to_numpy(), prepend=np.nan))
    s = np.minimum(tf_df["dmin"].to_numpy() // tf, slots - 1)
    m = mask & np.isfinite(d)
    vol = np.full(slots, np.nan)
    volm = np.full(slots, np.nan)
    g = pd.DataFrame({"s": s[m], "d": d[m], "v": tf_df["volume"].to_numpy()[m]}).groupby("s")
    vol[g["d"].mean().index] = g["d"].mean().to_numpy()
    volm[g["v"].median().index] = g["v"].median().to_numpy()
    fill = np.nanmean(vol)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, fill)
    vol = vol / np.mean(vol[np.isfinite(vol)])
    volm = np.where(np.isfinite(volm) & (volm > 0), volm, np.nanmedian(volm))
    return Profiles(tf, vol, volm, slots)


# --------------------------------------------------------------------------- volatility

def wilder_atr(h, l, c, n=14) -> np.ndarray:
    prev = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().to_numpy()


@njit(cache=True)
def _ewma_level(x2, lam, init):
    """level_t = lam*level_{t-1} + (1-lam)*x2_t ; forecast for t+1 is level_t."""
    out = np.empty(len(x2))
    lev = init
    for i in range(len(x2)):
        v = x2[i]
        if np.isfinite(v):
            lev = lam * lev + (1.0 - lam) * v
        out[i] = lev
    return out


def seasonal_vol(pc: np.ndarray, slot_factor: np.ndarray, lam: float = 0.97) -> tuple[np.ndarray, np.ndarray]:
    """Seasonally-adjusted EWMA volatility of bar changes.

    Returns (sigma_now, sigma_next_fcst):
      sigma_now[t]  std assumed for bar t's change, from information up to t-1
      level[t]      deseasonalised variance level after observing bar t
    The forecast for bar t+1 is sqrt(level[t]) * slot_factor[t+1].
    """
    d = np.diff(pc, prepend=np.nan)
    u2 = (d / slot_factor) ** 2
    fin = np.flatnonzero(np.isfinite(u2))
    if len(fin) == 0:
        nan = np.full(len(pc), np.nan)
        return nan, nan
    first = fin[0]
    # seeded with the first observation only, so no bar ever sees a later one
    level = _ewma_level(u2, lam, u2[first])
    level[:first] = np.nan
    prev_level = np.r_[np.nan, level[:-1]]
    prev_level[: first + 1] = np.nan
    sigma_now = np.sqrt(prev_level) * slot_factor
    return sigma_now, level


# --------------------------------------------------------------------------- trend / mean-reversion statistics

def z_momentum(pc: np.ndarray, sigma_now: np.ndarray, w: int) -> np.ndarray:
    d = np.diff(pc, prepend=np.nan)
    num = pd.Series(d).rolling(w, min_periods=w).sum().to_numpy()
    den = np.sqrt(pd.Series(sigma_now ** 2).rolling(w, min_periods=w).sum().to_numpy())
    return num / den


def reg_null_const(w: int) -> float:
    """Std of the OLS slope of W random-walk points with unit-variance increments."""
    x = np.arange(w, dtype=float)
    wts = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    tail = np.cumsum(wts[::-1])[::-1][1:]          # sum_{x>=j} w_x, j = 1..W-1
    return float(np.sqrt((tail ** 2).sum()))


def z_regression(pc: np.ndarray, sigma_now: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(slope in points/bar, null-standardised slope z)."""
    x = np.arange(w, dtype=float)
    wts = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    y = np.nan_to_num(pc, nan=0.0)
    slope = np.full(len(pc), np.nan)
    if len(pc) >= w:
        slope[w - 1:] = np.correlate(y, wts, mode="valid")
    sig = np.sqrt(pd.Series(sigma_now ** 2).rolling(w - 1, min_periods=w - 1).mean().to_numpy())
    return slope, slope / (sig * reg_null_const(w))


def z_bollinger(pc: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(pc)
    return ((s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std()).to_numpy()


def nw_causal(pc: np.ndarray, bandwidth: float) -> np.ndarray:
    """One-sided Nadaraya-Watson (Gaussian kernel) estimate at the END point.

    The popular charting 'Nadaraya-Watson envelope' uses a two-sided kernel and
    therefore repaints: its value at t changes once bars after t arrive. Only the
    one-sided endpoint estimate is usable in a backtest, and it is a smoother of
    the same family as an EMA — which is exactly what the comparison tests.
    """
    k = int(np.ceil(3 * bandwidth))
    wts = np.exp(-0.5 * (np.arange(k + 1) / bandwidth) ** 2)
    wts = wts / wts.sum()
    y = np.nan_to_num(pc, nan=0.0)
    out = np.full(len(pc), np.nan)
    if len(pc) > k:
        out[k:] = np.correlate(y, wts[::-1], mode="valid")   # wts[0] weights the newest bar
    return out


def z_deviation(pc: np.ndarray, center: np.ndarray, w: int) -> np.ndarray:
    dev = pd.Series(pc - center)
    return (dev / dev.rolling(w, min_periods=w).std()).to_numpy()


@njit(cache=True)
def kalman_llt(y, sig2, q_level, q_slope, r_obs):
    """Local-linear-trend Kalman filter; noise variances scale with sig2[t].

    Returns filtered slope, its posterior std, and standardised innovations.
    Filtering only (no smoothing): the estimate at t uses observations <= t.
    """
    n = len(y)
    slope = np.full(n, np.nan)
    sstd = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    a0 = y[0]
    a1 = 0.0
    p00, p01, p11 = 1e6, 0.0, 1e6
    for t in range(n):
        s2 = sig2[t]
        if not np.isfinite(s2) or not np.isfinite(y[t]):
            continue
        # predict
        a0 = a0 + a1
        p00 = p00 + 2 * p01 + p11 + q_level * s2
        p01 = p01 + p11
        p11 = p11 + q_slope * s2
        # update
        f = p00 + r_obs * s2
        v = y[t] - a0
        k0 = p00 / f
        k1 = p01 / f
        a0 = a0 + k0 * v
        a1 = a1 + k1 * v
        p11 = p11 - k1 * p01
        p01 = p01 - k0 * p01
        p00 = p00 - k0 * p00
        slope[t] = a1
        sstd[t] = np.sqrt(max(p11, 1e-12))
        innov[t] = v / np.sqrt(f)
    return slope, sstd, innov


# --------------------------------------------------------------------------- regime statistics

def variance_ratio(pc: np.ndarray, q: int, n: int) -> np.ndarray:
    """Rolling variance ratio VR(q) over the last n bars. >1 trending, <1 reverting."""
    d = pd.Series(np.diff(pc, prepend=np.nan))
    sq = pd.Series(pc).diff(q)
    v1 = d.rolling(n, min_periods=n).var()
    vq = sq.rolling(n, min_periods=n).var()
    return (vq / (q * v1)).to_numpy()


def efficiency_ratio(pc: np.ndarray, w: int) -> np.ndarray:
    d = np.abs(np.diff(pc, prepend=np.nan))
    path = pd.Series(d).rolling(w, min_periods=w).sum().to_numpy()
    return pd.Series(pc).diff(w).abs().to_numpy() / path


def rolling_autocorr(pc: np.ndarray, n: int) -> np.ndarray:
    d = pd.Series(np.diff(pc, prepend=np.nan))
    return d.rolling(n, min_periods=n).corr(d.shift(1)).to_numpy()


# --------------------------------------------------------------------------- assembly

def build(tf_df: pd.DataFrame, prof: Profiles, windows=(10, 20, 40, 80), lam: float = 0.97) -> pd.DataFrame:
    """All candidate features for one timeframe frame. Row-aligned with tf_df."""
    po, ph, pl, pc = (tf_df[k].to_numpy(float) for k in ("po", "ph", "pl", "pc"))
    slot = prof.slot(tf_df["dmin"].to_numpy())
    sf = prof.vol[slot]
    sig_now, level = seasonal_vol(pc, sf, lam)
    nxt = prof.slot(tf_df["dmin"].to_numpy() + prof.tf)   # next slot from the clock, not the data
    f = pd.DataFrame(index=tf_df.index)
    f["atr"] = wilder_atr(ph, pl, pc, 14)
    f["sigma"] = sig_now                                   # std assumed for this bar
    f["sigma_next"] = np.sqrt(level) * prof.vol[nxt]       # forecast for the next bar
    f["sigma_level"] = np.sqrt(level)                      # deseasonalised level
    lv = pd.Series(np.log(np.sqrt(level)))
    n_long = max(20, int(round(10 * 1380 / prof.tf)))      # ~10 trading days
    f["vol_regime"] = (lv - lv.rolling(n_long, min_periods=n_long // 2).median()).to_numpy()
    d = np.diff(pc, prepend=np.nan)
    f["ret_z"] = d / sig_now
    f["range_z"] = (ph - pl) / sig_now
    f["volume_z"] = np.log((tf_df["volume"].to_numpy() + 1.0) / (prof.volume[slot] + 1.0))
    for w in windows:
        f[f"zmom_{w}"] = z_momentum(pc, sig_now, w)
        _, f[f"zreg_{w}"] = z_regression(pc, sig_now, w)
        f[f"zbb_{w}"] = z_bollinger(pc, w)
        f[f"er_{w}"] = efficiency_ratio(pc, w)
        f[f"hh_{w}"] = pd.Series(ph).shift(1).rolling(w, min_periods=w).max().to_numpy()
        f[f"ll_{w}"] = pd.Series(pl).shift(1).rolling(w, min_periods=w).min().to_numpy()
    nreg = max(50, int(round(2 * 1380 / prof.tf)))          # ~2 trading days
    f["vr4"] = variance_ratio(pc, 4, nreg)
    f["ac1"] = rolling_autocorr(pc, nreg)
    f["po"] = po
    f["pc"] = pc
    f["dmin"] = tf_df["dmin"].to_numpy()
    f["tod_sin"] = np.sin(2 * np.pi * f["dmin"] / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * f["dmin"] / 1440)
    f["rth"] = tf_df["rth"].to_numpy()
    f["bucket"] = tf_df["bucket"].to_numpy()
    f["tday"] = tf_df["tday"].to_numpy()
    f["end_idx"] = tf_df["end_idx"].to_numpy()
    return f


def session_vwap_dev(m1: pd.DataFrame) -> np.ndarray:
    """Close minus the trading-day VWAP (typical price, from the day's first bar)."""
    tp = (m1["ph"] + m1["pl"] + m1["pc"]).to_numpy() / 3.0
    v = m1["volume"].to_numpy() + 1e-9
    g = m1["tday"].to_numpy()
    pv = pd.Series(tp * v).groupby(g).cumsum().to_numpy()
    vv = pd.Series(v).groupby(g).cumsum().to_numpy()
    return m1["pc"].to_numpy() - pv / vv
