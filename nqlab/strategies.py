"""The three candidate strategies and the entry-method study.

All signals are formed at the close of a timeframe bar from causal features and
filled by the engine at the next 1-minute bar's open. Stops are k x ATR(14) of
the signal timeframe; targets are RR x stop; holding time is capped and every
position is closed before the session ends.

Strategy A — statistical trend-following
    z_mom(W) = sum of the last W changes / sqrt(sum of their forecast variances).
    Under a driftless random walk z_mom ~ N(0,1). A fresh crossing of +tau
    (-tau) is evidence of drift at roughly the 1-Phi(tau) level; go long (short).
    Holding cap 2W bars. Hypothesis: drift detected over W bars persists.

Strategy B — statistical mean-reversion
    z_bb(W) = (P - SMA_W(P)) / SD_W(P). A fresh crossing beyond -z (+z) is a
    statistically large deviation; fade it. Holding cap W bars. Hypothesis:
    large deviations from the rolling mean are partly noise and revert.

Strategy C — regime-adaptive
    Route A's signals through bars classified TREND, B's through bars
    classified RANGE, and take nothing otherwise (or in a flagged volatility
    transition / extreme-volatility state). The classifier and its thresholds
    are fixed from TRAIN data. Hypothesis: the classifier knows, ahead of time,
    which of A or B the next bars will reward.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import features
from .engine import Signals

META_COLS = ["zmom_20", "zbb_20", "er_20", "vr4", "ac1", "vol_regime", "range_z", "volume_z",
             "tod_sin", "tod_cos", "rth", "bucket", "atr", "sigma"]


@dataclass(frozen=True)
class AParams:
    w: int
    tau: float
    k: float
    rr: float
    hold_mult: float = 2.0

    def key(self) -> str:
        return f"A w={self.w} tau={self.tau} k={self.k} rr={self.rr}"


@dataclass(frozen=True)
class BParams:
    w: int
    z: float
    k: float
    rr: float
    hold_mult: float = 1.0

    def key(self) -> str:
        return f"B w={self.w} z={self.z} k={self.k} rr={self.rr}"


A_GRID = {"w": [10, 20, 40], "tau": [1.5, 2.0, 2.5], "k": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 2.5, 3.0]}
B_GRID = {"w": [20, 40, 80], "z": [2.0, 2.5, 3.0], "k": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 2.5, 3.0]}


def grid(g: dict, cls):
    import itertools
    keys = list(g)
    for vals in itertools.product(*(g[k] for k in keys)):
        yield cls(**dict(zip(keys, vals)))


def cross_up(x: np.ndarray, th: float) -> np.ndarray:
    prev = np.r_[np.nan, x[:-1]]
    return (x >= th) & (prev < th)


def cross_down(x: np.ndarray, th: float) -> np.ndarray:
    prev = np.r_[np.nan, x[:-1]]
    return (x <= th) & (prev > th)


def _make(f: pd.DataFrame, long: np.ndarray, short: np.ndarray, k: float, rr: float,
          hold_bars: float, tf: int, z: np.ndarray | None = None, tag: str = "") -> Signals:
    atr = f["atr"].to_numpy()
    ok = np.isfinite(atr) & (atr > 0)
    long, short = long & ok, short & ok & ~long
    sel = np.flatnonzero(long | short)
    side = np.where(long[sel], 1, -1)
    stop = k * atr[sel]
    meta = f.iloc[sel][[c for c in META_COLS if c in f.columns]].reset_index(drop=True)
    meta["side"] = side
    meta["z_entry"] = z[sel] if z is not None else np.nan
    meta["stop_atr"] = k
    meta["rr"] = rr
    meta["tf"] = tf
    meta["tag"] = tag
    meta["bar"] = sel
    return Signals(f["end_idx"].to_numpy()[sel], side, stop, rr * stop,
                   np.full(len(sel), max(1, int(round(hold_bars * tf)))), meta)


def strategy_a(f: pd.DataFrame, tf: int, p: AParams) -> Signals:
    z = f[f"zmom_{p.w}"].to_numpy()
    return _make(f, cross_up(z, p.tau), cross_down(z, -p.tau), p.k, p.rr, p.hold_mult * p.w, tf, z, "A")


def strategy_b(f: pd.DataFrame, tf: int, p: BParams) -> Signals:
    z = f[f"zbb_{p.w}"].to_numpy()
    return _make(f, cross_down(z, -p.z), cross_up(z, p.z), p.k, p.rr, p.hold_mult * p.w, tf, z, "B")


def merge(parts: list[Signals]) -> Signals:
    parts = [p for p in parts if len(p)]
    if not parts:
        return Signals([], [], [], [], [])
    meta = pd.concat([p.meta for p in parts], ignore_index=True) if all(p.meta is not None for p in parts) else None
    return Signals(np.concatenate([p.idx for p in parts]), np.concatenate([p.side for p in parts]),
                   np.concatenate([p.stop for p in parts]), np.concatenate([p.tp for p in parts]),
                   np.concatenate([p.hold for p in parts]), meta)


def strategy_c(f: pd.DataFrame, tf: int, pa: AParams, pb: BParams, regime: np.ndarray,
               blocked: np.ndarray | None = None) -> Signals:
    """A where regime==TREND, B where regime==RANGE, nothing elsewhere or where blocked."""
    from .regime import RANGE, TREND
    a, b = strategy_a(f, tf, pa), strategy_b(f, tf, pb)
    keep_a = regime[a.meta["bar"].to_numpy()] == TREND
    keep_b = regime[b.meta["bar"].to_numpy()] == RANGE
    if blocked is not None:
        keep_a &= ~blocked[a.meta["bar"].to_numpy()]
        keep_b &= ~blocked[b.meta["bar"].to_numpy()]
    s = merge([a.subset(keep_a), b.subset(keep_b)])
    if s.meta is not None:
        s.meta["tag"] = np.where(s.meta["tag"] == "A", "C:A", "C:B")
    return s


# --------------------------------------------------------------------------- entry-method study (Module 4)

def entry_methods(f: pd.DataFrame, m1_vwap_dev: np.ndarray, tf: int, train: np.ndarray, w: int = 20,
                  k: float = 1.5, rr: float = 2.0) -> dict[str, Signals]:
    """Each entry trigger in isolation, with IDENTICAL exits (k x ATR stop, RR target,
    W-bar holding cap), so differences come from the trigger alone.
    Thresholds are the natural 2-sigma level of each statistic; the range-shock
    threshold is the top 2% of TRAINING bars (a full-sample quantile would leak)."""
    pc = f["pc"].to_numpy()
    sig = f["sigma"].to_numpy()
    out = {}
    z = f[f"zbb_{w}"].to_numpy()
    out["deviation_fade (z-score)"] = _make(f, cross_down(z, -2.0), cross_up(z, 2.0), k, rr, w, tf, z, "dev")
    nw = features.nw_causal(pc, w / 3.0)
    zn = features.z_deviation(pc, nw, w)
    out["deviation_fade (causal Nadaraya-Watson)"] = _make(f, cross_down(zn, -2.0), cross_up(zn, 2.0), k, rr, w, tf, zn, "nw")
    vd = m1_vwap_dev[f["end_idx"].to_numpy()]
    zv = vd / pd.Series(vd).rolling(w * 5, min_periods=w).std().to_numpy()
    out["deviation_fade (session VWAP)"] = _make(f, cross_down(zv, -2.0), cross_up(zv, 2.0), k, rr, w, tf, zv, "vwap")
    zr = f[f"zreg_{w}"].to_numpy()
    out["regression_trend_change (OLS slope z)"] = _make(f, cross_up(zr, 2.0), cross_down(zr, -2.0), k, rr, w, tf, zr, "reg")
    sl, ss, _ = features.kalman_llt(pc, sig ** 2, 0.05, 0.002, 1.0)
    zk = sl / ss
    out["regression_trend_change (Kalman slope z)"] = _make(f, cross_up(zk, 2.0), cross_down(zk, -2.0), k, rr, w, tf, zk, "kal")
    zm = f[f"zmom_{w}"].to_numpy()
    out["momentum_shift (z_mom)"] = _make(f, cross_up(zm, 2.0), cross_down(zm, -2.0), k, rr, w, tf, zm, "mom")
    rz = f["range_z"].to_numpy()
    thr = np.nanquantile(rz[train], 0.98)
    body = pc - f["po"].to_numpy()
    shock = rz >= thr
    out["volatility_expansion (range shock, follow)"] = _make(f, shock & (body > 0), shock & (body < 0), k, rr, w, tf, rz, "volx")
    hh, ll = f[f"hh_{w}"].to_numpy(), f[f"ll_{w}"].to_numpy()
    up = (pc > hh) & (zm > 1.0)
    dn = (pc < ll) & (zm < -1.0)
    out["breakout (N-bar high/low + z_mom>1)"] = _make(f, up & ~np.r_[False, up[:-1]], dn & ~np.r_[False, dn[:-1]],
                                                       k, rr, w, tf, zm, "brk")
    return out


def params_dict(p) -> dict:
    return asdict(p)
