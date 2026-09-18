"""The three 1h strategies, plus shared plumbing.

Each builder returns (orders, atr, context):
  orders  - Series of engine.Order|None, the intent formed at that bar's close
  atr     - ATR series used for trailing stops
  context - the indicator frame, kept for inspection and for the regime/setup tools

All three size positions the same way (the engine does it): risk_pct of current
equity between entry and stop. That is what makes their R-multiples comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .engine import Order


# --------------------------------------------------------------------------
# S1 - Donchian Trend Breakout with volatility-expansion filter
# --------------------------------------------------------------------------
@dataclass
class S1Params:
    donchian: int = 55
    ema_trend: int = 200
    adx_period: int = 14
    adx_min: float = 20.0
    atr_period: int = 14
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    vol_pct_window: int = 500
    vol_pct_min: float = 0.35
    max_bars: int = 240
    allow_short: bool = True


def s1_trend_breakout(df: pd.DataFrame, p: S1Params = S1Params()):
    dc = ind.donchian(df, p.donchian)
    ema_t = ind.ema(df["close"], p.ema_trend)
    adx_df = ind.adx(df, p.adx_period)
    atr_s = ind.atr(df, p.atr_period)
    rv = ind.realized_vol(df["close"], 24)
    rv_rank = ind.percentile_rank(rv, p.vol_pct_window)

    ctx = pd.DataFrame(
        {"dc_up": dc["upper"], "dc_dn": dc["lower"], "ema": ema_t,
         "adx": adx_df["adx"], "atr": atr_s, "rv_rank": rv_rank}
    )

    regime_ok = (ctx["adx"] >= p.adx_min) & (ctx["rv_rank"] >= p.vol_pct_min)
    long_sig = regime_ok & (df["close"] > ctx["dc_up"]) & (df["close"] > ctx["ema"])
    short_sig = regime_ok & (df["close"] < ctx["dc_dn"]) & (df["close"] < ctx["ema"])

    orders = pd.Series([None] * len(df), index=df.index, dtype=object)
    sd = p.stop_atr * atr_s
    for i, ts in enumerate(df.index):
        if not np.isfinite(sd.iloc[i]) or sd.iloc[i] <= 0:
            continue
        if long_sig.iloc[i]:
            orders.iloc[i] = Order("long", float(sd.iloc[i]), None, p.trail_atr, p.max_bars, "S1")
        elif p.allow_short and short_sig.iloc[i]:
            orders.iloc[i] = Order("short", float(sd.iloc[i]), None, p.trail_atr, p.max_bars, "S1")
    return orders, atr_s, ctx


# --------------------------------------------------------------------------
# S2 - Range Mean-Reversion (fade the band, target the mean)
# --------------------------------------------------------------------------
@dataclass
class S2Params:
    bb_window: int = 20
    bb_std: float = 2.5
    rsi_period: int = 2
    rsi_long_max: float = 5.0
    rsi_short_min: float = 95.0
    adx_period: int = 14
    adx_max: float = 18.0
    ema_regime: int = 200
    atr_period: int = 14
    stop_atr: float = 1.5
    tp_atr: float = 1.5          # ~ the band mid; expressed in ATR for stability
    max_bars: int = 48
    allow_short: bool = True


def s2_mean_reversion(df: pd.DataFrame, p: S2Params = S2Params()):
    bb = ind.bollinger(df["close"], p.bb_window, p.bb_std)
    r = ind.rsi(df["close"], p.rsi_period)
    adx_df = ind.adx(df, p.adx_period)
    atr_s = ind.atr(df, p.atr_period)
    ema_r = ind.ema(df["close"], p.ema_regime)

    ctx = pd.DataFrame({"bb_up": bb["upper"], "bb_dn": bb["lower"], "bb_mid": bb["mid"],
                        "rsi": r, "adx": adx_df["adx"], "atr": atr_s, "ema": ema_r})

    ranging = ctx["adx"] <= p.adx_max
    long_sig = ranging & (df["close"] < ctx["bb_dn"]) & (ctx["rsi"] < p.rsi_long_max)
    short_sig = ranging & (df["close"] > ctx["bb_up"]) & (ctx["rsi"] > p.rsi_short_min)

    orders = pd.Series([None] * len(df), index=df.index, dtype=object)
    sd = p.stop_atr * atr_s
    td = p.tp_atr * atr_s
    for i in range(len(df)):
        if not np.isfinite(sd.iloc[i]) or sd.iloc[i] <= 0:
            continue
        if long_sig.iloc[i]:
            orders.iloc[i] = Order("long", float(sd.iloc[i]), float(td.iloc[i]), None, p.max_bars, "S2")
        elif p.allow_short and short_sig.iloc[i]:
            orders.iloc[i] = Order("short", float(sd.iloc[i]), float(td.iloc[i]), None, p.max_bars, "S2")
    return orders, atr_s, ctx


# --------------------------------------------------------------------------
# S3 - Volatility Squeeze Expansion
# --------------------------------------------------------------------------
@dataclass
class S3Params:
    bb_window: int = 20
    bb_std: float = 2.0
    kc_window: int = 20
    kc_mult: float = 1.5
    squeeze_min_bars: int = 6
    ema_bias: int = 50
    atr_period: int = 14
    stop_atr: float = 1.2
    tp_r: float = 3.0            # take-profit as a multiple of initial risk
    trail_atr: float = 2.5
    max_bars: int = 72
    allow_short: bool = True


def s3_squeeze_expansion(df: pd.DataFrame, p: S3Params = S3Params()):
    sq = ind.squeeze_on(df, p.bb_window, p.bb_std, p.kc_window, p.kc_mult)
    kc = ind.keltner(df, p.kc_window, p.kc_mult)
    atr_s = ind.atr(df, p.atr_period)
    bias = ind.ema(df["close"], p.ema_bias)
    bias_up = bias > bias.shift(1)

    # Count consecutive squeeze bars ending at t-1, then require a release at t.
    sq_int = sq.fillna(False).astype(int)
    grp = (sq_int != sq_int.shift(1)).cumsum()
    run_len = sq_int.groupby(grp).cumsum()
    was_squeezed = run_len.shift(1) >= p.squeeze_min_bars
    released = was_squeezed & (~sq.fillna(False))

    ctx = pd.DataFrame({"squeeze": sq, "run_len": run_len, "kc_up": kc["upper"],
                        "kc_dn": kc["lower"], "atr": atr_s, "ema_bias": bias})

    long_sig = released & (df["close"] > ctx["kc_up"]) & bias_up
    short_sig = released & (df["close"] < ctx["kc_dn"]) & (~bias_up)

    orders = pd.Series([None] * len(df), index=df.index, dtype=object)
    sd = p.stop_atr * atr_s
    for i in range(len(df)):
        if not np.isfinite(sd.iloc[i]) or sd.iloc[i] <= 0:
            continue
        d = float(sd.iloc[i])
        if long_sig.iloc[i]:
            orders.iloc[i] = Order("long", d, p.tp_r * d, p.trail_atr, p.max_bars, "S3")
        elif p.allow_short and short_sig.iloc[i]:
            orders.iloc[i] = Order("short", d, p.tp_r * d, p.trail_atr, p.max_bars, "S3")
    return orders, atr_s, ctx


STRATEGIES = {
    "S1_trend_breakout": (s1_trend_breakout, S1Params),
    "S2_mean_reversion": (s2_mean_reversion, S2Params),
    "S3_squeeze_expansion": (s3_squeeze_expansion, S3Params),
}
