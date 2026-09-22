"""Signal generation for the live bot.

Reuses the backtested S1 rules so the bot trades exactly what was tested. The
bot evaluates only CLOSED bars — the forming bar is never used for a decision,
because its high/low/close are not final and acting on them is a subtle form of
lookahead that works beautifully in a backtest and loses money live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from quant import indicators as ind

from .config import TradingConfig


@dataclass
class Signal:
    action: str                 # "enter" | "none"
    side: Optional[str] = None  # "long"
    entry_ref: float = 0.0      # reference price for the limit order
    stop_distance: float = 0.0
    atr: float = 0.0
    reason: str = ""
    bar_time: Optional[pd.Timestamp] = None


def evaluate(df: pd.DataFrame, cfg: TradingConfig) -> Signal:
    """Decide from the last CLOSED bar of df."""
    need = max(cfg.ema_trend, cfg.donchian, cfg.vol_pct_window) + 10
    if len(df) < need:
        return Signal("none", reason=f"warming up: {len(df)}/{need} bars")

    dc = ind.donchian(df, cfg.donchian)
    ema_t = ind.ema(df["close"], cfg.ema_trend)
    adx_df = ind.adx(df, cfg.adx_period)
    atr_s = ind.atr(df, cfg.atr_period)
    rv = ind.realized_vol(df["close"], 24)
    rv_rank = ind.percentile_rank(rv, cfg.vol_pct_window)

    i = -1
    close = float(df["close"].iloc[i])
    atr_v = float(atr_s.iloc[i])
    adx_v = float(adx_df["adx"].iloc[i])
    rvr = float(rv_rank.iloc[i]) if np.isfinite(rv_rank.iloc[i]) else np.nan
    up = float(dc["upper"].iloc[i])
    ema_v = float(ema_t.iloc[i])
    ts = df.index[i]

    if not np.isfinite(atr_v) or atr_v <= 0:
        return Signal("none", reason="ATR unavailable", bar_time=ts)
    if not np.isfinite(adx_v) or adx_v < cfg.adx_min:
        return Signal("none", reason=f"ADX {adx_v:.1f} < {cfg.adx_min:g}", bar_time=ts)
    if not np.isfinite(rvr) or rvr < cfg.vol_pct_min:
        return Signal("none", reason=f"vol percentile {rvr:.2f} < {cfg.vol_pct_min:g}", bar_time=ts)
    if not (close > up):
        return Signal("none", reason=f"close {close:.2f} has not cleared {up:.2f}", bar_time=ts)
    if not (close > ema_v):
        return Signal("none", reason=f"close below EMA{cfg.ema_trend}", bar_time=ts)

    return Signal("enter", "long", close, cfg.stop_atr * atr_v, atr_v,
                  f"close {close:.2f} > Donchian{cfg.donchian} {up:.2f}, ADX {adx_v:.1f}, "
                  f"vol pct {rvr:.2f}", ts)


def trail_stop(cfg: TradingConfig, side: str, current_stop: float,
               close: float, atr: float) -> float:
    """Chandelier trail. Ratchets one way only — never loosens a stop."""
    if side == "long":
        return max(current_stop, close - cfg.trail_atr * atr)
    return min(current_stop, close + cfg.trail_atr * atr)
