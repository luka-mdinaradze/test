"""Market-regime classification: trend, volatility, and volume state.

This is deliberately mechanical. Point it at a data frame and it tells you what
regime the LATEST bar sits in, plus which of the three strategies that regime
favours. It cannot tell you anything about a market whose data you have not
loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def classify(df: pd.DataFrame, lookback_pct: int = 2000) -> dict:
    ema50 = ind.ema(df["close"], 50)
    ema200 = ind.ema(df["close"], 200)
    adx_df = ind.adx(df, 14)
    rv = ind.realized_vol(df["close"], 24)
    rv_rank = ind.percentile_rank(rv, min(lookback_pct, max(len(df) // 2, 50)))
    atr_s = ind.atr(df, 14)
    atr_pct = atr_s / df["close"] * 100

    last = df.index[-1]
    px = float(df["close"].iloc[-1])
    adx_v = float(adx_df["adx"].iloc[-1])
    di_plus = float(adx_df["plus_di"].iloc[-1])
    di_minus = float(adx_df["minus_di"].iloc[-1])
    e50, e200 = float(ema50.iloc[-1]), float(ema200.iloc[-1])
    rvr = float(rv_rank.iloc[-1]) if np.isfinite(rv_rank.iloc[-1]) else np.nan

    # --- trend -----------------------------------------------------------
    if adx_v < 18:
        trend = "SIDEWAYS / RANGE"
    elif px > e50 > e200 and di_plus > di_minus:
        trend = "BULL (aligned uptrend)"
    elif px < e50 < e200 and di_minus > di_plus:
        trend = "BEAR (aligned downtrend)"
    elif px > e200:
        trend = "BULL, weakening / corrective"
    else:
        trend = "BEAR, weakening / corrective"

    # --- volatility ------------------------------------------------------
    if not np.isfinite(rvr):
        vol_state = "UNKNOWN (insufficient history)"
    elif rvr > 0.8:
        vol_state = "HIGH (top quintile)"
    elif rvr > 0.55:
        vol_state = "ELEVATED"
    elif rvr > 0.25:
        vol_state = "NORMAL"
    else:
        vol_state = "COMPRESSED (bottom quartile) - expansion risk"

    # --- volume ----------------------------------------------------------
    vol_note = "n/a (no volume column)"
    vol_ratio = np.nan
    if "volume" in df.columns:
        v_fast = df["volume"].rolling(24, min_periods=24).mean().iloc[-1]
        v_slow = df["volume"].rolling(24 * 30, min_periods=50).mean().iloc[-1]
        if np.isfinite(v_fast) and np.isfinite(v_slow) and v_slow > 0:
            vol_ratio = float(v_fast / v_slow)
            if vol_ratio > 1.3:
                vol_note = f"EXPANDING ({vol_ratio:.2f}x the 30-day average) - participation confirms the move"
            elif vol_ratio < 0.75:
                vol_note = f"DRYING UP ({vol_ratio:.2f}x the 30-day average) - breakouts here are suspect"
            else:
                vol_note = f"NORMAL ({vol_ratio:.2f}x the 30-day average)"

    # --- what to run / what to avoid -------------------------------------
    if trend.startswith(("BULL (", "BEAR (")) and (np.isnan(rvr) or rvr > 0.35):
        best = "S1 (Donchian trend breakout) - directional, trail wide, let winners run"
        avoid = "Fading extremes. In an aligned trend with ADX>25, mean-reversion shorts into strength are the classic account-killer."
    elif trend == "SIDEWAYS / RANGE" and (np.isnan(rvr) or rvr < 0.6):
        best = "S2 (range mean-reversion) - fade band extremes back to the mean"
        avoid = "Breakout entries. In a low-ADX range most breaks are false and S1 will bleed by a thousand cuts."
    elif vol_state.startswith("COMPRESSED"):
        best = "S3 (squeeze expansion) - position for the break, do not predict direction"
        avoid = "Wide stops and large size. Compression resolves violently and gaps through stops."
    else:
        best = "Reduced size across all three. This is a transition regime - no model has an edge in it."
        avoid = "Adding risk. Transition regimes are where correlated stops cluster."

    return {
        "as_of": last, "price": px, "adx": adx_v, "plus_di": di_plus, "minus_di": di_minus,
        "ema50": e50, "ema200": e200, "trend": trend,
        "realized_vol_annual_pct": float(rv.iloc[-1] * 100) if np.isfinite(rv.iloc[-1]) else np.nan,
        "vol_percentile": rvr, "vol_state": vol_state, "atr_pct_of_price": float(atr_pct.iloc[-1]),
        "volume_state": vol_note, "volume_ratio_24h_vs_30d": vol_ratio,
        "best_strategy": best, "avoid": avoid,
        "data_source": df.attrs.get("source", "unknown"),
        "is_synthetic": bool(df.attrs.get("is_synthetic", False)),
    }


def render(r: dict) -> str:
    lines = [
        f"Regime report - {r['data_source']}  (as of {r['as_of']})",
        f"  Price               {r['price']:,.2f}",
        f"  Trend               {r['trend']}   [ADX {r['adx']:.1f}, +DI {r['plus_di']:.1f} / -DI {r['minus_di']:.1f}]",
        f"  EMA50 / EMA200      {r['ema50']:,.2f} / {r['ema200']:,.2f}",
        f"  Volatility          {r['vol_state']}   [realised {r['realized_vol_annual_pct']:.0f}% ann., "
        f"ATR {r['atr_pct_of_price']:.2f}% of price]",
        f"  Volume              {r['volume_state']}",
        f"  Favoured strategy   {r['best_strategy']}",
        f"  Avoid               {r['avoid']}",
    ]
    if r["is_synthetic"]:
        lines.append("  *** SYNTHETIC DATA - this describes the simulator, not a real market ***")
    return "\n".join(lines)
