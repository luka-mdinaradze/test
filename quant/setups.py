"""Trade-setup generator.

Turns the current state of the three strategies into concrete, checkable levels:
trigger, entry, stop, target, R:R, and the exact position size a $100 account
risking 1% would take — including whether that size clears the venue's minimum
order value.

A setup is ARMED (conditions already met, trigger on the next bar) or PENDING
(what price must do before it is valid). These are rule outputs, not advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .strategies import S1Params, S2Params, S3Params


def _size(entry: float, stop: float, equity: float, risk_pct: float,
          min_notional: float, max_lev: float) -> dict:
    dist = abs(entry - stop)
    if dist <= 0:
        return {"qty": 0.0, "notional": 0.0, "risk": 0.0, "viable": False,
                "note": "degenerate stop distance"}
    risk_amt = equity * risk_pct
    qty = risk_amt / dist
    notional = qty * entry
    capped = False
    if notional > equity * max_lev:
        qty = equity * max_lev / entry
        notional = qty * entry
        risk_amt = qty * dist
        capped = True
    viable = notional >= min_notional
    note = []
    if capped:
        note.append(f"leverage-capped at {max_lev:g}x, real risk ${risk_amt:.2f} (<{risk_pct*100:g}%)")
    if not viable:
        note.append(f"notional ${notional:.2f} below the ${min_notional:g} venue minimum - NOT TRADEABLE")
    return {"qty": qty, "notional": notional, "risk": risk_amt, "viable": viable,
            "stop_pct": dist / entry * 100, "note": "; ".join(note) or "ok"}


def generate(df: pd.DataFrame, equity: float = 100.0, risk_pct: float = 0.01,
             min_notional: float = 5.0, max_lev: float = 3.0,
             p1: S1Params = S1Params(), p2: S2Params = S2Params(),
             p3: S3Params = S3Params()) -> pd.DataFrame:
    px = float(df["close"].iloc[-1])
    atr_s = ind.atr(df, 14)
    a = float(atr_s.iloc[-1])
    dc = ind.donchian(df, p1.donchian)
    adx_df = ind.adx(df, p1.adx_period)
    bb = ind.bollinger(df["close"], p2.bb_window, p2.bb_std)
    kc = ind.keltner(df, p3.kc_window, p3.kc_mult)
    sq = ind.squeeze_on(df, p3.bb_window, p3.bb_std, p3.kc_window, p3.kc_mult)
    ema200 = float(ind.ema(df["close"], p1.ema_trend).iloc[-1])
    ema50 = ind.ema(df["close"], p3.ema_bias)
    adx_v = float(adx_df["adx"].iloc[-1])
    rows = []

    # --- S1: breakout of the prior 55-bar range, with trend + ADX filter ----
    for side, level, ok_trend in (
        ("long", float(dc["upper"].iloc[-1]), px > ema200),
        ("short", float(dc["lower"].iloc[-1]), px < ema200),
    ):
        entry = level * (1.0005 if side == "long" else 0.9995)
        stop = entry - p1.stop_atr * a if side == "long" else entry + p1.stop_atr * a
        # Trend trades trail rather than take a fixed profit; 3R is the planning target.
        target = entry + 3 * abs(entry - stop) * (1 if side == "long" else -1)
        s = _size(entry, stop, equity, risk_pct, min_notional, max_lev)
        gate = adx_v >= p1.adx_min and ok_trend
        rows.append({
            "strategy": "S1 trend breakout", "side": side,
            "status": "ARMED" if gate else "BLOCKED",
            "trigger": f"1h close {'above' if side == 'long' else 'below'} {level:,.2f} "
                       f"({'prior 55-bar high' if side == 'long' else 'prior 55-bar low'})",
            "gate": f"ADX {adx_v:.1f} vs min {p1.adx_min:g}; price {'>' if px > ema200 else '<'} EMA200 {ema200:,.2f}",
            "entry": entry, "stop": stop, "target": target,
            "rr": 3.0, "qty": s["qty"], "notional": s["notional"], "risk_usd": s["risk"],
            "stop_pct": s["stop_pct"], "viable": s["viable"], "note": s["note"],
            "exit_rule": f"chandelier trail {p1.trail_atr:g}xATR from the close; time-stop {p1.max_bars} bars",
        })

    # --- S2: fade the band back to the mean, range regime only -------------
    for side, level in (("long", float(bb["lower"].iloc[-1])), ("short", float(bb["upper"].iloc[-1]))):
        entry = level
        stop = entry - p2.stop_atr * a if side == "long" else entry + p2.stop_atr * a
        target = float(bb["mid"].iloc[-1])
        rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else np.nan
        s = _size(entry, stop, equity, risk_pct, min_notional, max_lev)
        gate = adx_v <= p2.adx_max
        rows.append({
            "strategy": "S2 mean reversion", "side": side,
            "status": "ARMED" if gate else "BLOCKED",
            "trigger": f"1h close {'below' if side == 'long' else 'above'} {level:,.2f} "
                       f"(BB {p2.bb_std:g}σ) with RSI(2) {'<' + str(p2.rsi_long_max) if side == 'long' else '>' + str(p2.rsi_short_min)}",
            "gate": f"ADX {adx_v:.1f} vs max {p2.adx_max:g} (range regime required)",
            "entry": entry, "stop": stop, "target": target, "rr": rr,
            "qty": s["qty"], "notional": s["notional"], "risk_usd": s["risk"],
            "stop_pct": s["stop_pct"], "viable": s["viable"], "note": s["note"],
            "exit_rule": f"target = 20-bar mean; hard time-stop {p2.max_bars} bars",
        })

    # --- S3: squeeze release ------------------------------------------------
    sq_run = int(sq.fillna(False).astype(int).iloc[::-1].cumprod().sum())
    bias_up = bool(ema50.iloc[-1] > ema50.iloc[-2])
    side = "long" if bias_up else "short"
    level = float(kc["upper"].iloc[-1]) if bias_up else float(kc["lower"].iloc[-1])
    entry = level
    stop = entry - p3.stop_atr * a if side == "long" else entry + p3.stop_atr * a
    target = entry + p3.tp_r * abs(entry - stop) * (1 if side == "long" else -1)
    s = _size(entry, stop, equity, risk_pct, min_notional, max_lev)
    rows.append({
        "strategy": "S3 squeeze expansion", "side": side,
        "status": "ARMED" if sq_run >= p3.squeeze_min_bars else "PENDING",
        "trigger": f"close {'above' if bias_up else 'below'} Keltner {level:,.2f} after ≥"
                   f"{p3.squeeze_min_bars} squeeze bars (current streak: {sq_run})",
        "gate": f"EMA50 slope {'up' if bias_up else 'down'}; squeeze streak {sq_run}",
        "entry": entry, "stop": stop, "target": target, "rr": p3.tp_r,
        "qty": s["qty"], "notional": s["notional"], "risk_usd": s["risk"],
        "stop_pct": s["stop_pct"], "viable": s["viable"], "note": s["note"],
        "exit_rule": f"{p3.tp_r:g}R target or {p3.trail_atr:g}xATR trail; time-stop {p3.max_bars} bars",
    })

    out = pd.DataFrame(rows)
    out.attrs["as_of"] = df.index[-1]
    out.attrs["price"] = px
    out.attrs["source"] = df.attrs.get("source", "unknown")
    out.attrs["is_synthetic"] = bool(df.attrs.get("is_synthetic", False))
    return out


def render(setups: pd.DataFrame, top: int = 3) -> str:
    lines = [f"Trade setups from {setups.attrs['source']} @ {setups.attrs['as_of']} "
             f"(last price {setups.attrs['price']:,.2f})"]
    if setups.attrs.get("is_synthetic"):
        lines.append("*** SYNTHETIC PRICES - levels illustrate the mechanism only ***")
    rank = {"ARMED": 0, "PENDING": 1, "BLOCKED": 2}
    view = setups.sort_values("status", key=lambda s: s.map(rank)).head(top)
    for _, r in view.iterrows():
        lines += [
            "",
            f"[{r['status']}] {r['strategy']} - {r['side'].upper()}",
            f"  Trigger    {r['trigger']}",
            f"  Filter     {r['gate']}",
            f"  Entry      {r['entry']:,.2f}",
            f"  Stop       {r['stop']:,.2f}   ({r['stop_pct']:.2f}% away)",
            f"  Target     {r['target']:,.2f}   R:R {r['rr']:.2f}",
            f"  Size       {r['qty']:.6f} units = ${r['notional']:.2f} notional, risking ${r['risk_usd']:.2f}",
            f"  Exit       {r['exit_rule']}",
            f"  Viability  {r['note']}",
        ]
    return "\n".join(lines)
