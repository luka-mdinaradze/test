#!/usr/bin/env python3
"""New York 30-minute opening-range breakout on NDX / NQ.

Rules (one trade per day, the first breakout):
  1. Opening range = high/low of 09:30-10:00 ET.
  2. From 10:00 on, watch 5-minute candles. The first one that CLOSES above
     the range high goes long; the first that closes below the range low
     goes short. Entry is the next 5m candle's open (the breakout close is
     only known once that candle is done).
  3. Stop = just beyond the breakout candle: its low minus `buffer` for a
     long, its high plus `buffer` for a short.
  4. Target = 1.5 x the entry-to-stop distance.
  5. Anything still open at 16:00 ET is closed at the last price.

Risk is a fixed 1% of the starting $10,000 ($100) per trade, sized
fractionally (CFD-style). If a bar touches both stop and target, the stop is
assumed first. With 1-minute input the fills are walked minute by minute,
which resolves most of those ambiguous bars.

    python3 orb/ndx_orb.py --csv nq_1m.csv --out results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"


def load_csv(path: str) -> pd.DataFrame:
    """Read an OHLCV CSV (Yahoo, TradingView or Databento ohlcv-1m) into ET-indexed bars."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = next(c for c in ("ts_event", "datetime", "timestamp", "time", "date") if c in df.columns)
    ts = df[tcol]
    if np.issubdtype(ts.dtype, np.number):
        unit = "ns" if ts.iloc[0] > 1e17 else "s"
        ts = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    df.index = ts.dt.tz_convert(TZ)
    if "symbol" in df.columns:
        df = front_month(df)
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Databento GLBX mixes contracts and spreads: keep the busiest outright each day."""
    df = df[~df["symbol"].astype(str).str.contains("-")]
    day = df.index.date
    vol = df.groupby([day, df["symbol"]])["volume"].sum()
    best = vol.groupby(level=0).idxmax().map(lambda k: k[1])
    return df[df["symbol"].values == best.reindex(day).values]


def backtest(bars: pd.DataFrame, rr: float = 1.5, buffer: float = 2.0,
             risk_usd: float = 100.0, start_equity: float = 10_000.0,
             cost_pts: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    rth = bars.between_time("09:30", "15:59")
    trades = []
    for day, d in rth.groupby(rth.index.date):
        orng = d.between_time("09:30", "09:59")
        if len(orng) == 0 or d.index[-1].time() < pd.Timestamp("15:00").time():
            continue  # half days / holes
        hi, lo = orng["high"].max(), orng["low"].min()
        after = d[d.index >= d.index[0].normalize() + pd.Timedelta(hours=10)]
        m5 = after.resample("5min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        sig = m5[(m5["close"] > hi) | (m5["close"] < lo)]
        if sig.empty:
            continue
        t0, c = sig.index[0], sig.iloc[0]
        side = 1 if c["close"] > hi else -1
        rest = after[after.index >= t0 + pd.Timedelta(minutes=5)]
        if rest.empty:
            continue
        entry = rest["open"].iloc[0]
        stop = c["low"] - buffer if side == 1 else c["high"] + buffer
        risk = (entry - stop) * side
        if risk <= 0:  # opened past the stop already
            continue
        target = entry + side * rr * risk
        exit_px, exit_t, why = rest["close"].iloc[-1], rest.index[-1], "16:00"
        for t, b in rest.iterrows():
            hit_stop = b["low"] <= stop if side == 1 else b["high"] >= stop
            hit_tp = b["high"] >= target if side == 1 else b["low"] <= target
            if hit_stop:  # gap through the stop fills at the open
                exit_px = min(b["open"], stop) if side == 1 else max(b["open"], stop)
                exit_t, why = t, "stop"
                break
            if hit_tp:
                exit_px, exit_t, why = target, t, "target"
                break
        pts = (exit_px - entry) * side - cost_pts
        r = pts / risk
        trades.append(dict(date=pd.Timestamp(day), side="long" if side == 1 else "short",
                           or_high=hi, or_low=lo, signal_time=t0.strftime("%H:%M"),
                           entry=entry, stop=stop, target=target, exit=exit_px,
                           exit_time=exit_t.strftime("%H:%M"), exit_reason=why,
                           risk_pts=risk, r=r, pnl_usd=r * risk_usd))
    tr = pd.DataFrame(trades)
    eq = start_equity + tr["pnl_usd"].cumsum() if len(tr) else pd.Series(dtype=float)
    if len(tr):
        eq.index = tr["date"]
    return tr, eq


def summary(tr: pd.DataFrame, eq: pd.Series, start_equity: float = 10_000.0) -> dict:
    if tr.empty:
        return {"trades": 0}
    curve = pd.concat([pd.Series([start_equity]), eq.reset_index(drop=True)])
    dd = curve - curve.cummax()
    i = int(dd.idxmin()) if len(dd) else 0
    wins, losses = tr[tr.pnl_usd > 0], tr[tr.pnl_usd <= 0]
    streak = (tr.pnl_usd <= 0).astype(int)
    max_loss_streak = int(streak.groupby((streak != streak.shift()).cumsum()).sum().max())
    return {
        "first_day": str(tr.date.iloc[0].date()), "last_day": str(tr.date.iloc[-1].date()),
        "trades": len(tr), "longs": int((tr.side == "long").sum()), "shorts": int((tr.side == "short").sum()),
        "win_rate_pct": round(100 * len(wins) / len(tr), 1),
        "targets": int((tr.exit_reason == "target").sum()), "stops": int((tr.exit_reason == "stop").sum()),
        "closed_at_1600": int((tr.exit_reason == "16:00").sum()),
        "total_pnl_usd": round(tr.pnl_usd.sum(), 2),
        "return_pct": round(100 * tr.pnl_usd.sum() / start_equity, 2),
        "avg_r": round(tr.r.mean(), 3),
        "profit_factor": round(wins.pnl_usd.sum() / -losses.pnl_usd.sum(), 2) if losses.pnl_usd.sum() < 0 else None,
        "max_drawdown_usd": round(-dd.min(), 2),
        "max_drawdown_pct": round(100 * (-dd.min()) / curve.cummax().iloc[i], 2),
        "max_losing_streak": max_loss_streak,
        "final_equity": round(curve.iloc[-1], 2),
        "by_year": {str(y): round(g.pnl_usd.sum(), 2) for y, g in tr.groupby(tr.date.dt.year)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--buffer", type=float, default=2.0, help="points beyond the breakout candle for the stop")
    ap.add_argument("--risk", type=float, default=100.0)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--cost", type=float, default=0.0, help="round-trip cost in index points")
    ap.add_argument("--out", default="orb_results")
    a = ap.parse_args()
    tr, eq = backtest(load_csv(a.csv), a.rr, a.buffer, a.risk, a.equity, a.cost)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out / "trades.csv", index=False)
    s = summary(tr, eq, a.equity)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
