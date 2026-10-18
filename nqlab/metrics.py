"""Performance metrics from a simulation result.

What each number can and cannot tell you is documented in the report
(section 6); the short version is in the comments here. Sharpe/Sortino/Calmar
are computed from DAILY P&L over every trading day in the window, including
days with no trade — computing them per trade, or only on active days,
overstates them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Result

TRADING_DAYS = 252


def _streaks(x: np.ndarray) -> tuple[int, int]:
    best_w = best_l = cur = 0
    for v in x:
        if v > 0:
            cur = cur + 1 if cur > 0 else 1
        elif v < 0:
            cur = cur - 1 if cur < 0 else -1
        else:
            cur = 0
        best_w, best_l = max(best_w, cur), max(best_l, -cur)
    return best_w, best_l


def _max_dd(equity: np.ndarray) -> tuple[float, float]:
    """(max drawdown in $, as a fraction of the running peak)."""
    if len(equity) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    i = int(np.argmax(dd))
    return float(dd[i]), float(dd[i] / peak[i]) if peak[i] > 0 else 0.0


def trade_stats(tr: pd.DataFrame) -> dict:
    n = len(tr)
    if n == 0:
        return {"trades": 0, "exp_r": np.nan, "win_rate": np.nan, "pf": np.nan}
    r = tr["r"].to_numpy()
    pnl = tr["pnl"].to_numpy()
    wins, losses = pnl > 0, pnl < 0
    gross_w, gross_l = pnl[wins].sum(), -pnl[losses].sum()
    sw, sl = _streaks(pnl)
    return {
        "trades": n,
        "exp_r": float(r.mean()),                    # expectancy, R per trade after costs
        "exp_usd": float(pnl.mean()),
        "win_rate": float(wins.mean()),
        "avg_win_r": float(r[wins].mean()) if wins.any() else 0.0,
        "avg_loss_r": float(r[losses].mean()) if losses.any() else 0.0,
        "avg_win_usd": float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loss_usd": float(pnl[losses].mean()) if losses.any() else 0.0,
        "pf": float(gross_w / gross_l) if gross_l > 0 else np.inf,
        "total_r": float(r.sum()),
        "gross_r": float(((tr["pnl"] + tr["cost"]) / tr["risk"]).mean()),   # before all costs
        "cost_r": float((tr["cost"] / tr["risk"]).mean()),
        "avg_minutes": float(tr["minutes"].mean()),
        "max_consec_wins": sw,
        "max_consec_losses": sl,
        "target_rate": float((tr["reason"] == "target").mean()),
        "stop_rate": float(tr["reason"].isin(["stop", "gap_stop"]).mean()),
    }


def summarize(res: Result) -> dict:
    tr = res.trades
    eq0 = res.account.equity
    out = trade_stats(tr)
    daily = res.daily_pnl()
    days = len(daily)
    if res.fractional:
        eq_close = eq0 + np.cumsum(tr["pnl"].to_numpy()) if len(tr) else np.array([eq0])
        base = np.full(days, eq0)
    else:
        eq_close = tr["equity"].to_numpy() if len(tr) else np.array([eq0])
        base = eq0 + np.r_[0.0, np.cumsum(daily.to_numpy())[:-1]]
    ret = daily.to_numpy() / np.where(base > 0, base, np.nan)
    mu, sd = np.nanmean(ret) if days else 0.0, np.nanstd(ret, ddof=1) if days > 1 else np.nan
    down = np.sqrt(np.nanmean(np.minimum(ret, 0.0) ** 2)) if days else np.nan
    dd_usd, dd_pct = _max_dd(np.r_[eq0, eq_close])
    net = float(tr["pnl"].sum()) if len(tr) else 0.0
    years = days / TRADING_DAYS if days else np.nan
    cagr = (1 + net / eq0) ** (1 / years) - 1 if years and net > -eq0 else np.nan
    out.update({
        "days": days,
        "net_usd": net,
        "return_pct": 100 * net / eq0,
        "sharpe": float(mu / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else np.nan,
        "sortino": float(mu / down * np.sqrt(TRADING_DAYS)) if down and down > 0 else np.nan,
        "max_dd_usd": dd_usd,
        "max_dd_pct": 100 * dd_pct,
        "calmar": float(cagr / dd_pct) if dd_pct > 0 and np.isfinite(cagr) else np.nan,
        "trades_per_day": (out["trades"] / days) if days else 0.0,
        **{f"n_{k}": v for k, v in res.counts.items()},
    })
    return out


def breakdown(tr: pd.DataFrame, by: str, names: dict | None = None) -> pd.DataFrame:
    """Expectancy, win rate and count per group (session, side, regime ...)."""
    if len(tr) == 0:
        return pd.DataFrame()
    g = tr.groupby(by)
    df = pd.DataFrame({"trades": g.size(), "exp_r": g["r"].mean(), "win_rate": g["pnl"].apply(lambda x: (x > 0).mean()),
                       "total_r": g["r"].sum(), "pnl_usd": g["pnl"].sum()})
    if names:
        df.index = [names.get(i, i) for i in df.index]
    return df


def long_short(tr: pd.DataFrame) -> pd.DataFrame:
    return breakdown(tr, "side", {1: "long", -1: "short"})
