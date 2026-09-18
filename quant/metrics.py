"""Performance and risk statistics computed from an equity curve + trade log."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """Peak-to-trough episodes, deepest first, with recovery times."""
    dd = drawdown_series(equity)
    in_dd = dd < 0
    episodes = []
    start = None
    for ts, flag in in_dd.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            seg = dd.loc[start:ts]
            episodes.append((start, seg.idxmin(), ts, float(seg.min())))
            start = None
    if start is not None:                      # still under water at the end
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), pd.NaT, float(seg.min())))

    rows = []
    for s, trough, rec, depth in episodes:
        rows.append(
            {
                "start": s,
                "trough": trough,
                "recovered": rec,
                "depth_pct": depth * 100,
                "to_trough_bars": int(equity.index.get_indexer([trough])[0] - equity.index.get_indexer([s])[0]),
                "recovery_bars": (
                    int(equity.index.get_indexer([rec])[0] - equity.index.get_indexer([trough])[0])
                    if pd.notna(rec) else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("depth_pct").head(top).reset_index(drop=True)


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown_series(equity) * 100
    return float(np.sqrt((dd ** 2).mean()))


def cagr(equity: pd.Series, bars_per_year: int) -> float:
    years = len(equity) / bars_per_year
    if years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(equity: pd.Series, bars_per_year: int, rf_annual: float = 0.0) -> float:
    r = equity.pct_change().dropna()
    if r.std(ddof=0) == 0 or r.empty:
        return float("nan")
    excess = r - rf_annual / bars_per_year
    return float(excess.mean() / r.std(ddof=0) * np.sqrt(bars_per_year))


def sortino(equity: pd.Series, bars_per_year: int) -> float:
    r = equity.pct_change().dropna()
    downside = r[r < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        return float("nan")
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(bars_per_year))


def summarize(result, bars_per_year: int = 8760, label: str = "strategy") -> dict:
    """One row of headline statistics for a BacktestResult."""
    eq = result.equity
    tr = result.trades
    mdd = max_drawdown(eq)
    cg = cagr(eq, bars_per_year)
    wins = tr[tr["pnl"] > 0] if not tr.empty else tr
    losses = tr[tr["pnl"] <= 0] if not tr.empty else tr
    gross_win = float(wins["pnl"].sum()) if not tr.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not tr.empty else 0.0
    n = len(tr)
    years = len(eq) / bars_per_year

    return {
        "strategy": label,
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
        "CAGR_pct": cg * 100 if np.isfinite(cg) else np.nan,
        "Sharpe": sharpe(eq, bars_per_year),
        "Sortino": sortino(eq, bars_per_year),
        "max_drawdown_pct": mdd * 100,
        "Calmar": (cg / abs(mdd)) if mdd < 0 and np.isfinite(cg) else np.nan,
        "ulcer_index": ulcer_index(eq),
        "trades": n,
        "trades_per_year": n / years if years > 0 else np.nan,
        "win_rate_pct": (len(wins) / n * 100) if n else np.nan,
        "avg_R": float(tr["r_multiple"].mean()) if n else np.nan,
        "expectancy_R": float(tr["r_multiple"].mean()) if n else np.nan,
        "avg_win_R": float(wins["r_multiple"].mean()) if len(wins) else np.nan,
        "avg_loss_R": float(losses["r_multiple"].mean()) if len(losses) else np.nan,
        "payoff_ratio": (
            float(wins["r_multiple"].mean() / abs(losses["r_multiple"].mean()))
            if len(wins) and len(losses) and losses["r_multiple"].mean() != 0 else np.nan
        ),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.nan,
        "total_costs": float(tr["costs"].sum()) if n else 0.0,
        "cost_drag_pct_of_start": (float(tr["costs"].sum()) / eq.iloc[0] * 100) if n else 0.0,
        "rejected_orders": result.rejected_orders,
    }
