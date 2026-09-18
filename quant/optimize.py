"""Walk-forward parameter optimisation.

An in-sample grid search on 6 years of 1h bars will always produce a beautiful
equity curve. It means nothing. The only number worth reading is the stitched
OUT-OF-SAMPLE curve produced here: parameters are chosen on each in-sample
window and then traded blind on the window that follows.

`grid_search` (pure in-sample) is provided only so you can SEE the gap between
the two, which is the real output of this module.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .engine import Costs, ExecConfig, run_backtest
from .metrics import cagr, max_drawdown, sharpe, summarize


def _objective(result, bars_per_year: int, kind: str) -> float:
    eq = result.equity
    if len(result.trades) < 10:            # not enough evidence to rank
        return -np.inf
    if kind == "sharpe":
        v = sharpe(eq, bars_per_year)
    elif kind == "calmar":
        mdd = max_drawdown(eq)
        v = cagr(eq, bars_per_year) / abs(mdd) if mdd < 0 else -np.inf
    elif kind == "sharpe_dd":              # Sharpe penalised by drawdown depth
        v = sharpe(eq, bars_per_year) * (1.0 + max_drawdown(eq))
    else:
        raise ValueError(kind)
    return v if np.isfinite(v) else -np.inf


def param_grid(base, **axes: Iterable) -> list:
    """Cartesian product over dataclass fields: param_grid(S1Params(), stop_atr=[1.5,2.0])."""
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        out.append(replace(base, **dict(zip(keys, combo))))
    return out


def evaluate(df: pd.DataFrame, builder: Callable, params, costs: Costs, cfg: ExecConfig):
    orders, atr_s, _ = builder(df, params)
    return run_backtest(df, orders, atr_s, costs, cfg)


def grid_search(df, builder, candidates, costs=Costs(), cfg=ExecConfig(),
                objective="sharpe") -> pd.DataFrame:
    rows = []
    for p in candidates:
        res = evaluate(df, builder, p, costs, cfg)
        s = summarize(res, cfg.bars_per_year, label=str(p))
        s["objective"] = _objective(res, cfg.bars_per_year, objective)
        s["params"] = p
        rows.append(s)
    return pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)


def walk_forward(
    df: pd.DataFrame,
    builder: Callable,
    candidates: list,
    n_folds: int = 6,
    is_frac: float = 0.7,
    costs: Costs = Costs(),
    cfg: ExecConfig = ExecConfig(),
    objective: str = "sharpe_dd",
    warmup: int = 300,
) -> dict:
    """Anchored-walk-forward: each fold trains on its own slice, trades the next.

    Returns the stitched OOS equity curve, the per-fold chosen parameters, and
    the in-sample vs out-of-sample gap (the overfitting tax).
    """
    n = len(df)
    fold_len = n // n_folds
    if fold_len < warmup * 2:
        raise ValueError(f"not enough bars: {n} rows / {n_folds} folds is too thin")

    oos_curves, fold_rows = [], []
    equity = cfg.initial_equity

    for k in range(n_folds):
        lo, hi = k * fold_len, (k + 1) * fold_len if k < n_folds - 1 else n
        split = lo + int((hi - lo) * is_frac)
        is_df, oos_df = df.iloc[lo:split], df.iloc[max(split - warmup, lo):hi]
        if len(is_df) < warmup * 2 or len(oos_df) < warmup:
            continue

        best, best_obj, best_is = None, -np.inf, None
        for p in candidates:
            res = evaluate(is_df, builder, p, costs, cfg)
            ob = _objective(res, cfg.bars_per_year, objective)
            if ob > best_obj:
                best, best_obj, best_is = p, ob, res
        if best is None:
            continue

        oos_cfg = replace(cfg, initial_equity=equity)      # compound across folds
        oos_res = evaluate(oos_df, builder, best, costs, oos_cfg)
        oos_eq = oos_res.equity.iloc[warmup:] if len(oos_res.equity) > warmup else oos_res.equity
        if oos_eq.empty:
            continue
        equity = float(oos_eq.iloc[-1])
        oos_curves.append(oos_eq)

        fold_rows.append({
            "fold": k + 1,
            "is_start": is_df.index[0], "is_end": is_df.index[-1],
            "oos_start": oos_eq.index[0], "oos_end": oos_eq.index[-1],
            "is_sharpe": sharpe(best_is.equity, cfg.bars_per_year),
            "oos_sharpe": sharpe(oos_eq, cfg.bars_per_year),
            "oos_return_pct": float(oos_eq.iloc[-1] / oos_eq.iloc[0] - 1) * 100,
            "oos_maxdd_pct": max_drawdown(oos_eq) * 100,
            "oos_trades": len(oos_res.trades),
            "chosen": best,
        })

    if not oos_curves:
        return {"folds": pd.DataFrame(), "oos_equity": pd.Series(dtype=float), "gap": np.nan}

    stitched = pd.concat(oos_curves)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    folds = pd.DataFrame(fold_rows)
    return {
        "folds": folds,
        "oos_equity": stitched,
        "mean_is_sharpe": float(folds["is_sharpe"].mean()),
        "mean_oos_sharpe": float(folds["oos_sharpe"].mean()),
        "gap": float(folds["is_sharpe"].mean() - folds["oos_sharpe"].mean()),
    }
