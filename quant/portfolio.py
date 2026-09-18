"""Portfolio construction for a small multi-asset book (BTC + Nasdaq + cash).

Everything is driven by an explicit `AssetView`. Change the views and the
allocation changes — nothing here is hard-coded to a market call. Expected
returns are the weakest input in any allocator, so the defaults lean on
volatility and correlation (which are far more stable) and the risk-tolerance
presets cap crypto exposure directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AssetView:
    name: str
    exp_return: float      # expected ANNUAL total return, decimal
    vol: float             # expected ANNUAL volatility, decimal
    max_dd_hist: float     # a realistic historical peak-to-trough, decimal (negative)
    rationale: str = ""


RISK_PRESETS = {
    # cap on the crypto sleeve, portfolio vol target, cash floor
    "low":    {"crypto_cap": 0.05, "vol_target": 0.08, "cash_floor": 0.30},
    "medium": {"crypto_cap": 0.15, "vol_target": 0.13, "cash_floor": 0.10},
    "high":   {"crypto_cap": 0.35, "vol_target": 0.22, "cash_floor": 0.00},
}


def cov_matrix(views: list[AssetView], corr: np.ndarray) -> np.ndarray:
    vols = np.array([v.vol for v in views])
    return np.outer(vols, vols) * corr


def port_stats(w: np.ndarray, views: list[AssetView], corr: np.ndarray,
               rf: float = 0.04) -> dict:
    mu = np.array([v.exp_return for v in views])
    cov = cov_matrix(views, corr)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return {"exp_return": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol > 0 else np.nan}


def _simplex(n: int, step: float = 0.01):
    """All long-only weight vectors on a grid. Fine for n<=4, exact enough here."""
    k = int(round(1 / step))
    if n == 1:
        yield np.array([1.0]); return
    for combo in _compositions(k, n):
        yield np.array(combo) / k


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


def optimise(views: list[AssetView], corr: np.ndarray, objective: str = "max_sharpe",
             rf: float = 0.04, caps: dict | None = None, step: float = 0.01) -> np.ndarray:
    """Long-only optimisation over a weight grid, honouring per-asset caps."""
    caps = caps or {}
    best_w, best_score = None, -np.inf
    for w in _simplex(len(views), step):
        if any(w[i] > caps.get(v.name, 1.0) + 1e-9 for i, v in enumerate(views)):
            continue
        s = port_stats(w, views, corr, rf)
        if objective == "max_sharpe":
            score = s["sharpe"]
        elif objective == "min_vol":
            score = -s["vol"]
        else:
            raise ValueError(objective)
        if np.isfinite(score) and score > best_score:
            best_w, best_score = w, score
    return best_w if best_w is not None else np.ones(len(views)) / len(views)


def inverse_vol(views: list[AssetView]) -> np.ndarray:
    iv = np.array([1.0 / v.vol for v in views])
    return iv / iv.sum()


def simulate_drawdown(w: np.ndarray, views: list[AssetView], corr: np.ndarray,
                      years: float = 2.0, n_sims: int = 10_000, seed: int = 11,
                      df_t: int = 4) -> dict:
    """Fat-tailed (Student-t) Monte Carlo of the portfolio path.

    Gaussian simulation understates crypto drawdowns badly; t(4) shocks are a
    much closer match to what BTC actually does.
    """
    rng = np.random.default_rng(seed)
    n_days = int(252 * years)
    mu = np.array([v.exp_return for v in views]) / 252
    cov = cov_matrix(views, corr) / 252
    L = np.linalg.cholesky(cov + np.eye(len(views)) * 1e-12)

    z = rng.standard_t(df_t, size=(n_sims, n_days, len(views))) / np.sqrt(df_t / (df_t - 2))
    shocks = z @ L.T + mu
    port_r = shocks @ w
    # A long-only book cannot lose more than 100% in a period; without this clip
    # a fat-tailed draw can push the cumulative product negative and report
    # impossible sub -100% drawdowns.
    growth = np.maximum(1.0 + port_r, 0.0)
    paths = np.cumprod(growth, axis=1)
    peak = np.maximum.accumulate(paths, axis=1)
    dd = (paths / peak - 1.0).min(axis=1)
    final = paths[:, -1] - 1.0
    return {
        "exp_total_return_p50": float(np.percentile(final, 50)) * 100,
        "total_return_p5": float(np.percentile(final, 5)) * 100,
        "total_return_p95": float(np.percentile(final, 95)) * 100,
        "prob_negative": float((final < 0).mean()) * 100,
        "maxdd_median": float(np.percentile(dd, 50)) * 100,
        "maxdd_p95": float(np.percentile(dd, 5)) * 100,
        "maxdd_worst": float(dd.min()) * 100,
    }


def build(views: list[AssetView], corr: np.ndarray, risk: str = "medium",
          horizon_years: float = 2.0, rf: float = 0.04) -> dict:
    preset = RISK_PRESETS[risk]
    caps = {v.name: (preset["crypto_cap"] if "BTC" in v.name.upper() else 1.0) for v in views}
    caps["Cash / T-bills"] = 1.0

    w = optimise(views, corr, "max_sharpe", rf, caps)
    # Enforce the cash floor, scaling the risk assets down proportionally.
    names = [v.name for v in views]
    if "Cash / T-bills" in names:
        ci = names.index("Cash / T-bills")
        if w[ci] < preset["cash_floor"]:
            deficit = preset["cash_floor"] - w[ci]
            risky = np.array([i != ci for i in range(len(w))])
            if w[risky].sum() > 0:
                w[risky] *= (w[risky].sum() - deficit) / w[risky].sum()
            w[ci] = preset["cash_floor"]
    w = w / w.sum()

    stats = port_stats(w, views, corr, rf)
    # Volatility-target overlay: lever down (never up) toward the preset.
    scale = min(1.0, preset["vol_target"] / stats["vol"]) if stats["vol"] > 0 else 1.0

    sim = simulate_drawdown(w, views, corr, years=horizon_years)
    table = pd.DataFrame({
        "asset": names,
        "weight_pct": (w * 100).round(1),
        "exp_return_pct": [v.exp_return * 100 for v in views],
        "vol_pct": [v.vol * 100 for v in views],
        "hist_maxdd_pct": [v.max_dd_hist * 100 for v in views],
        "why": [v.rationale for v in views],
    })
    return {
        "risk_profile": risk, "weights": w, "table": table, "stats": stats,
        "vol_target": preset["vol_target"], "suggested_scale": scale, "sim": sim,
        "horizon_years": horizon_years,
    }
