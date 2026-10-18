"""Statistical validation: bootstrap, Monte Carlo, multiple-testing correction.

Trades on the same day share the same market, so they are not independent.
Every resampling here draws whole DAYS (block bootstrap); resampling individual
trades would understate the uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def day_block_bootstrap(r: np.ndarray, day: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap the mean R per trade by resampling trading days with replacement.

    Returns the point estimate, a 95% percentile interval, and the one-sided
    p-value of 'mean <= 0' (fraction of resamples at or below zero).
    """
    r = np.asarray(r, float)
    if len(r) < 2:
        return {"mean": float(r.mean()) if len(r) else np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": len(r)}
    days, inv = np.unique(day, return_inverse=True)
    sums = np.bincount(inv, weights=r)
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(days), size=(n_boot, len(days)))
    means = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return {"mean": float(r.mean()), "lo": float(np.quantile(means, 0.025)),
            "hi": float(np.quantile(means, 0.975)), "p": float((means <= 0).mean()), "n": int(len(r))}


def monte_carlo(r: np.ndarray, day: np.ndarray, risk_pct: float, n_paths: int = 5000,
                horizon_days: int = 252, seed: int = 0) -> dict:
    """Simulate `horizon_days` of trading by drawing whole historical days.

    Equity compounds at risk_pct per 1R. Reports the distribution of max
    drawdown, final return, the longest losing streak, and the probability of
    drawdowns that typically end a retail account or a prop evaluation.
    """
    r = np.asarray(r, float)
    days, inv = np.unique(day, return_inverse=True)
    per_day = [r[inv == k] for k in range(len(days))]
    rng = np.random.default_rng(seed)
    dds, finals, streaks = np.empty(n_paths), np.empty(n_paths), np.empty(n_paths)
    for p in range(n_paths):
        seq = np.concatenate([per_day[k] for k in rng.integers(0, len(days), horizon_days)])
        eq = np.cumprod(1.0 + risk_pct * seq)
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        dds[p] = np.max(1 - np.r_[1.0, eq] / peak)
        finals[p] = eq[-1] - 1 if len(eq) else 0.0
        best = cur = 0
        for v in seq:
            cur = cur + 1 if v < 0 else 0
            best = max(best, cur)
        streaks[p] = best
    q = lambda x, a: float(np.quantile(x, a))  # noqa: E731
    return {
        "dd_median_pct": 100 * q(dds, 0.5), "dd_p95_pct": 100 * q(dds, 0.95), "dd_p99_pct": 100 * q(dds, 0.99),
        "p_dd_gt_10pct": float((dds > 0.10).mean()), "p_dd_gt_20pct": float((dds > 0.20).mean()),
        "ret_median_pct": 100 * q(finals, 0.5), "ret_p05_pct": 100 * q(finals, 0.05),
        "ret_p95_pct": 100 * q(finals, 0.95), "p_loss": float((finals < 0).mean()),
        "losing_streak_median": q(streaks, 0.5), "losing_streak_p95": q(streaks, 0.95),
        "paths": n_paths, "horizon_days": horizon_days,
    }


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_ref: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_ref) given n observations (Bailey & Lopez de Prado)."""
    if n < 3 or not np.isfinite(sr):
        return np.nan
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_ref) * np.sqrt(n - 1) / den))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum of n_trials Sharpe estimates under the null of zero skill."""
    if n_trials < 2:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - g) * z1 + g * z2))


def deflated_sharpe(returns: np.ndarray, n_trials: int, var_sr_trials: float) -> dict:
    """Deflated Sharpe Ratio: PSR against the best Sharpe expected from luck alone.

    `returns` is the selected strategy's per-period (daily) series; var_sr_trials
    is the cross-sectional variance of per-period Sharpe across ALL configurations
    tried. DSR > 0.95 is the conventional bar.
    """
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    if len(x) < 10 or x.std(ddof=1) == 0:
        return {"sr": np.nan, "sr0": np.nan, "dsr": np.nan}
    sr = x.mean() / x.std(ddof=1)
    sk, ku = float(stats.skew(x)), float(stats.kurtosis(x, fisher=False))
    sr0 = expected_max_sharpe(var_sr_trials, n_trials)
    # Alternative benchmark: every trial is pure noise, so each Sharpe estimate has
    # sampling variance ~1/T. The empirical cross-trial variance above also contains
    # real, KNOWN differences between trials (e.g. cost-dominated 1-minute configs),
    # which inflates the hurdle; both are reported.
    sr0_null = expected_max_sharpe(1.0 / len(x), n_trials)
    return {"sr": float(sr), "sr_ann": float(sr * np.sqrt(252)), "sr0": sr0, "sr0_ann": float(sr0 * np.sqrt(252)),
            "dsr": probabilistic_sharpe(sr, len(x), sk, ku, sr0),
            "sr0_null_ann": float(sr0_null * np.sqrt(252)),
            "dsr_null": probabilistic_sharpe(sr, len(x), sk, ku, sr0_null),
            "n_trials": n_trials}


def cluster_mean_test(x: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean of x with a cluster-robust (by day) standard error and t-statistic."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, cluster = x[ok], cluster[ok]
    n = len(x)
    if n < 10:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": n}
    mu = x.mean()
    _, inv = np.unique(cluster, return_inverse=True)
    g = np.bincount(inv, weights=x - mu)
    G = len(g)
    se = np.sqrt((g ** 2).sum() * G / max(G - 1, 1)) / n
    t = mu / se if se > 0 else np.nan
    return {"mean": float(mu), "se": float(se), "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), max(G - 1, 1))) if np.isfinite(t) else np.nan, "n": n, "clusters": G}


def neighbours(grid_point: dict, grid: dict) -> list[dict]:
    """Grid points that differ from `grid_point` by one step in exactly one parameter."""
    out = []
    for k, vals in grid.items():
        vals = list(vals)
        i = vals.index(grid_point[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**grid_point, k: vals[j]})
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family of hypothesis tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)
