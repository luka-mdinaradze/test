"""Monte Carlo robustness testing.

Two entry points:
  * `bootstrap_r` resamples an ACTUAL trade log's R-multiples.
  * `parametric_r` builds the R distribution from stated assumptions
    (win rate, average win in R, average loss in R) — useful before you have a
    trade log, and honest as long as the assumptions are labelled as such.

Both compound fractional risk the way the live system does: each trade risks
`risk_pct` of CURRENT equity, so the paths capture the real interaction between
drawdown and position size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parametric_r(win_rate: float, win_R: float, loss_R: float, size: int,
                 rng: np.random.Generator) -> np.ndarray:
    wins = rng.random(size) < win_rate
    return np.where(wins, win_R, -abs(loss_R))


def simulate(
    r_samples: np.ndarray,
    n_trades: int = 300,
    n_sims: int = 20_000,
    risk_pct: float = 0.01,
    initial_equity: float = 100.0,
    ruin_level: float = 0.5,
    seed: int = 42,
) -> dict:
    """Resample trade outcomes with replacement and compound them.

    Returns final-equity distribution, drawdown distribution, probability of
    loss, risk of ruin, and worst-case paths.
    """
    rng = np.random.default_rng(seed)
    r_samples = np.asarray(r_samples, dtype=float)
    r_samples = r_samples[np.isfinite(r_samples)]
    if r_samples.size == 0:
        raise ValueError("no finite R-multiples to resample")

    draws = rng.choice(r_samples, size=(n_sims, n_trades), replace=True)
    # Equity path: E_{k+1} = E_k * (1 + risk_pct * R_k)
    growth = 1.0 + risk_pct * draws
    growth = np.maximum(growth, 0.0)                  # a -100R tail cannot go below zero
    paths = initial_equity * np.cumprod(growth, axis=1)
    paths = np.concatenate([np.full((n_sims, 1), initial_equity), paths], axis=1)

    running_max = np.maximum.accumulate(paths, axis=1)
    dd = paths / running_max - 1.0
    max_dd = dd.min(axis=1)
    final = paths[:, -1]

    # Longest losing streak per simulation
    losing = draws < 0
    streaks = np.zeros(n_sims, dtype=int)
    cur = np.zeros(n_sims, dtype=int)
    for k in range(n_trades):
        cur = np.where(losing[:, k], cur + 1, 0)
        streaks = np.maximum(streaks, cur)

    pct = lambda a, q: float(np.percentile(a, q))     # noqa: E731
    return {
        "n_sims": n_sims,
        "n_trades": n_trades,
        "risk_pct": risk_pct,
        "expectancy_R": float(r_samples.mean()),
        "prob_loss": float((final < initial_equity).mean()),
        "prob_ruin": float((paths.min(axis=1) <= initial_equity * ruin_level).mean()),
        "ruin_level_pct": ruin_level * 100,
        "final_p5": pct(final, 5), "final_p25": pct(final, 25), "final_p50": pct(final, 50),
        "final_p75": pct(final, 75), "final_p95": pct(final, 95),
        "final_mean": float(final.mean()),
        "final_worst": float(final.min()), "final_best": float(final.max()),
        "maxdd_p50": pct(max_dd, 50) * 100, "maxdd_p95": pct(max_dd, 5) * 100,
        "maxdd_p99": pct(max_dd, 1) * 100, "maxdd_worst": float(max_dd.min()) * 100,
        "streak_p50": pct(streaks, 50), "streak_p95": pct(streaks, 95),
        "streak_worst": int(streaks.max()),
        "_paths": paths,
    }


def summary_frame(res: dict) -> pd.DataFrame:
    rows = [
        ("Probability of ending below start", f"{res['prob_loss'] * 100:.1f}%"),
        (f"Probability of ≥{100 - res['ruin_level_pct']:.0f}% loss (ruin)", f"{res['prob_ruin'] * 100:.2f}%"),
        ("Median final equity", f"${res['final_p50']:.2f}"),
        ("5th percentile final equity", f"${res['final_p5']:.2f}"),
        ("95th percentile final equity", f"${res['final_p95']:.2f}"),
        ("Worst simulated path", f"${res['final_worst']:.2f}"),
        ("Median max drawdown", f"{res['maxdd_p50']:.1f}%"),
        ("95th-percentile max drawdown", f"{res['maxdd_p95']:.1f}%"),
        ("Worst max drawdown", f"{res['maxdd_worst']:.1f}%"),
        ("Median longest losing streak", f"{res['streak_p50']:.0f} trades"),
        ("95th-percentile losing streak", f"{res['streak_p95']:.0f} trades"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def robustness_verdict(res: dict) -> str:
    """A blunt read on fragility, from the simulated distribution."""
    if res["expectancy_R"] <= 0:
        return ("FRAGILE - negative expectancy. No position-sizing scheme rescues this; "
                "the only fix is a better signal or lower costs.")
    if res["prob_ruin"] > 0.05:
        return (f"FRAGILE - {res['prob_ruin'] * 100:.1f}% of paths lose half the account. "
                "Cut risk per trade or reduce trade frequency.")
    if res["prob_loss"] > 0.35:
        return (f"BORDERLINE - {res['prob_loss'] * 100:.0f}% of paths end below the starting "
                "balance despite positive expectancy. The edge is real but thin relative to variance.")
    return (f"ROBUST under the sampled distribution - {res['prob_loss'] * 100:.0f}% of paths end "
            f"below start and worst-case drawdown is {res['maxdd_worst']:.0f}%.")
