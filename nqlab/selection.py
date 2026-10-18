"""Parameter selection that does not reward luck.

Rules (pre-registered in PREREGISTRATION.md):

score       mean R minus 1.96 day-clustered standard errors: the lower edge of a
            95% interval. Ranks by what we can be confident of, so a config with
            a high mean from 25 trades does not beat a steady one with 400.
eligible    train mean > 0 AND validation mean > 0 AND enough validation trades
            AND a PLATEAU: at least 60% of the config's grid neighbours (one step
            in one parameter) are also positive on validation. An isolated peak
            surrounded by losers is an artefact of noise, not a parameter value.
selected    the eligible config with the best VALIDATION score — chosen on data
            the rule did not fit, never on the highest training profit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import cluster_mean_test, neighbours

MIN_VAL_TRADES = 30
MIN_IS_TRADES = 50
PLATEAU_SHARE = 0.60


def score(r: np.ndarray, day: np.ndarray) -> dict:
    t = cluster_mean_test(np.asarray(r, float), np.asarray(day))
    lo = t["mean"] - 1.96 * t["se"] if np.isfinite(t.get("se", np.nan)) else np.nan
    return {"n": t["n"], "mean": t["mean"], "se": t.get("se", np.nan), "t": t.get("t", np.nan), "score": lo}


def stats_for(tr: pd.DataFrame, lo: int, hi: int) -> dict:
    s = tr[(tr["exit_tday"] >= lo) & (tr["exit_tday"] <= hi)]
    if len(s) < 3:
        return {"n": len(s), "mean": np.nan, "se": np.nan, "t": np.nan, "score": np.nan,
                "total_r": float(s["r"].sum()) if len(s) else 0.0, "win": np.nan, "pf": np.nan}
    sc = score(s["r"].to_numpy(), s["exit_tday"].to_numpy())
    w = s.loc[s["pnl"] > 0, "pnl"].sum()
    l_ = -s.loc[s["pnl"] < 0, "pnl"].sum()
    return {**sc, "total_r": float(s["r"].sum()), "win": float((s["pnl"] > 0).mean()),
            "pf": float(w / l_) if l_ > 0 else np.inf}


def plateau(table: pd.DataFrame, row: pd.Series, grid: dict, col: str = "val_mean") -> float:
    """Share of grid neighbours (same family and timeframe) with positive `col`."""
    pt = {k: row[k] for k in grid}
    nb = neighbours(pt, grid)
    if not nb:
        return np.nan
    sub = table[(table["family"] == row["family"]) & (table["tf"] == row["tf"])]
    vals = []
    for n in nb:
        m = np.ones(len(sub), bool)
        for k, v in n.items():
            m &= np.isclose(sub[k].to_numpy(float), float(v))
        if m.any():
            vals.append(float(sub.loc[m, col].iloc[0]))
    vals = np.array(vals)
    return float((np.nan_to_num(vals, nan=-1) > 0).mean()) if len(vals) else np.nan


def eligibility(table: pd.DataFrame, grid: dict) -> pd.DataFrame:
    t = table.copy()
    t["plateau"] = [plateau(t, r, grid) for _, r in t.iterrows()]
    t["eligible"] = ((t["train_mean"] > 0) & (t["val_mean"] > 0) & (t["val_n"] >= MIN_VAL_TRADES)
                     & (t["plateau"] >= PLATEAU_SHARE))
    return t


def pick(t: pd.DataFrame, require_eligible: bool = True) -> pd.Series | None:
    pool = t[t["eligible"]] if require_eligible else t[t["val_n"] >= MIN_VAL_TRADES]
    pool = pool[np.isfinite(pool["val_score"])]
    if pool.empty:
        return None
    return pool.sort_values(["val_score", "val_n"], ascending=False).iloc[0]


# --------------------------------------------------------------------------- walk-forward

@dataclass(frozen=True)
class Fold:
    name: str
    is_lo: int
    is_hi: int
    oos_lo: int
    oos_hi: int


def quarterly_folds(dev_start: str, dev_end: str, is_months: int = 12) -> list[Fold]:
    start = pd.Timestamp(dev_start).to_period("M").start_time
    end = pd.Timestamp(dev_end)
    q = (start + pd.DateOffset(months=is_months)).to_period("Q").start_time
    if q < start + pd.DateOffset(months=is_months):
        q = q + pd.DateOffset(months=3)
    folds = []
    while q + pd.DateOffset(months=3) - pd.Timedelta(days=1) <= end + pd.Timedelta(days=1):
        oos_hi = q + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        is_lo = q - pd.DateOffset(months=is_months)
        i = lambda d: int(d.strftime("%Y%m%d"))  # noqa: E731
        folds.append(Fold(f"{q.year}Q{(q.month - 1) // 3 + 1}", i(is_lo), i(q - pd.Timedelta(days=1)),
                          i(q), i(oos_hi)))
        q = q + pd.DateOffset(months=3)
    return folds


def walk_forward(store: dict, keys: list, folds: list[Fold]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At each fold, pick the config with the best IN-SAMPLE score (IS mean > 0,
    enough trades) among `keys`, then record its trades in the next quarter.
    Returns (concatenated out-of-sample trades, per-fold choices)."""
    oos, choices = [], []
    for fd in folds:
        best, best_sc = None, -np.inf
        for k in keys:
            st = stats_for(store[k], fd.is_lo, fd.is_hi)
            if st["n"] >= MIN_IS_TRADES and st["mean"] > 0 and np.isfinite(st["score"]) and st["score"] > best_sc:
                best, best_sc = k, st["score"]
        if best is None:
            choices.append({"fold": fd.name, "choice": "NO TRADE (nothing positive in-sample)",
                            "is_score": np.nan, "oos_n": 0, "oos_mean": np.nan})
            continue
        tr = store[best]
        o = tr[(tr["exit_tday"] >= fd.oos_lo) & (tr["exit_tday"] <= fd.oos_hi)].copy()
        o["fold"] = fd.name
        oos.append(o)
        choices.append({"fold": fd.name, "choice": best, "is_score": best_sc, "oos_n": len(o),
                        "oos_mean": float(o["r"].mean()) if len(o) else np.nan})
    return (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=["r", "exit_tday", "pnl"]),
            pd.DataFrame(choices))


# --------------------------------------------------------------------------- strategy C router

def session_router(a_tr: pd.DataFrame, b_tr: pd.DataFrame, lo: int, hi: int) -> dict:
    """For RTH and ETH separately, route to the family with the higher TRAIN
    expectancy — or to nothing if neither is positive there."""
    route = {}
    for name, flag in (("RTH", True), ("ETH", False)):
        ea = a_tr[(a_tr["exit_tday"] >= lo) & (a_tr["exit_tday"] <= hi) & (a_tr["rth"] == flag)]["r"].mean()
        eb = b_tr[(b_tr["exit_tday"] >= lo) & (b_tr["exit_tday"] <= hi) & (b_tr["rth"] == flag)]["r"].mean()
        ea, eb = np.nan_to_num(ea, nan=-9), np.nan_to_num(eb, nan=-9)
        route[name] = "A" if (ea >= eb and ea > 0) else ("B" if eb > 0 else "NONE")
        route[name + "_train_exp"] = {"A": float(ea), "B": float(eb)}
    return route
