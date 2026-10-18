"""P(target before stop): estimation, calibration, and whether it is worth using.

The label is the triple-barrier outcome from engine.label: 1 if the take-profit
filled before the stop (and before the time limit), on 1-minute bars with
stop-first tie-breaking. Every feature is the value at the signal bar.

A probability is only reported by the live system if, on data the model never
saw, it is (a) better than the base rate (positive Brier skill), (b) calibrated
(reliability close to the diagonal), and (c) USEFUL: filtering on it improves
expectancy after costs. Failing any one, the system prints 'unavailable'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["side", "z_entry", "zmom_20", "zbb_20", "er_20", "vr4", "ac1", "vol_regime",
            "range_z", "volume_z", "tod_sin", "tod_cos", "rth", "stop_atr"]


def brier(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": q, "y": y, "p": p}).groupby("bin").agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return df


def ece(y, p, bins: int = 10) -> float:
    r = reliability(y, p, bins)
    return float((r["n"] * (r["predicted"] - r["observed"]).abs()).sum() / r["n"].sum())


def scores(y, p, base_rate: float) -> dict:
    b = brier(y, p)
    b0 = brier(y, np.full(len(y), base_rate))
    return {"n": int(len(y)), "observed_rate": float(np.mean(y)), "mean_pred": float(np.mean(p)),
            "brier": b, "brier_base": b0, "brier_skill": float(1 - b / b0) if b0 > 0 else np.nan,
            "log_loss": log_loss(y, p), "auc": auc(y, p), "ece": ece(y, p)}


@dataclass
class FittedModel:
    name: str
    model: object
    calibrator: object | None
    features: list
    fill: np.ndarray            # TRAINING medians used to impute missing features

    def matrix(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.features].to_numpy(float)
        return np.where(np.isfinite(A), A, self.fill)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(self.matrix(X))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 1e-4, 1 - 1e-4)


def fit(name: str, X: pd.DataFrame, y: np.ndarray, calibrate: bool = True, seed: int = 0) -> FittedModel:
    """Fit on the first 75% of the (time-ordered) rows, calibrate on the last 25%."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in FEATURES if c in X.columns]
    A = X[cols].to_numpy(float)
    cut = int(len(A) * 0.75) if calibrate else len(A)
    fill = np.nanmedian(A[:cut], axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    A = np.where(np.isfinite(A), A, fill)
    if name == "logistic":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    elif name == "gbm":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=seed)
    else:
        raise ValueError(name)
    m.fit(A[:cut], y[:cut])
    cal = None
    if calibrate and len(A) - cut > 50:
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(m.predict_proba(A[cut:])[:, 1], y[cut:])
    return FittedModel(name, m, cal, cols, fill)


def bayes_online(y: np.ndarray, group: np.ndarray, decided_at: np.ndarray, known_at: np.ndarray,
                 prior_p: float, prior_n: float = 20.0) -> np.ndarray:
    """Beta-binomial win probability per group, updated only with outcomes that
    were KNOWN (trade closed) before each signal was decided.

    decided_at / known_at are comparable integer clocks (1-minute bar index of the
    signal, and of the outcome's exit). The prior is deliberately sceptical:
    `prior_n` pseudo-trades at `prior_p` (use the break-even rate).
    """
    order_known = np.argsort(known_at, kind="stable")
    groups = np.unique(group)
    wins = {g: 0.0 for g in groups}
    cnt = {g: 0.0 for g in groups}
    out = np.empty(len(y))
    j = 0
    for i in np.argsort(decided_at, kind="stable"):
        while j < len(order_known) and known_at[order_known[j]] < decided_at[i]:
            k = order_known[j]
            wins[group[k]] += y[k]
            cnt[group[k]] += 1
            j += 1
        g = group[i]
        out[i] = (prior_p * prior_n + wins[g]) / (prior_n + cnt[g])
    return out


def breakeven_p(win_r: np.ndarray) -> np.ndarray:
    """Win probability at which EV = 0 when a win pays win_r and a loss costs 1R."""
    return 1.0 / (1.0 + win_r)
