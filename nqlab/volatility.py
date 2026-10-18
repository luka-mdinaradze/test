"""Volatility forecasting: does anything beat ATR at predicting the next bar?

Forecasters, each producing a std forecast for bar t+1 from bars <= t:
  atr        Wilder ATR(14) of the gap-removed bars
  ewma       EWMA std of changes, no seasonality
  seasonal   EWMA of deseasonalised changes x time-of-day factor (features.seasonal_vol)
  garch      GARCH(1,1) on deseasonalised changes, parameters fitted on TRAIN,
             then filtered forward with those fixed parameters (no refit on later data)

Because ATR measures range, not standard deviation, every forecaster is rescaled
by a single constant fitted on TRAIN so that E[d^2 / s^2] = 1 there. The
comparison is therefore about the shape of the forecast through time, which is
what matters for sizing and stops, not about its units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features


def _qlike(d2: np.ndarray, s2: np.ndarray) -> float:
    ok = np.isfinite(d2) & np.isfinite(s2) & (s2 > 0) & (d2 > 0)
    x = d2[ok] / s2[ok]
    return float(np.mean(x - np.log(x) - 1))


def _mz_r2(absd: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(absd) & np.isfinite(s)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(absd[ok], s[ok])[0, 1] ** 2)


def garch_filter(u: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional variance forecast for bar t+1 made at t (causal recursion)."""
    h = np.empty(len(u))
    prev = omega / max(1 - alpha - beta, 1e-6)
    for i in range(len(u)):
        x = u[i]
        if np.isfinite(x):
            prev = omega + alpha * x * x + beta * prev
        h[i] = prev
    return h


def fit_garch(u_train: np.ndarray, max_n: int = 150_000) -> dict:
    from arch import arch_model
    x = u_train[np.isfinite(u_train)][-max_n:]
    scale = np.std(x)
    am = arch_model(x / scale, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    r = am.fit(disp="off")
    p = r.params
    return {"omega": float(p["omega"]) * scale ** 2, "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "nu": float(p["nu"]), "persistence": float(p["alpha[1]"] + p["beta[1]"])}


def compare(f: pd.DataFrame, prof: features.Profiles, train: np.ndarray, test: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample forecast comparison. Returns (table, garch params)."""
    pc = f["pc"].to_numpy()
    d = np.diff(pc, prepend=np.nan)
    d_next = np.r_[d[1:], np.nan]
    slot = prof.slot(f["dmin"].to_numpy())
    sf = prof.vol[slot]
    sf_next = prof.vol[prof.slot(f["dmin"].to_numpy() + prof.tf)]
    same = np.r_[f["tday"].to_numpy()[1:] == f["tday"].to_numpy()[:-1], False]
    d_next[~same] = np.nan          # do not score forecasts across the daily halt

    u = d / sf
    gp = fit_garch(u[train])
    raw = {
        "atr": f["atr"].to_numpy(),
        "ewma": np.sqrt(pd.Series(d ** 2).ewm(alpha=0.03, adjust=False).mean().to_numpy()),
        "seasonal": f["sigma_next"].to_numpy(),
        "garch": np.sqrt(garch_filter(u, gp["omega"], gp["alpha"], gp["beta"])) * sf_next,
    }
    rows = []
    for name, s in raw.items():
        k = np.sqrt(np.nanmean((d_next[train] ** 2) / (s[train] ** 2)))
        s = s * k
        rows.append({"model": name,
                     "qlike_train": _qlike(d_next[train] ** 2, s[train] ** 2),
                     "qlike_test": _qlike(d_next[test] ** 2, s[test] ** 2),
                     "mz_r2_test": _mz_r2(np.abs(d_next[test]), s[test]),
                     "scale_k": float(k)})
    tab = pd.DataFrame(rows).set_index("model")
    base = tab.loc["atr", "qlike_test"]
    tab["qlike_gain_vs_atr_pct"] = 100 * (base - tab["qlike_test"]) / base
    return tab, gp
