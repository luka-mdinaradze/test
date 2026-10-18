"""Regime detection, strictly causal.

Three families, each evaluated by whether it predicts something useful about
the NEXT bars out of sample — not by how plausible its labels look in hindsight.

1. Hidden Markov Model. Parameters are fitted on TRAIN with hmmlearn. The state
   probabilities used anywhere in this project come from `forward_filter`, which
   only uses observations up to t. hmmlearn's predict()/predict_proba() run the
   forward-BACKWARD pass (or Viterbi), which reads the future — using them in a
   backtest is look-ahead, and a common reason HMM strategies look brilliant.
2. Rule-based: trend/range from the rolling variance ratio and efficiency ratio,
   volatility terciles from the deseasonalised vol level; thresholds are
   quantiles of TRAIN data.
3. Change-point: a two-sided CUSUM on the log vol level that flags a
   'transition' for a fixed number of bars after an alarm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

TREND, RANGE, NEUTRAL = 1, -1, 0


# --------------------------------------------------------------------------- HMM

def hmm_obs(f: pd.DataFrame) -> np.ndarray:
    """Observation vector: standardised change (clipped) and log standardised range."""
    r = np.clip(f["ret_z"].to_numpy(), -8, 8)
    g = np.log(np.clip(f["range_z"].to_numpy(), 0.05, 50))
    return np.column_stack([r, g])


@dataclass
class HMMModel:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray      # diagonal covariances, (k, d)
    order: np.ndarray          # states sorted by mean log-range (0 = calmest)

    @property
    def expected_duration(self) -> np.ndarray:
        return 1.0 / (1.0 - np.diag(self.transmat))


def fit_hmm(X: np.ndarray, n_states: int = 3, seed: int = 0, max_n: int = 200_000) -> HMMModel:
    from hmmlearn.hmm import GaussianHMM
    ok = np.all(np.isfinite(X), axis=1)
    Xf = X[ok][-max_n:]
    m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, tol=1e-4,
                    random_state=seed, init_params="stmc")
    m.fit(Xf)
    order = np.argsort(m.means_[:, 1])
    var = np.array([np.diag(c) for c in m.covars_])
    return HMMModel(m.startprob_[order], m.transmat_[np.ix_(order, order)],
                    m.means_[order], var[order], order)


@njit(cache=True)
def _forward(logb, logA, logpi):
    n, k = logb.shape
    out = np.full((n, k), np.nan)
    alpha = logpi.copy()
    started = False
    for t in range(n):
        if not np.isfinite(logb[t, 0]):
            if started:
                out[t] = np.exp(alpha)
            continue
        if started:
            pred = np.empty(k)
            for j in range(k):
                m = -np.inf
                for i in range(k):
                    m = max(m, alpha[i] + logA[i, j])
                s = 0.0
                for i in range(k):
                    s += np.exp(alpha[i] + logA[i, j] - m)
                pred[j] = m + np.log(s)
        else:
            pred = logpi.copy()
            started = True
        a = pred + logb[t]
        mx = a.max()
        a = a - (mx + np.log(np.exp(a - mx).sum()))
        alpha = a
        out[t] = np.exp(alpha)
    return out


def forward_filter(model: HMMModel, X: np.ndarray) -> np.ndarray:
    """P(state_t | x_1..x_t) — filtered, never smoothed."""
    var = model.variances
    logb = -0.5 * (np.log(2 * np.pi * var)[None, :, :]
                   + (X[:, None, :] - model.means[None, :, :]) ** 2 / var[None, :, :]).sum(-1)
    logb[~np.all(np.isfinite(X), axis=1)] = np.nan
    return _forward(logb, np.log(model.transmat + 1e-300), np.log(model.startprob + 1e-300))


# --------------------------------------------------------------------------- rules

@dataclass
class RuleThresholds:
    vr_hi: float
    vr_lo: float
    er_hi: float
    er_lo: float
    vol_q: tuple          # (33%, 67%, 95%) quantiles of vol_regime on TRAIN
    er_col: str


def fit_rules(f: pd.DataFrame, train: np.ndarray, er_col: str = "er_20", q: float = 0.30) -> RuleThresholds:
    vr = f["vr4"].to_numpy()[train]
    er = f[er_col].to_numpy()[train]
    vg = f["vol_regime"].to_numpy()[train]
    return RuleThresholds(float(np.nanquantile(vr, 1 - q)), float(np.nanquantile(vr, q)),
                          float(np.nanquantile(er, 1 - q)), float(np.nanquantile(er, q)),
                          tuple(float(np.nanquantile(vg, a)) for a in (0.33, 0.67, 0.95)), er_col)


def rule_regime(f: pd.DataFrame, th: RuleThresholds, use: str = "vr") -> np.ndarray:
    """TREND / RANGE / NEUTRAL from either the variance ratio or the efficiency ratio."""
    if use == "vr":
        x, hi, lo = f["vr4"].to_numpy(), th.vr_hi, th.vr_lo
    else:
        x, hi, lo = f[th.er_col].to_numpy(), th.er_hi, th.er_lo
    out = np.full(len(f), NEUTRAL, np.int64)
    out[x >= hi] = TREND
    out[x <= lo] = RANGE
    return out


def vol_tercile(f: pd.DataFrame, th: RuleThresholds) -> np.ndarray:
    """0 low / 1 normal / 2 high / 3 extreme (top 5% of TRAIN)."""
    v = f["vol_regime"].to_numpy()
    out = np.digitize(v, th.vol_q)
    out[~np.isfinite(v)] = 1
    return out


# --------------------------------------------------------------------------- change points

@njit(cache=True)
def cusum_transition(x, k, h, hold, slow):
    """Two-sided CUSUM on x around a slow EWMA mean. Returns a 0/1 'transition' flag
    that stays on for `hold` bars after each alarm."""
    n = len(x)
    flag = np.zeros(n, np.int64)
    mu = np.nan
    sp = 0.0
    sn = 0.0
    left = 0
    for t in range(n):
        v = x[t]
        if not np.isfinite(v):
            flag[t] = 1 if left > 0 else 0
            continue
        if not np.isfinite(mu):
            mu = v
        dev = v - mu
        sp = max(0.0, sp + dev - k)
        sn = max(0.0, sn - dev - k)
        if sp > h or sn > h:
            left = hold
            sp = 0.0
            sn = 0.0
            mu = v
        if left > 0:
            flag[t] = 1
            left -= 1
        mu = slow * mu + (1 - slow) * v
    return flag


def change_points(f: pd.DataFrame, k: float = 0.25, h: float = 3.0, hold: int = 10, slow: float = 0.995) -> np.ndarray:
    x = np.log(f["sigma_level"].to_numpy())
    sd = np.nanstd(np.diff(x))
    return cusum_transition(x / max(sd, 1e-12) * 0.1, k, h, hold, slow)


# --------------------------------------------------------------------------- evaluation

def forward_realised(f: pd.DataFrame, h: int) -> pd.DataFrame:
    """What actually happened over the next h bars (same day only): realised vol
    and realised variance ratio. Used ONLY to score classifiers, never as an input."""
    pc = f["pc"].to_numpy()
    d = np.diff(pc, prepend=np.nan)
    tday = f["tday"].to_numpy()
    n = len(pc)
    fwd_abs = pd.Series(np.abs(d)).rolling(h).mean().shift(-h).to_numpy().copy()
    move = np.r_[pc[h:] - pc[:-h], np.full(h, np.nan)]
    path = pd.Series(d ** 2).rolling(h).sum().shift(-h).to_numpy()
    vr_fwd = move ** 2 / path
    same = np.r_[tday[h:] == tday[:-h], np.zeros(h, bool)] if n > h else np.zeros(n, bool)
    fwd_abs[~same] = np.nan
    vr_fwd[~same] = np.nan
    return pd.DataFrame({"fwd_absd": fwd_abs, "fwd_vr": vr_fwd}, index=f.index)
