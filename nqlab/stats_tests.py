"""Hypothesis tests on price changes — run before any strategy is built.

Each test answers one question of the form "is there measurable serial
structure of this kind, at this timeframe, in this period?". A strategy is only
worth building on structure that shows up here AND persists from the training
to the validation period.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .validation import cluster_mean_test


def lo_mackinlay(d: np.ndarray, q: int) -> dict:
    """Variance ratio VR(q) with the heteroskedasticity-robust z* (Lo & MacKinlay 1988).

    VR > 1: positive serial correlation (trend); VR < 1: mean reversion.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    mu = d.mean()
    e = d - mu
    var_a = (e ** 2).sum() / (T - 1)
    sq = np.convolve(d, np.ones(q), mode="valid") - q * mu
    m = q * (T - q + 1) * (1 - q / T)
    var_c = (sq ** 2).sum() / m
    vr = var_c / var_a
    e2 = e ** 2
    den = e2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        delta = T * (e2[j:] * e2[:-j]).sum() / den
        theta += (2 * (q - j) / q) ** 2 * delta
    z = np.sqrt(T) * (vr - 1) / np.sqrt(theta)      # psi* in Campbell, Lo & MacKinlay (1997)
    return {"q": q, "vr": float(vr), "z": float(z), "p": float(2 * stats.norm.sf(abs(z))), "n": T}


def ljung_box(x: np.ndarray, lags: int) -> tuple[float, float]:
    """Ljung-Box Q from the FFT autocorrelation: Q = n(n+2) sum rho_k^2/(n-k).
    Same statistic as statsmodels' acorr_ljungbox, without its O(n^2) correlation."""
    from statsmodels.tsa.stattools import acf
    n = len(x)
    rho = acf(x, nlags=lags, fft=True)[1:]
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def acf_ljungbox(d: np.ndarray, lags: int = 10) -> dict:
    """Autocorrelation of changes and of squared changes (volatility clustering).

    Ljung-Box assumes homoskedasticity; with volatility clustering it over-rejects,
    so read lb_p together with the robust variance-ratio z*."""
    from statsmodels.tsa.stattools import acf
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    a = acf(d, nlags=lags, fft=True)
    _, lb_p = ljung_box(d, lags)
    q2, p2 = ljung_box(d ** 2 - np.mean(d ** 2), lags)
    return {"acf1": float(a[1]), "acf2": float(a[2]), "acf5": float(a[5]),
            "lb_p": lb_p, "lb_sq_stat": q2, "lb_sq_p": p2,
            "se_acf": float(1 / np.sqrt(len(d))), "n": len(d)}


def stationarity(x: np.ndarray, max_n: int = 50_000) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary), on the last max_n points."""
    from statsmodels.tsa.stattools import adfuller, kpss
    x = np.asarray(x, float)
    x = x[np.isfinite(x)][-max_n:]
    adf = adfuller(x, maxlag=10, autolag=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(x, regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_p": float(adf[1]), "kpss_stat": float(kp[0]),
            "kpss_p": float(kp[1]), "n": len(x)}


def arch_lm(d: np.ndarray, lags: int = 10, max_n: int = 200_000) -> dict:
    from statsmodels.stats.diagnostic import het_arch
    d = np.asarray(d, float)
    d = d[np.isfinite(d)][-max_n:]
    lm, p, _, _ = het_arch(d - d.mean(), nlags=lags)
    return {"arch_lm": float(lm), "arch_p": float(p)}


def conditional_forward(f: pd.DataFrame, cond_long: np.ndarray, cond_short: np.ndarray,
                        horizon: int, sigma_col: str = "sigma") -> dict:
    """Mean forward change over `horizon` bars after a condition, in units of sigma*sqrt(h).

    Signed so that a positive number means 'price moved in the long direction
    after cond_long and in the short direction after cond_short'. Day-clustered
    standard errors. Under no predictability the expected value is ~0.
    """
    pc = f["pc"].to_numpy()
    fwd = np.r_[pc[horizon:] - pc[:-horizon], np.full(horizon, np.nan)]
    same_day = np.r_[f["tday"].to_numpy()[horizon:] == f["tday"].to_numpy()[:-horizon], np.zeros(horizon, bool)]
    norm = fwd / (f[sigma_col].to_numpy() * np.sqrt(horizon))
    norm[~same_day] = np.nan
    x = np.r_[norm[cond_long], -norm[cond_short]]
    cl = np.r_[f["tday"].to_numpy()[cond_long], f["tday"].to_numpy()[cond_short]]
    return cluster_mean_test(x, cl)


def intraday_momentum(m1: pd.DataFrame, rth_start: int, rth_end: int) -> dict:
    """Gao, Han, Li & Zhou (2018): does the first 30 minutes' return (plus the
    overnight return) predict the last 30 minutes' return? Daily regression,
    HAC standard errors. Returns are in points of the gap-removed price for the
    within-session legs; the overnight leg uses raw prices.
    """
    import statsmodels.api as sm
    mod = m1["mod"].to_numpy()
    c = m1["close"].to_numpy()
    o = m1["open"].to_numpy()
    g = pd.DataFrame({"tday": m1["tday"].to_numpy(), "mod": mod, "o": o, "c": c})
    first_o = g[g["mod"] == rth_start].groupby("tday")["o"].first()
    first_c = g[g["mod"] == rth_start + 29].groupby("tday")["c"].first()
    last_o = g[g["mod"] == rth_end - 30].groupby("tday")["o"].first()
    last_c = g[g["mod"] == rth_end - 1].groupby("tday")["c"].first()
    prev_close = last_c.shift(1)
    df = pd.DataFrame({"r_first": first_c - first_o, "r_on": first_o - prev_close,
                       "r_last": last_c - last_o, "level": first_o}).dropna()
    for k in ("r_first", "r_on", "r_last"):
        df[k] = df[k] / df["level"]      # fraction of price; level differences cancel in ratios
    X = sm.add_constant(df[["r_first", "r_on"]])
    fit = sm.OLS(df["r_last"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    sign_hit = float((np.sign(df["r_first"] + df["r_on"]) == np.sign(df["r_last"])).mean())
    return {"days": len(df), "beta_first": float(fit.params["r_first"]), "t_first": float(fit.tvalues["r_first"]),
            "beta_on": float(fit.params["r_on"]), "t_on": float(fit.tvalues["r_on"]),
            "r2": float(fit.rsquared), "sign_agreement": sign_hit}
