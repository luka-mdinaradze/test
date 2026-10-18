"""Causality and null-calibration tests for nqlab.features.

The causality test is the important one: features computed on data truncated
at bar t must equal the full-history features at bar t. Any look-ahead —
a centred window, a two-sided kernel, a seed from future bars — fails it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import data, features  # noqa: E402
from nqlab.config import NQ_SESSION  # noqa: E402


def _synthetic_m1(n_days=12, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range("2024-01-02", periods=n_days):
        idx = pd.date_range(d - pd.Timedelta(hours=6), periods=1380, freq="1min")  # 18:00 prev day
        c = 20000 + np.cumsum(rng.standard_normal(1380) * 2)
        o = np.r_[c[0], c[:-1]] + rng.standard_normal(1380) * 0.2
        hi = np.maximum(o, c) + np.abs(rng.standard_normal(1380))
        lo = np.minimum(o, c) - np.abs(rng.standard_normal(1380))
        frames.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                                    "volume": rng.integers(1, 500, 1380).astype(float)}, index=idx))
    df = pd.concat(frames)
    df.index.name = "ts"
    return data.prepare(df, NQ_SESSION)


@pytest.mark.parametrize("tf", [1, 5, 60])
def test_features_are_causal(tf):
    m1 = _synthetic_m1()
    tfd = data.resample(m1, tf, NQ_SESSION)
    prof = features.fit_profiles(tfd, tf, np.ones(len(tfd), bool))
    full = features.build(tfd, prof)
    rng = np.random.default_rng(0)
    for t in sorted(rng.choice(np.arange(50, len(tfd)), 6, replace=False)):
        part = features.build(tfd.iloc[: t + 1], prof)
        a = full.iloc[t].to_numpy(float)
        b = part.iloc[t].to_numpy(float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        bad = list(full.columns[~same])
        assert not bad, f"tf={tf} t={t}: look-ahead in {bad}"


def test_kalman_and_nw_are_causal():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.standard_normal(600))
    s2 = np.ones(600)
    sl, ss, inn = features.kalman_llt(y, s2, 0.01, 0.001, 1.0)
    sl2, _, _ = features.kalman_llt(y[:400], s2[:400], 0.01, 0.001, 1.0)
    assert np.allclose(sl[:400], sl2, equal_nan=True)
    nw = features.nw_causal(y, 5.0)
    nw2 = features.nw_causal(y[:400], 5.0)
    assert np.allclose(nw[:400], nw2, equal_nan=True)
    # weights favour the newest bar: after a step up, the estimate moves toward it
    step = np.r_[np.zeros(100), np.ones(5) * 10]
    out = features.nw_causal(step, 3.0)
    assert out[-1] > out[-3] > 0


def test_null_standardisation_has_unit_variance():
    rng = np.random.default_rng(11)
    n = 200_000
    pc = np.cumsum(rng.standard_normal(n) * 3.0)
    sig = np.full(n, 3.0)
    for w in (10, 40):
        zm = features.z_momentum(pc, sig, w)
        _, zr = features.z_regression(pc, sig, w)
        assert abs(np.nanstd(zm) - 1) < 0.03, np.nanstd(zm)
        assert abs(np.nanstd(zr) - 1) < 0.03, np.nanstd(zr)
        assert abs(np.nanmean(zm)) < 0.03


def test_variance_ratio_detects_persistence():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(20_000)
    rw = np.cumsum(e)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.3 * ar[i - 1] + e[i]           # positively autocorrelated increments
    tr = np.cumsum(ar)
    assert abs(np.nanmean(features.variance_ratio(rw, 4, 500)) - 1) < 0.05
    assert np.nanmean(features.variance_ratio(tr, 4, 500)) > 1.3


def test_variance_ratio_z_is_standard_normal_under_iid():
    from nqlab import stats_tests
    rng = np.random.default_rng(4)
    zs = [stats_tests.lo_mackinlay(rng.standard_normal(20_000), 4)["z"] for _ in range(200)]
    assert abs(np.mean(zs)) < 0.2 and abs(np.std(zs) - 1) < 0.15, (np.mean(zs), np.std(zs))
    e = rng.standard_normal(50_000)
    ar = np.zeros_like(e)
    for i in range(1, len(e)):
        ar[i] = 0.05 * ar[i - 1] + e[i]
    assert stats_tests.lo_mackinlay(ar, 4)["z"] > 5      # weak persistence, large sample: detected
