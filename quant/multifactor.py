"""Cross-sectional multi-factor model: momentum, value, volatility, trend.

Works on a universe of price series (daily bars are the natural frequency; the
same code runs on 1h bars if you shorten the windows). Each factor is turned
into a cross-sectional z-score at each rebalance date, combined with fixed
weights, and mapped to positions with inverse-volatility sizing.

The factors are deliberately simple and slow-moving. Cross-sectional edges
survive costs precisely because they do not need to be traded often.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorWeights:
    momentum: float = 0.35
    trend: float = 0.30
    volatility: float = 0.20
    value: float = 0.15

    def as_dict(self) -> dict:
        return {"momentum": self.momentum, "trend": self.trend,
                "volatility": self.volatility, "value": self.value}

    def normalised(self) -> dict:
        d = self.as_dict()
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}


@dataclass
class FactorConfig:
    mom_lookback: int = 252       # bars in the momentum window
    mom_skip: int = 21            # skip the most recent month (reversal effect)
    vol_window: int = 63
    value_anchor: int = 504       # ~2y median price as the "fair value" anchor
    trend_mas: tuple = (50, 100, 200)
    top_n: int = 3
    long_short: bool = False
    rebalance: str = "ME"         # pandas offset alias: ME=month-end, W-FRI=weekly
    target_vol: float = 0.15
    max_weight: float = 0.40


def _zscore_row(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def compute_factors(prices: pd.DataFrame, cfg: FactorConfig = FactorConfig()) -> dict:
    """prices: wide frame, one close-price column per asset."""
    out = {}

    # MOMENTUM: total return over the lookback, skipping the last `mom_skip`
    # bars. Risk-adjusted by realised vol so a quiet 40% beats a violent 40%.
    lb, sk = cfg.mom_lookback, cfg.mom_skip
    raw_mom = prices.shift(sk) / prices.shift(lb) - 1.0
    vol = np.log(prices).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std(ddof=0) * np.sqrt(252)
    out["momentum"] = raw_mom / vol.replace(0, np.nan)

    # TREND: time-series (not cross-sectional) participation score in [-1, 1].
    trend = None
    for w in cfg.trend_mas:
        ma = prices.rolling(w, min_periods=w // 2).mean()
        sig = np.sign(prices - ma)
        trend = sig if trend is None else trend + sig
    out["trend"] = trend / len(cfg.trend_mas)

    # VOLATILITY: low-vol preference -> negate realised vol.
    out["volatility"] = -vol

    # VALUE: price relative to a long anchor. Below anchor = cheap = positive.
    anchor = prices.rolling(cfg.value_anchor, min_periods=cfg.value_anchor // 3).median()
    out["value"] = -(prices / anchor - 1.0)

    return out


def score(prices: pd.DataFrame, weights: FactorWeights = FactorWeights(),
          cfg: FactorConfig = FactorConfig()) -> pd.DataFrame:
    f = compute_factors(prices, cfg)
    w = weights.normalised()
    total = None
    for name, frame in f.items():
        z = frame.apply(_zscore_row, axis=1)          # cross-sectional at each date
        contrib = z * w[name]
        total = contrib if total is None else total.add(contrib, fill_value=0.0)
    return total


def target_weights(scores: pd.DataFrame, prices: pd.DataFrame,
                   cfg: FactorConfig = FactorConfig()) -> pd.DataFrame:
    """Map composite scores to portfolio weights at each rebalance date."""
    vol = np.log(prices).diff().rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std(ddof=0) * np.sqrt(252)
    dates = prices.resample(cfg.rebalance).last().index
    rows = {}
    for d in dates:
        if d not in scores.index:
            prior = scores.index[scores.index <= d]
            if prior.empty:
                continue
            d_use = prior[-1]
        else:
            d_use = d
        s = scores.loc[d_use].dropna()
        if s.empty:
            continue
        v = vol.loc[d_use].reindex(s.index)
        longs = s.nlargest(min(cfg.top_n, len(s)))
        longs = longs[longs > 0]
        w = pd.Series(0.0, index=prices.columns)
        if not longs.empty:
            iv = (1.0 / v.reindex(longs.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if iv.sum() > 0:
                w.loc[longs.index] = iv / iv.sum()
        if cfg.long_short:
            shorts = s.nsmallest(min(cfg.top_n, len(s)))
            shorts = shorts[shorts < 0]
            if not shorts.empty:
                iv = (1.0 / v.reindex(shorts.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if iv.sum() > 0:
                    w.loc[shorts.index] -= (iv / iv.sum()) * 0.5
        w = w.clip(-cfg.max_weight, cfg.max_weight)

        # Volatility targeting on the resulting book (scale down only).
        if w.abs().sum() > 0:
            port_vol = float(np.sqrt((w * v.reindex(w.index).fillna(0)) @ (w * v.reindex(w.index).fillna(0))))
            if port_vol > 0:
                w = w * min(1.0, cfg.target_vol / port_vol)
        rows[d_use] = w
    return pd.DataFrame(rows).T.sort_index()


def backtest(prices: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 10.0) -> pd.Series:
    """Hold each rebalance's weights until the next one; charge turnover costs."""
    rets = prices.pct_change().fillna(0.0)
    held = weights.reindex(rets.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * cost_bps / 10_000.0
    return (1 + net).cumprod()
