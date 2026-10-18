"""Research harness: one object holding a market's bars, features and arrays.

The development lab is built with `upto` = the last validation day, so test
and holdout bars are not even loaded while anything is being selected. The
final evaluation builds a separate lab on the full data with the frozen config.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features
from .config import Account, Instrument, Market
from .engine import Market1m, Result, Signals, label, simulate

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Edge research sizes exactly (fractional units, constant equity) and switches
# the risk governor off, so the numbers measure the strategy, not the account.
RESEARCH_ACCOUNT = Account(equity=5_000.0, risk_pct=0.005, max_risk_pct=0.01,
                           daily_loss_limit_pct=1e9, max_consecutive_losses=10 ** 9, compounding=False)


def tday_int(s: str) -> int:
    return int(s.replace("-", ""))


@dataclass
class Lab:
    market: Market
    m1: pd.DataFrame
    mk: Market1m
    tf_frames: dict = field(default_factory=dict)
    feats: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    vwap_dev: np.ndarray | None = None
    source: str = ""

    @classmethod
    def build(cls, market: Market, source: str = "dev", upto: str | None = None,
              timeframes=(1, 5, 15, 60), profiles: dict | None = None) -> "Lab":
        if market.name == "NQ":
            raw = data.load_nq_a() if source == "dev" else data.load_nq_b()
        else:
            raw = data.load_btc()
        m1 = data.prepare(raw, market.session)
        if upto is not None:
            m1 = m1[m1["tday"] <= tday_int(upto)]
        lab = cls(market, m1, Market1m.build(m1, market.instrument), source=source)
        train = market.splits[0]
        for tf in timeframes:
            tfd = data.resample(m1, tf, market.session)
            if profiles is None:
                mask = data.split_mask(tfd["tday"].to_numpy(), train.start, train.end)
                prof = features.fit_profiles(tfd, tf, mask)
            else:
                prof = profiles[tf]
            lab.tf_frames[tf] = tfd
            lab.profiles[tf] = prof
            lab.feats[tf] = features.build(tfd, prof)
        lab.vwap_dev = features.session_vwap_dev(m1)
        log.info("lab %s/%s: %d 1m bars, %s .. %s", market.name, source, len(m1), m1.index[0], m1.index[-1])
        return lab

    def mask(self, tf: int, split) -> np.ndarray:
        return data.split_mask(self.feats[tf]["tday"].to_numpy(), split.start, split.end)

    def window(self, split) -> tuple[int, int]:
        return tday_int(split.start), tday_int(split.end)

    def split(self, name: str):
        for s in self.market.splits:
            if s.name == name:
                return s
        if name == "holdout":
            return self.market.holdout
        raise KeyError(name)

    def market1m(self, inst: Instrument) -> Market1m:
        return self.mk if inst == self.market.instrument else Market1m.build(self.m1, inst)

    def run(self, sig: Signals, split, inst: Instrument | None = None, acct: Account = RESEARCH_ACCOUNT,
            fractional: bool = True) -> Result:
        inst = inst or self.market.instrument
        return simulate(self.market1m(inst), sig, inst, acct, fractional=fractional, window=self.window(split))

    def label(self, sig: Signals, inst: Instrument | None = None) -> pd.DataFrame:
        inst = inst or self.market.instrument
        return label(self.market1m(inst), sig, inst)


def save_json(obj, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return p


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = True) -> str:
    """Minimal GitHub-markdown table (no extra dependency)."""
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def save_md(text: str, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(text)
    return p
