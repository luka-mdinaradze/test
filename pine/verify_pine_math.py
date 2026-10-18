#!/usr/bin/env python3
"""Check that the Pine script's formulas reproduce the Python engine's triggers.

TradingView cannot run here, so this mirrors the Pine logic bar by bar on
TradingView-like 15-minute bars (every minute of the Globex day, clock-aligned,
including the minutes after the flatten time) and compares the entry triggers
with the Python candidates on the development data.

    python3 pine/verify_pine_math.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nqlab import data, features  # noqa: E402
from nqlab.config import NQ_MARKET  # noqa: E402
from nqlab.live import regime_thresholds, train_profiles  # noqa: E402
from nqlab.strategies import from_candidate  # noqa: E402


def tv_bars(m1: pd.DataFrame, tf: int) -> pd.DataFrame:
    r = m1.reset_index()
    key = r["tday"].astype(str) + "_" + (r["mod"] // tf).astype(str)
    pre = ~r["post"]
    g, gp = r.groupby(key, sort=False), r[pre].groupby(key[pre], sort=False)
    return pd.DataFrame({"ts": g["ts"].first(), "open": g["open"].first(), "close": g["close"].last(),
                         "dmin": g["dmin"].first(), "c_pre": gp["close"].last()}).set_index("ts").sort_index()


def pine_mirror(tv: pd.DataFrame, prof, tf: int, flat_d: int, a, b):
    o, c, cpre, dmin = (tv[k].to_numpy() for k in ("open", "close", "c_pre", "dmin"))
    n = len(c)
    adj, lev = 0.0, np.nan
    pc, sig = np.empty(n), np.empty(n)
    for i in range(n):
        if i > 0 and dmin[i] < dmin[i - 1]:
            adj += o[i] - c[i - 1]
        straddle = dmin[i] < flat_d < dmin[i] + tf
        pc[i] = (cpre[i] if straddle else c[i]) - adj
        prev = pc[i - 1] if i > 0 else pc[i]
        sf = prof.vol[min(dmin[i] // tf, len(prof.vol) - 1)]
        u2 = ((pc[i] - prev) / sf) ** 2
        prev_lev = lev
        lev = (np.nan if i == 0 else u2) if np.isnan(lev) else 0.97 * lev + 0.03 * u2
        sig[i] = math.sqrt(prev_lev) * sf if not np.isnan(prev_lev) else np.nan
    d = np.diff(pc, prepend=pc[0])
    zmom = (pd.Series(d).rolling(a["w"]).sum() / np.sqrt(pd.Series(sig ** 2).rolling(a["w"]).sum())).to_numpy()
    s = pd.Series(pc)
    zbb = ((s - s.rolling(b["w"]).mean()) / s.rolling(b["w"]).std()).to_numpy()
    ew = (dmin + tf >= NQ_MARKET.session.no_entry_first_min) & (dmin + tf < flat_d - NQ_MARKET.session.no_entry_last_min)

    def up(x, t):
        p = np.r_[np.nan, x[:-1]]
        return (x > t) & (p <= t)

    def dn(x, t):
        p = np.r_[np.nan, x[:-1]]
        return (x < t) & (p >= t)
    return {"A": tv.index[ew & (up(zmom, a["tau"]) | dn(zmom, -a["tau"]))],
            "B": tv.index[ew & (dn(zbb, -b["z"]) | up(zbb, b["z"]))]}


def main():
    frozen = json.loads((ROOT / "results" / "frozen_config_NQ.json").read_text())
    tf = frozen["candidates"]["A"]["tf"]
    sess = NQ_MARKET.session
    m1 = data.prepare(data.load_nq_a(), sess)
    m1 = m1[m1["tday"] <= int(NQ_MARKET.splits[1].end.replace("-", ""))]
    f = features.build(data.resample(m1, tf, sess), train_profiles(NQ_MARKET, tf))
    th = regime_thresholds(NQ_MARKET, tf)
    tv = tv_bars(m1, tf)
    flat_d = (sess.flatten_min - sess.day_start_min) % 1440
    mirror = pine_mirror(tv, train_profiles(NQ_MARKET, tf), tf, flat_d,
                         frozen["candidates"]["A"]["params"], frozen["candidates"]["B"]["params"])
    start = tv.index[3000]                       # skip warm-up, where seeding differs
    for fam in ("A", "B"):
        s = from_candidate(f, tf, frozen["candidates"][fam], th)
        py = pd.Index(f.index[s.meta["bar"].to_numpy()])
        py, pi = py[py >= start], mirror[fam][mirror[fam] >= start]
        both = len(py.intersection(pi))
        print(f"{fam}: python {len(py)} triggers, pine-mirror {len(pi)}, common {both} "
              f"({100 * both / len(py):.1f}% of python, {100 * both / max(len(pi), 1):.1f}% of pine)")


if __name__ == "__main__":
    main()
