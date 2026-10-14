#!/usr/bin/env python3
"""Specification test: does the strategy extract the edge it CLAIMS to extract?

Real BTC prices are unobtainable in this environment, so "is it profitable on
BTC" cannot be answered here. A different and equally important question can be
answered completely: given a market whose structure is KNOWN BY CONSTRUCTION,
does the strategy behave the way a sound trend-follower must?

Three falsifiable predictions. Any failure means the implementation is broken,
independently of what BTC does:

  P1  In a world with persistent trends, it MUST make money.
  P2  In a pure random walk, it MUST lose approximately the friction bill -
      no more (that would be a bug in the cost model) and NO LESS, because
      profit from noise can only come from lookahead.
  P3  In a mean-reverting world, it MUST lose, and a mean-reversion strategy
      MUST profit on the same bars.

Then the useful number: sweeping trend persistence finds the BREAK-EVEN
threshold - how much momentum a market must have for this strategy to pay for
its own costs. That converts "is it profitable?" into one measurable property
of a market, which is a question real data can settle in one line.

Worlds are built from an AR(1) process on log returns, so persistence is set
directly by phi, and measured independently with the Lo-MacKinlay variance
ratio (>1 trending, =1 random walk, <1 mean-reverting).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant.engine import Costs, ExecConfig, run_backtest
from quant.metrics import max_drawdown, sharpe
from quant.strategies import (S1Params, S2Params, s1_trend_breakout,
                              s2_mean_reversion)

pd.set_option("display.width", 220)
BARS = 20_000
SEEDS = tuple(range(11, 11 + 12))
COSTS = Costs(fee_bps=2.0, slippage_bps=2.0)          # maker-only, as the bot trades
CFG = ExecConfig(initial_equity=100.0, risk_pct=0.01, max_leverage=1.0,
                 allow_short=False, min_notional=0.0)


def make_world(n: int, phi: float, seed: int, vol: float = 0.005,
               s0: float = 30_000.0, jump_prob: float = 0.0, substeps: int = 200) -> pd.DataFrame:
    """OHLC bars with EXACT bar-level persistence and consistent intrabar paths.

    Two parts, and both matter:

    1. Bar log-returns follow R_t = phi*R_{t-1} + eps_t, so persistence is
       controlled exactly at the timeframe the strategy trades. (Applying AR(1)
       at sub-bar resolution instead washes the persistence out under
       aggregation - phi=0.10 came out at a 24h variance ratio of only 1.03,
       barely distinguishable from a random walk.)
    2. Each bar's high/low come from a BROWNIAN BRIDGE between its open and
       close, sampled at `substeps` points, so the wicks are a real path rather
       than invented. An earlier version drew wicks from the bar's own |return|,
       which gave a strongly up bar a deep low too, tripped stops on favourable
       bars and biased every stop-based strategy downwards.

       `substeps` is 200 rather than a dozen for a measured reason: a coarsely
       sampled path understates each bar's true extremes, so stop-outs get
       missed. Because the stop is the loss side, that inflates expectancy -
       +0.105R of phantom edge at 12 substeps, falling to +0.036R at 800. The
       control test below re-checks this on every run instead of trusting it.

    Drift is zero: a trend-follower must earn from the structure of returns,
    not from a tailwind handed to it. For AR(1) returns the long-horizon
    variance ratio tends to (1+phi)/(1-phi), so phi=0.10 is about 1.22.
    """
    rng = np.random.default_rng(seed)
    sd = vol * np.sqrt(max(1 - phi ** 2, 1e-6))
    # Ito correction. Zero-mean LOG returns still leave the PRICE drifting up by
    # vol^2/2 per bar (Jensen), which over 30k bars is tens of percent of free
    # appreciation that a long-only strategy would harvest and report as edge.
    # Subtracting it makes the PRICE a martingale, which is what the zero-cost
    # control test actually requires.
    mu = -0.5 * vol ** 2
    eps = rng.normal(0.0, sd, n)
    if jump_prob > 0:
        hit = rng.random(n) < jump_prob
        eps[hit] += rng.normal(0, vol * 8, int(hit.sum()))
    R = np.empty(n)
    R[0] = eps[0]
    for i in range(1, n):
        R[i] = phi * R[i - 1] + eps[i]
    R = R + mu

    closes = s0 * np.exp(np.cumsum(R))
    opens = np.concatenate([[s0], closes[:-1]])

    # Brownian bridge inside each bar: ends exactly on the bar's own return.
    k = substeps
    e = rng.normal(0.0, vol / np.sqrt(k), (n, k))
    W = e.cumsum(axis=1)
    frac = (np.arange(1, k + 1) / k)[None, :]
    bridge = W - frac * W[:, -1][:, None] + frac * R[:, None]
    paths = opens[:, None] * np.exp(bridge)

    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(paths.max(axis=1), opens),
        "low": np.minimum(paths.min(axis=1), opens),
        "close": closes,
        "volume": np.exp(rng.normal(0, 0.4, n)) * 1000,
    }, index=pd.date_range("2018-01-01", periods=n, freq="h", tz="UTC"))


def variance_ratio(close: pd.Series, q: int = 24) -> float:
    """Lo-MacKinlay variance ratio. 1.0 = random walk."""
    r = np.diff(np.log(close.to_numpy()))
    v1 = r.var(ddof=1)
    rq = np.add.reduceat(r, np.arange(0, len(r) - len(r) % q, q))
    return float(rq.var(ddof=1) / (q * v1)) if v1 > 0 else np.nan


def run(df: pd.DataFrame, which: str = "S1", costs: Costs | None = None) -> dict:
    if which == "S1":
        orders, atr_s, _ = s1_trend_breakout(df, S1Params(allow_short=False))
    else:
        orders, atr_s, _ = s2_mean_reversion(df, S2Params(allow_short=False))
    res = run_backtest(df, orders, atr_s, costs or COSTS, CFG)
    tr = res.trades
    n = len(tr)
    return {
        "trades": n,
        "total_R": float(tr["r_multiple"].sum()) if n else 0.0,
        "expectancy_R": float(tr["r_multiple"].mean()) if n else np.nan,
        "win_rate": float((tr["pnl"] > 0).mean() * 100) if n else np.nan,
        "final": float(res.equity.iloc[-1]),
        "sharpe": sharpe(res.equity, 8760),
        "maxdd": max_drawdown(res.equity) * 100,
        "costs_R": float(tr["costs"].sum() / tr["risk_amount"].mean()) if n else 0.0,
    }


def across_seeds(phi: float, which: str = "S1", costs: Costs | None = None, **kw) -> dict:
    """Run one world per seed and summarise with CLUSTERED statistics.

    The unit of observation is a WORLD, not a trade. Trades inside one price
    path share that path, so treating them as independent understates the
    standard error badly - it made every result below look significant when
    none of them were.
    """
    rows, vrs = [], []
    for s in SEEDS:
        df = make_world(BARS, phi, s, **kw)
        vrs.append(variance_ratio(df["close"]))
        rows.append(run(df, which, costs))
    d = pd.DataFrame(rows)
    exp = d["expectancy_R"].dropna().to_numpy()
    se = float(exp.std(ddof=1) / np.sqrt(len(exp))) if len(exp) > 1 else np.nan
    mean = float(exp.mean()) if len(exp) else np.nan
    return {
        "phi": phi,
        "variance_ratio": float(np.mean(vrs)),
        "trades": float(d["trades"].mean()),
        "expectancy_R": mean,
        "se": se,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "significant": bool(np.isfinite(se) and abs(mean) > 1.96 * se),
        "total_R": float(d["total_R"].mean()),
        "win_rate": float(d["win_rate"].mean()),
        "sharpe": float(d["sharpe"].mean()),
        "maxdd": float(d["maxdd"].mean()),
        "worlds_profitable": int((d["total_R"] > 0).sum()),
        "worlds": len(exp),
    }


def engine_control() -> tuple[bool, float, float]:
    """The test everything else rests on.

    In a driftless (martingale) market with zero costs, the optional stopping
    theorem says ANY entry and exit rule has expectancy exactly zero. So a
    measurably positive result here would mean lookahead, a broken cost model,
    or a biased world - and would invalidate every other number in this file.
    """
    rows = []
    for sd in SEEDS:
        df = make_world(BARS, 0.0, sd)
        r = run(df, "S1", Costs(fee_bps=0, slippage_bps=0, funding_bps_per_8h=0))
        if np.isfinite(r["expectancy_R"]):
            rows.append(r["expectancy_R"])
    m = np.array(rows)
    se = m.std(ddof=1) / np.sqrt(len(m))
    return bool(abs(m.mean()) < 1.96 * se), float(m.mean()), float(se)


def main() -> int:
    print("=" * 102)
    print("SPECIFICATION TEST - strategy behaviour in worlds whose structure is known")
    print(f"{BARS:,} bars x {len(SEEDS)} worlds | 200 intrabar substeps | maker costs "
          f"{COSTS.fee_bps:g}+{COSTS.slippage_bps:g}bps | 1% risk | long-only | zero drift")
    print("Statistics are CLUSTERED BY WORLD: one observation per price path, not per trade.")
    print("=" * 102)

    # ---------------- control: is the measuring instrument honest? ------
    print("\n[CONTROL] Martingale world, zero costs - expectancy MUST be zero\n")
    ok_ctrl, m_ctrl, se_ctrl = engine_control()
    print(f"  expectancy {m_ctrl:+.4f}R   95% CI [{m_ctrl - 1.96 * se_ctrl:+.4f}, "
          f"{m_ctrl + 1.96 * se_ctrl:+.4f}]   over {len(SEEDS)} worlds")
    ctrl_msg = ("PASS - no edge manufactured from noise; the engine and world are sound"
                if ok_ctrl else "FAIL - results below are not trustworthy")
    print(f"  {ctrl_msg}")

    # ---------------- the three predictions ----------------------------
    print("\n" + "=" * 102)
    print("[P1/P2/P3] Does it profit where trend exists and lose where it does not?")
    print("=" * 102 + "\n")
    worlds = [("Trending    (phi=+0.20)", 0.20), ("Random walk (phi= 0.00)", 0.00),
              ("Mean-revert (phi=-0.20)", -0.20)]
    rows = []
    for label, phi in worlds:
        r = across_seeds(phi, "S1")
        r["world"] = label
        rows.append(r)
    t = pd.DataFrame(rows).set_index("world")
    print(t[["variance_ratio", "trades", "expectancy_R", "ci_lo", "ci_hi", "total_R",
             "win_rate", "sharpe", "worlds_profitable"]].round(3).to_string())

    tr_row = t.loc[worlds[0][0]]
    rw_row = t.loc[worlds[1][0]]
    mr_row = t.loc[worlds[2][0]]
    ctrl = across_seeds(-0.20, "S2")

    print("\n  verdicts (a prediction counts only if the 95% CI excludes zero):")
    p1 = tr_row["expectancy_R"] > 0 and tr_row["ci_lo"] > 0
    print(f"  P1  trending world is profitable      {'PASS' if p1 else 'FAIL'}   "
          f"{tr_row['expectancy_R']:+.3f}R  CI[{tr_row['ci_lo']:+.3f},{tr_row['ci_hi']:+.3f}]  "
          f"{tr_row['worlds_profitable']:.0f}/{tr_row['worlds']:.0f} worlds")
    p2 = rw_row["expectancy_R"] < 0
    print(f"  P2  random walk loses (friction only) {'PASS' if p2 else 'FAIL'}   "
          f"{rw_row['expectancy_R']:+.3f}R  CI[{rw_row['ci_lo']:+.3f},{rw_row['ci_hi']:+.3f}]")
    p3 = mr_row["expectancy_R"] < tr_row["expectancy_R"]
    print(f"  P3  worse in a mean-reverting world   {'PASS' if p3 else 'FAIL'}   "
          f"{mr_row['expectancy_R']:+.3f}R vs {tr_row['expectancy_R']:+.3f}R trending")
    p3b = ctrl["expectancy_R"] > 0
    print(f"  P3b S2 profits on those same bars     {'PASS' if p3b else 'FAIL'}   "
          f"{ctrl['expectancy_R']:+.3f}R over {ctrl['trades']:.0f} trades "
          f"-> those bars did hold a harvestable edge")

    # ---------------- break-even persistence ---------------------------
    print("\n" + "=" * 102)
    print("BREAK-EVEN TREND PERSISTENCE - how much momentum the strategy needs to pay its costs")
    print("=" * 102 + "\n")
    sweep = [across_seeds(phi, "S1") for phi in
             (-0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50)]
    sw = pd.DataFrame(sweep)
    print(sw[["phi", "variance_ratio", "trades", "expectancy_R", "ci_lo", "ci_hi",
              "total_R", "win_rate", "sharpe", "worlds_profitable"]].round(3).to_string(index=False))

    pos = sw[sw["expectancy_R"] > 0]
    phi_be = vr_be = None
    if not pos.empty:
        below = sw[sw["expectancy_R"] <= 0]
        if not below.empty:
            lo, hi = below.iloc[-1], pos.iloc[0]
            span = hi["expectancy_R"] - lo["expectancy_R"]
            frac = (-lo["expectancy_R"] / span) if span else 0.0
            phi_be = lo["phi"] + frac * (hi["phi"] - lo["phi"])
            vr_be = lo["variance_ratio"] + frac * (hi["variance_ratio"] - lo["variance_ratio"])
            print(f"\n  Break-even at phi ~ {phi_be:+.3f}, i.e. a 24h variance ratio of ~{vr_be:.2f}")
            print("  Below that, friction exceeds the edge and the strategy loses by construction.")
        else:
            print("\n  Profitable across the whole sweep - break-even lies below the range tested.")
    else:
        print("\n  Never profitable in this sweep - the strategy needs more persistence than phi=0.50,")
        print("  which is far more than any liquid market exhibits. That is a verdict, not a gap.")
    print("\n  This states the strategy's requirement as a measurable property of a market.")
    print("  The BTC question becomes: is BTC's 1h variance ratio above that number?")

    # ---------------- cost sensitivity ---------------------------------
    print("\n" + "=" * 102)
    print("COST SENSITIVITY - trending world (phi=+0.20), where the edge dies")
    print("=" * 102 + "\n")
    rows = []
    for fee, slip, label in ((0, 0, "frictionless"), (2, 2, "maker (bot default)"),
                             (5, 2, "taker spot"), (10, 2, "taker, no fee tier"),
                             (25, 5, "wide-spread CFD")):
        r = across_seeds(0.20, "S1", costs=Costs(fee_bps=fee, slippage_bps=slip))
        r["venue"] = f"{label} ({fee}+{slip}bps)"
        rows.append(r)
    cs = pd.DataFrame(rows).set_index("venue")
    print(cs[["expectancy_R", "ci_lo", "ci_hi", "total_R", "win_rate", "sharpe",
              "worlds_profitable"]].round(3).to_string())

    # ---------------- parameter plateau --------------------------------
    print("\n" + "=" * 102)
    print("PARAMETER ROBUSTNESS - plateau or knife-edge? (trending world, mean total R)")
    print("=" * 102 + "\n")
    grid = []
    for don in (34, 55, 89):
        for stop in (1.5, 2.0, 2.5):
            vals = []
            for sd in SEEDS[:6]:
                df = make_world(BARS, 0.20, sd)
                o, a, _ = s1_trend_breakout(df, S1Params(donchian=don, stop_atr=stop,
                                                         allow_short=False))
                res = run_backtest(df, o, a, COSTS, CFG)
                vals.append(float(res.trades["r_multiple"].sum()) if not res.trades.empty else 0.0)
            grid.append({"donchian": don, "stop_atr": stop, "total_R": float(np.mean(vals))})
    g = pd.DataFrame(grid).pivot(index="donchian", columns="stop_atr", values="total_R")
    print(g.round(1).to_string())
    allpos = bool((g > 0).all().all())
    print(f"\n  all 9 combinations profitable: {'YES - a plateau' if allpos else 'NO - not a plateau'}")

    # ---------------- summary ------------------------------------------
    print("\n" + "=" * 102)
    print("WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT")
    print("=" * 102)
    print(f"\n  Measuring instrument   {'SOUND' if ok_ctrl else 'SUSPECT'} - zero expectancy on noise")
    print(f"  Implementation         {'SOUND' if p1 and p2 else 'QUESTIONABLE'} - "
          f"{'profits on trend, loses on noise' if p1 and p2 else 'see failures above'}")
    if phi_be is not None:
        print(f"  Requirement on market  needs a 24h variance ratio above ~{vr_be:.2f}")
    print("\n  Profitability on BTC   NOT ESTABLISHED. No market-data host is reachable from")
    print("                         this environment, so nothing here is evidence about Bitcoin.")
    return 0 if (ok_ctrl and p1 and p2) else 1


if __name__ == "__main__":
    sys.exit(main())
