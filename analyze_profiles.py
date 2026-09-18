#!/usr/bin/env python3
"""Assumption-driven risk analysis for the three strategy archetypes.

No price history is used. Given a trade-outcome profile (win rate, payoff) and
the friction implied by the stop width, this computes expectancy, break-even
requirements, Kelly, and the full Monte Carlo path distribution.

Change the profiles to match YOUR measured backtest and re-run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import montecarlo
from quant.friction import breakeven_winrate, cost_in_R

PER_SIDE_BPS = 7.0     # 5bps taker + 2bps slippage
RISK_PCT = 0.01
EQUITY = 100.0

# Archetype assumptions: what each design TYPICALLY produces on 1h crypto.
# These are priors for sizing the risk envelope, not backtest results.
PROFILES = {
    "S1 trend breakout":   dict(win=0.33, win_R=3.0, stop_pct=2.5, trades_yr=180),
    "S2 mean reversion":   dict(win=0.62, win_R=0.9, stop_pct=1.2, trades_yr=320),
    "S3 squeeze expansion": dict(win=0.30, win_R=3.0, stop_pct=1.6, trades_yr=140),
}

rows, mc_rows = [], []
for name, p in PROFILES.items():
    c = cost_in_R(p["stop_pct"], PER_SIDE_BPS)
    gross_exp = p["win"] * p["win_R"] - (1 - p["win"]) * 1.0
    net_exp = p["win"] * (p["win_R"] - c) - (1 - p["win"]) * (1 + c)
    be = breakeven_winrate(p["win_R"], c) * 100
    kelly = (p["win"] * p["win_R"] - (1 - p["win"])) / p["win_R"] if p["win_R"] > 0 else np.nan
    rows.append({
        "strategy": name, "win_rate_%": p["win"] * 100, "payoff_R": p["win_R"],
        "stop_%": p["stop_pct"], "cost_R": round(c, 3),
        "gross_exp_R": round(gross_exp, 3), "net_exp_R": round(net_exp, 3),
        "cost_eats_%_of_edge": round((gross_exp - net_exp) / gross_exp * 100, 1) if gross_exp > 0 else np.nan,
        "breakeven_WR_%": round(be, 1), "actual_minus_BE": round(p["win"] * 100 - be, 1),
        "full_kelly_%": round(kelly * 100, 1),
        "annual_net_R": round(net_exp * p["trades_yr"], 1),
        "annual_return_%_at_1%_risk": round(net_exp * p["trades_yr"] * RISK_PCT * 100, 1),
    })

    rng = np.random.default_rng(11)
    r = np.where(rng.random(20000) < p["win"], p["win_R"] - c, -(1 + c))
    mc = montecarlo.simulate(r, n_trades=300, n_sims=20000, risk_pct=RISK_PCT,
                             initial_equity=EQUITY, seed=7)
    mc_rows.append({
        "strategy": name, "P(end below $100)_%": round(mc["prob_loss"] * 100, 1),
        "P(lose half)_%": round(mc["prob_ruin"] * 100, 2),
        "median_$": round(mc["final_p50"], 2), "p5_$": round(mc["final_p5"], 2),
        "p95_$": round(mc["final_p95"], 2), "worst_$": round(mc["final_worst"], 2),
        "median_maxDD_%": round(mc["maxdd_p50"], 1), "p95_maxDD_%": round(mc["maxdd_p95"], 1),
        "worst_streak": mc["streak_worst"], "p95_streak": int(mc["streak_p95"]),
    })

pd.set_option("display.width", 250)
print("EXPECTANCY AFTER FRICTION (7bps/side, risk-based sizing)\n")
print(pd.DataFrame(rows).to_string(index=False))
print("\n\nMONTE CARLO, 300 trades at 1% risk from $100 (20,000 paths)\n")
print(pd.DataFrame(mc_rows).to_string(index=False))

print("\n\nRISK-PER-TRADE SENSITIVITY - S1 profile, 300 trades\n")
p = PROFILES["S1 trend breakout"]
c = cost_in_R(p["stop_pct"], PER_SIDE_BPS)
rng = np.random.default_rng(11)
r = np.where(rng.random(20000) < p["win"], p["win_R"] - c, -(1 + c))
sens = []
for rp in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
    mc = montecarlo.simulate(r, 300, 20000, rp, EQUITY, seed=7)
    sens.append({"risk_per_trade_%": rp * 100, "median_$": round(mc["final_p50"], 2),
                 "p5_$": round(mc["final_p5"], 2), "P(loss)_%": round(mc["prob_loss"] * 100, 1),
                 "P(lose half)_%": round(mc["prob_ruin"] * 100, 2),
                 "median_maxDD_%": round(mc["maxdd_p50"], 1),
                 "worst_maxDD_%": round(mc["maxdd_worst"], 1)})
print(pd.DataFrame(sens).to_string(index=False))
print("\nNote: median equity peaks then falls as risk rises - that is the volatility drag /")
print("over-betting effect. Past the optimal fraction, MORE risk produces LESS money.")
