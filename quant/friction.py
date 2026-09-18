"""Transaction-cost arithmetic for fixed-fractional risk sizing.

This is the part of the problem that does NOT depend on price history, so it
can be settled exactly.

With risk-based sizing, notional is set by the stop distance:

    notional = risk_amount / stop_pct

so the round-trip cost, expressed in units of R (the amount risked), is:

    cost_R = 2 * (fee_bps + slippage_bps) / 10_000 / stop_pct

The stop distance therefore controls cost, and it does so HYPERBOLICALLY: halve
the stop and you double the fee burden per unit of risk. This is why tight
stops on 1h crypto quietly destroy otherwise sound systems.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cost_in_R(stop_pct: float, per_side_bps: float, round_trips: int = 2) -> float:
    """Round-trip friction as a fraction of the amount risked."""
    return round_trips * (per_side_bps / 10_000.0) / (stop_pct / 100.0)


def breakeven_winrate(payoff: float, cost_R: float = 0.0) -> float:
    """Win rate needed for zero expectancy at a given payoff ratio, after costs.

    Expectancy = p*(payoff - cost_R) - (1-p)*(1 + cost_R) = 0
    """
    num = 1.0 + cost_R
    den = payoff - cost_R + 1.0 + cost_R
    return num / den if den > 0 else np.nan


def friction_table(stop_pcts=(0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
                   per_side_bps: float = 7.0, payoff: float = 2.0,
                   equity: float = 100.0, risk_pct: float = 0.01,
                   min_notional: float = 5.0, max_lev: float = 3.0) -> pd.DataFrame:
    """Cost, break-even win rate and tradeability across stop widths."""
    rows = []
    risk_amt = equity * risk_pct
    for sp in stop_pcts:
        c = cost_in_R(sp, per_side_bps)
        notional = risk_amt / (sp / 100.0)
        lev = notional / equity
        capped = lev > max_lev
        eff_notional = min(notional, equity * max_lev)
        rows.append({
            "stop_%_of_price": sp,
            "notional_$": round(notional, 2),
            "implied_leverage": round(lev, 2),
            "cost_per_trade_R": round(c, 3),
            "cost_per_trade_$": round(c * risk_amt, 4),
            "breakeven_winrate_%": round(breakeven_winrate(payoff, c) * 100, 1),
            "breakeven_winrate_nocost_%": round(breakeven_winrate(payoff, 0.0) * 100, 1),
            "tradeable": "yes" if eff_notional >= min_notional and not capped
                         else ("leverage-capped" if capped else "below min notional"),
        })
    return pd.DataFrame(rows)


def edge_required(stop_pct: float, per_side_bps: float, trades_per_year: float,
                  risk_pct: float = 0.01) -> dict:
    """How much gross edge the signal must carry just to pay the bill."""
    c = cost_in_R(stop_pct, per_side_bps)
    annual_cost_R = c * trades_per_year
    return {
        "cost_per_trade_R": c,
        "trades_per_year": trades_per_year,
        "annual_cost_R": annual_cost_R,
        "annual_cost_pct_of_equity": annual_cost_R * risk_pct * 100,
        "gross_expectancy_needed_R": c,
    }
