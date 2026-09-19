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


# --------------------------------------------------------------------------
# Venue comparison: exchange spot vs perpetual futures vs CFD
# --------------------------------------------------------------------------
def venue_cost_R(stop_pct: float, hold_hours: float, spread_bps: float = 0.0,
                 commission_bps_per_side: float = 0.0, slippage_bps: float = 2.0,
                 carry_pct_per_day: float = 0.0) -> dict:
    """Total round-trip cost in R for one venue.

    stop_pct            stop distance as % of price (sets notional = risk/stop%)
    spread_bps          full bid/ask spread, paid once per round trip
    commission_bps      per side
    carry_pct_per_day   overnight financing / swap / funding, % of NOTIONAL per day

    Carry is the term retail traders forget. Because notional = risk / stop%,
    a daily carry of c% costs c * days / stop% in R — a 2-day hold on a 2.5%
    stop with 0.05%/day financing is 0.04R, but the same carry on a 0.8% stop
    held a week is 0.44R, which is most of an average edge.
    """
    days = hold_hours / 24.0
    entry_exit_bps = spread_bps + 2 * commission_bps_per_side + 2 * slippage_bps
    trade_cost_R = (entry_exit_bps / 10_000.0) / (stop_pct / 100.0)
    carry_R = (carry_pct_per_day / 100.0) * days / (stop_pct / 100.0)
    return {"trade_cost_R": trade_cost_R, "carry_R": carry_R,
            "total_R": trade_cost_R + carry_R}


def compare_venues(stop_pct: float, hold_hours: float, trades_per_year: float,
                   venues: dict | None = None, risk_pct: float = 0.01) -> "pd.DataFrame":
    """Side-by-side venue costs for a given strategy shape.

    Default venue parameters are ILLUSTRATIVE RANGES, not quotes. Replace them
    with your broker's actual spread and swap table before drawing conclusions.
    """
    venues = venues or {
        "Binance spot (taker)":      dict(spread_bps=0, commission_bps_per_side=10.0, carry_pct_per_day=0.0),
        "Binance spot (maker)":      dict(spread_bps=0, commission_bps_per_side=2.0, carry_pct_per_day=0.0),
        "Perp future (taker+funding)": dict(spread_bps=0, commission_bps_per_side=4.5, carry_pct_per_day=0.03),
        "CFD - tight broker":        dict(spread_bps=5.0, commission_bps_per_side=0.0, carry_pct_per_day=0.03),
        "CFD - typical broker":      dict(spread_bps=25.0, commission_bps_per_side=0.0, carry_pct_per_day=0.06),
        "CFD - wide broker":         dict(spread_bps=60.0, commission_bps_per_side=0.0, carry_pct_per_day=0.10),
    }
    rows = []
    for name, v in venues.items():
        c = venue_cost_R(stop_pct, hold_hours, **v)
        rows.append({
            "venue": name,
            "trade_cost_R": round(c["trade_cost_R"], 3),
            "carry_R": round(c["carry_R"], 3),
            "total_R": round(c["total_R"], 3),
            "annual_R": round(c["total_R"] * trades_per_year, 1),
            "annual_%_of_equity": round(c["total_R"] * trades_per_year * risk_pct * 100, 1),
        })
    return pd.DataFrame(rows)
