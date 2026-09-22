"""Risk gates. The bot asks permission here before it is allowed to do anything.

Design principle: every gate FAILS CLOSED. If a check cannot be evaluated, the
answer is no. A bot that trades when it is confused is how $100 becomes $0.

Two kinds of stop:
  - SOFT (daily loss limit, loss-streak de-risk): pause or shrink, auto-clears.
  - HARD (max drawdown, consecutive losses, API error storm): halts the bot and
    requires a human to reset it. Deliberately not self-clearing — if the model
    is broken, letting it resume on a timer just finishes the job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import TradingConfig
from .state import BotState


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    risk_pct: float = 0.0
    qty: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0


def effective_risk_pct(cfg: TradingConfig, st: BotState) -> float:
    """Risk fraction after probation and loss-streak de-risking."""
    r = cfg.risk_pct
    if st.phase == "LIVE" and st.live_trades_done < cfg.probation_trades:
        r = min(r, cfg.probation_risk_pct)      # earn full size, don't assume it
    if st.consecutive_losses >= cfg.loss_streak_derisk:
        r *= 0.5                                # de-risk INTO a losing streak
    return r


def check_halts(cfg: TradingConfig, st: BotState) -> Optional[str]:
    """Hard stops. Returns a reason string if the bot must halt, else None."""
    if st.drawdown <= -cfg.max_drawdown_pct:
        return (f"max drawdown breached: {st.drawdown*100:.1f}% "
                f"(limit {-cfg.max_drawdown_pct*100:.0f}%)")
    if st.consecutive_losses >= cfg.max_consecutive_losses:
        return f"{st.consecutive_losses} consecutive losses (limit {cfg.max_consecutive_losses})"
    if st.api_error_count >= cfg.max_api_errors:
        return f"{st.api_error_count} consecutive API errors (limit {cfg.max_api_errors})"
    return None


def can_open(cfg: TradingConfig, st: BotState, entry: float, stop: float,
             equity: float) -> RiskDecision:
    """Full pre-trade gate. Returns the size if allowed, the reason if not."""
    if st.halted:
        return RiskDecision(False, f"bot halted: {st.halt_reason}")

    halt = check_halts(cfg, st)
    if halt:
        return RiskDecision(False, f"halt condition: {halt}")

    if st.position is not None:
        return RiskDecision(False, "position already open")
    if st.pending_entry is not None:
        return RiskDecision(False, "entry order already working")

    # Soft daily stop: pause until the next UTC day.
    if st.day_realised_R <= -cfg.daily_loss_limit_R:
        return RiskDecision(False, f"daily loss limit hit ({st.day_realised_R:.2f}R)")

    if equity <= 0:
        return RiskDecision(False, "no equity")

    dist = abs(entry - stop)
    if dist <= 0 or not (entry > 0 and stop > 0):
        return RiskDecision(False, f"invalid levels: entry={entry}, stop={stop}")

    rpct = effective_risk_pct(cfg, st)
    risk_amount = equity * rpct
    qty = risk_amount / dist
    notional = qty * entry

    cap = equity * cfg.max_leverage
    if notional > cap:
        # Spot with no borrowing: shrink to what the cash actually buys. Real
        # risk then falls BELOW the target, which is the safe direction.
        qty = cap / entry
        notional = cap
        risk_amount = qty * dist

    step = cfg.qty_step
    qty = int(qty / step) * step if step > 0 else qty
    notional = qty * entry
    risk_amount = qty * dist

    if qty <= 0:
        return RiskDecision(False, "size rounds to zero")
    if notional < cfg.min_notional:
        return RiskDecision(False, f"notional ${notional:.2f} below venue minimum "
                                   f"${cfg.min_notional:.2f}")
    if notional > equity * cfg.max_leverage + 1e-9:
        return RiskDecision(False, "leverage cap breached")

    return RiskDecision(True, "ok", rpct, qty, notional, risk_amount)


def promotion_ready(cfg: TradingConfig, st: BotState) -> tuple[bool, str]:
    """Is the paper record good enough to go live? The bot never decides alone —
    this only reports; a human passes --promote."""
    paper = [t for t in st.trades if t.get("phase") == "PAPER"]
    n = len(paper)
    if n < cfg.min_paper_trades:
        return False, f"{n}/{cfg.min_paper_trades} paper trades"
    exp = sum(t.get("r_multiple", 0.0) for t in paper) / n
    if exp < cfg.min_paper_expectancy_R:
        return False, f"paper expectancy {exp:.3f}R below required {cfg.min_paper_expectancy_R:.3f}R"
    slips = [abs(t.get("slippage_bps", 0.0)) for t in paper if "slippage_bps" in t]
    if slips:
        avg = sum(slips) / len(slips)
        if avg > cfg.max_slippage_bps_observed:
            return False, f"observed slippage {avg:.1f}bps exceeds {cfg.max_slippage_bps_observed:.1f}bps"
    return True, f"{n} paper trades, expectancy {exp:+.3f}R"
