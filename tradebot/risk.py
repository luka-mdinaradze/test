"""Position sizing and the risk governor.

Sizing: quantity = floor(equity x risk% / risk-per-unit), where risk-per-unit is
the stop distance plus exit slippage, commissions and fees. If even the minimum
tradeable size exceeds the budget, the size is ZERO and the trade is refused —
never rounded up. Nothing here can increase size after a loss (no martingale).

Governor: blocks new entries after the daily loss limit, N consecutive losses
or too many trades in a day (all reset the next day), and HARD-halts at the
maximum drawdown or when the kill switch file exists (a human must clear it).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import AccountSection, InstrumentSection


@dataclass(frozen=True)
class Sizing:
    qty: float
    risk_amount: float      # $ lost if the stop fills as assumed
    per_unit: float         # $ risk of one unit
    budget: float
    note: str = ""


def adverse(inst: InstrumentSection, price: float) -> float:
    return inst.slippage_ticks * inst.tick_size + inst.slippage_bps * 1e-4 * price


def risk_per_unit(stop_distance: float, price: float, inst: InstrumentSection) -> float:
    return ((stop_distance + adverse(inst, price)) * inst.point_value
            + 2 * inst.commission_per_unit + 2 * inst.fee_bps * 1e-4 * price * inst.point_value)


def size_position(equity: float, stop_distance: float, price: float,
                  inst: InstrumentSection, acct: AccountSection) -> Sizing:
    per_unit = risk_per_unit(stop_distance, price, inst)
    budget = equity * acct.risk_pct / 100.0
    steps = math.floor(budget / per_unit / inst.qty_step + 1e-9)
    qty = min(round(steps * inst.qty_step, 10), inst.max_qty)
    if qty < inst.min_qty - 1e-12:
        return Sizing(0.0, 0.0, per_unit, budget,
                      f"one minimum unit ({inst.min_qty:g}) risks ${per_unit * inst.min_qty:,.2f} "
                      f"> budget ${budget:,.2f} ({acct.risk_pct:g}% of ${equity:,.0f}) — trade refused")
    risk = qty * per_unit
    if risk > equity * acct.max_risk_pct / 100.0 + 1e-9:          # unreachable while risk_pct <= max_risk_pct
        return Sizing(0.0, 0.0, per_unit, budget, "risk would exceed max_risk_pct — trade refused")
    return Sizing(qty, risk, per_unit, budget)


@dataclass
class GovernorState:
    day: str = ""
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    paused_reason: str = ""       # lifted at the next trading day
    halted_reason: str = ""       # needs a human: `python3 -m tradebot reset-halt`

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Governor:
    acct: AccountSection
    state: GovernorState = field(default_factory=GovernorState)
    kill_file: Path | None = None

    def roll_day(self, day: str, equity: float) -> None:
        s = self.state
        if s.day != day:
            s.day, s.day_start_equity, s.day_pnl, s.trades_today = day, equity, 0.0, 0
            s.consecutive_losses, s.paused_reason = 0, ""
        s.peak_equity = max(s.peak_equity, equity)
        dd = 100 * (1 - equity / s.peak_equity) if s.peak_equity > 0 else 0.0
        if dd >= self.acct.max_drawdown_pct and not s.halted_reason:
            s.halted_reason = f"max drawdown {dd:.1f}% >= {self.acct.max_drawdown_pct:g}% from peak ${s.peak_equity:,.2f}"

    def killed(self) -> bool:
        return bool(self.kill_file and self.kill_file.exists())

    def allows_entry(self) -> tuple[bool, str]:
        s = self.state
        if self.killed():
            return False, f"kill switch present ({self.kill_file})"
        if s.halted_reason:
            return False, "HALTED: " + s.halted_reason
        if s.paused_reason:
            return False, "paused for today: " + s.paused_reason
        if s.trades_today >= self.acct.max_trades_per_day:
            return False, f"max {self.acct.max_trades_per_day} trades per day reached"
        return True, ""

    def record_trade(self, pnl: float) -> None:
        s = self.state
        s.day_pnl += pnl
        s.trades_today += 1
        s.consecutive_losses = s.consecutive_losses + 1 if pnl < 0 else 0
        if s.day_start_equity > 0 and s.day_pnl <= -self.acct.daily_loss_limit_pct / 100 * s.day_start_equity:
            s.paused_reason = f"daily loss {s.day_pnl:,.2f} reached -{self.acct.daily_loss_limit_pct:g}%"
        elif s.consecutive_losses >= self.acct.max_consecutive_losses:
            s.paused_reason = f"{s.consecutive_losses} consecutive losses"
