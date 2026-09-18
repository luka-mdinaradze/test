"""Bar-by-bar backtest engine with realistic frictions.

Design rules (these are the ones that decide whether a backtest tells the truth):

1. No lookahead. A signal computed from bar t's close is executed at bar t+1's
   OPEN. Trailing stops updated at bar t's close only apply from bar t+1.
2. Pessimistic intrabar fills. If a bar's range touches both the stop and the
   take-profit, the STOP is assumed to have hit first. Real 1h bars often do
   both; assuming the good one first is the single most common way a retail
   backtest inflates itself.
3. Gaps are honoured. If a bar opens through the stop, the fill is the open,
   not the stop price.
4. Costs are charged on both sides: exchange fee + slippage, on notional.
5. Exchange minimum notional is enforced, and rejected orders are counted
   rather than silently skipped. On a $100 account this matters enormously.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import numpy as np
import pandas as pd

Side = Literal["long", "short"]


@dataclass
class Costs:
    """All-in transaction costs, in basis points of notional, per side."""
    fee_bps: float = 5.0           # Binance spot taker ~10bps, USD-M futures taker ~4.5bps
    slippage_bps: float = 2.0      # market-order slippage on a liquid 1h BTC book
    funding_bps_per_8h: float = 1.0  # perp funding drag, charged to the held side

    @property
    def per_side_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass
class ExecConfig:
    initial_equity: float = 100.0
    risk_pct: float = 0.01          # fraction of equity risked between entry and stop
    max_leverage: float = 3.0       # cap on notional / equity
    min_notional: float = 5.0       # exchange minimum order size in quote currency
    qty_step: float = 1e-6          # lot rounding
    bars_per_year: int = 8760       # 1h bars
    allow_short: bool = True
    apply_funding: bool = False     # set True when modelling perps


@dataclass
class Order:
    """An entry intent produced at bar t, to be filled at bar t+1's open."""
    side: Side
    stop_dist: float                # distance from entry to stop, in price units
    tp_dist: Optional[float] = None  # distance from entry to take-profit
    trail_atr_mult: Optional[float] = None
    max_bars: Optional[int] = None
    tag: str = ""


@dataclass
class Trade:
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    stop_price: float
    risk_amount: float
    pnl: float                      # net of all costs
    costs: float
    r_multiple: float               # net pnl / risk_amount
    bars_held: int
    reason: str
    tag: str = ""


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    rejected_orders: int
    config: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().fillna(0.0)


def _round_qty(q: float, step: float) -> float:
    return np.floor(q / step) * step if step > 0 else q


def run_backtest(
    df: pd.DataFrame,
    orders: pd.Series,
    atr_series: Optional[pd.Series] = None,
    costs: Costs = Costs(),
    cfg: ExecConfig = ExecConfig(),
) -> BacktestResult:
    """Execute a stream of entry intents.

    df:          OHLCV frame, datetime index, ascending.
    orders:      Series aligned to df.index holding Order objects (or None) —
                 the intent formed at that bar's close.
    atr_series:  ATR used for trailing stops (required if any order trails).
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"df needs columns {required}, got {set(df.columns)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted ascending")

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)
    atr_v = atr_series.to_numpy(float) if atr_series is not None else np.full(n, np.nan)
    order_list = orders.reindex(idx).tolist()

    equity = cfg.initial_equity
    equity_curve = np.empty(n)
    trades: list[Trade] = []
    rejected = 0

    # Open position state
    in_pos = False
    side: Side = "long"
    entry_px = stop_px = qty = risk_amt = 0.0
    init_stop_px = 0.0        # the stop as first placed, so trail exits are distinguishable
    tp_px: Optional[float] = None
    trail_mult: Optional[float] = None
    max_bars: Optional[int] = None
    entry_i = 0
    entry_cost = 0.0
    tag = ""
    pending: Optional[Order] = None

    slip = costs.per_side_bps / 10_000.0

    for i in range(n):
        # ---- 1. Fill any pending entry at THIS bar's open -------------------
        if pending is not None and not in_pos:
            raw = o[i]
            fill = raw * (1 + slip) if pending.side == "long" else raw * (1 - slip)
            sd = pending.stop_dist
            if sd > 0 and np.isfinite(sd):
                risk_amt_try = equity * cfg.risk_pct
                q = risk_amt_try / sd
                notional = q * fill
                cap = equity * cfg.max_leverage
                if notional > cap:                       # leverage cap binds
                    q = cap / fill
                    notional = cap
                q = _round_qty(q, cfg.qty_step)
                notional = q * fill
                if q > 0 and notional >= cfg.min_notional:
                    in_pos = True
                    side = pending.side
                    entry_px = fill
                    qty = q
                    # Realised risk can be < intended if the leverage cap bound.
                    risk_amt = q * sd
                    stop_px = fill - sd if side == "long" else fill + sd
                    init_stop_px = stop_px
                    tp_px = (fill + pending.tp_dist if side == "long" else fill - pending.tp_dist) \
                        if pending.tp_dist else None
                    trail_mult = pending.trail_atr_mult
                    max_bars = pending.max_bars
                    entry_i = i
                    entry_cost = notional * costs.fee_bps / 10_000.0
                    tag = pending.tag
                else:
                    rejected += 1
            else:
                rejected += 1
            pending = None

        # ---- 2. Manage an open position against THIS bar's range ------------
        if in_pos:
            exit_px: Optional[float] = None
            reason = ""
            trailed = abs(stop_px - init_stop_px) > 1e-12
            stop_label = "trail_stop" if trailed else "stop"
            if side == "long":
                if o[i] <= stop_px:                       # gapped through the stop
                    exit_px, reason = o[i], "gap_stop"
                elif lo[i] <= stop_px:                    # stop before target (pessimistic)
                    exit_px, reason = stop_px, stop_label
                elif tp_px is not None and h[i] >= tp_px:
                    exit_px, reason = tp_px, "take_profit"
            else:
                if o[i] >= stop_px:
                    exit_px, reason = o[i], "gap_stop"
                elif h[i] >= stop_px:
                    exit_px, reason = stop_px, stop_label
                elif tp_px is not None and lo[i] <= tp_px:
                    exit_px, reason = tp_px, "take_profit"

            if exit_px is None and max_bars is not None and (i - entry_i) >= max_bars:
                exit_px, reason = c[i], "time_exit"

            if exit_px is not None:
                fill = exit_px * (1 - slip) if side == "long" else exit_px * (1 + slip)
                gross = (fill - entry_px) * qty if side == "long" else (entry_px - fill) * qty
                exit_cost = abs(fill * qty) * costs.fee_bps / 10_000.0
                fund = 0.0
                if cfg.apply_funding:
                    hours = max(i - entry_i, 1)
                    fund = abs(entry_px * qty) * (costs.funding_bps_per_8h / 10_000.0) * (hours / 8.0)
                total_cost = entry_cost + exit_cost + fund
                pnl = gross - total_cost
                equity += pnl
                trades.append(
                    Trade(
                        side=side, entry_time=idx[entry_i], exit_time=idx[i],
                        entry_price=entry_px, exit_price=fill, qty=qty, stop_price=stop_px,
                        risk_amount=risk_amt, pnl=pnl, costs=total_cost,
                        r_multiple=pnl / risk_amt if risk_amt > 0 else 0.0,
                        bars_held=i - entry_i, reason=reason, tag=tag,
                    )
                )
                in_pos = False
                qty = 0.0

        # ---- 3. Mark to market ---------------------------------------------
        if in_pos:
            unreal = (c[i] - entry_px) * qty if side == "long" else (entry_px - c[i]) * qty
            equity_curve[i] = equity + unreal
        else:
            equity_curve[i] = equity

        if equity <= 0:                                   # account blown
            equity_curve[i:] = max(equity, 0.0)
            break

        # ---- 4. Update trailing stop (effective from the NEXT bar) ----------
        if in_pos and trail_mult is not None and np.isfinite(atr_v[i]):
            if side == "long":
                stop_px = max(stop_px, c[i] - trail_mult * atr_v[i])
            else:
                stop_px = min(stop_px, c[i] + trail_mult * atr_v[i])

        # ---- 5. Take this bar's intent, to be filled next bar ---------------
        if not in_pos and pending is None:
            intent = order_list[i]
            if isinstance(intent, Order) and i + 1 < n:
                if intent.side == "short" and not cfg.allow_short:
                    intent = None
                pending = intent

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    cfg_dump = {**asdict(cfg), **{f"cost_{k}": v for k, v in asdict(costs).items()}}
    return BacktestResult(
        equity=pd.Series(equity_curve, index=idx, name="equity"),
        trades=trades_df,
        rejected_orders=rejected,
        config=cfg_dump,
    )
