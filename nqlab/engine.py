"""Trade simulation on 1-minute bars, whatever timeframe the signal came from.

Execution rules — each one is covered by a test in tests/test_nqlab_engine.py:

1. A signal formed at the close of 1-minute bar t (the last minute of its
   timeframe bar) is filled at the OPEN of bar t+1, plus adverse slippage.
   Never at bar t's own close.
2. Stops and targets are resolved on 1-minute bars even for 1h signals. If one
   1-minute bar touches both, the STOP is booked: the order inside a minute is
   unknowable from OHLC, so the conservative outcome is assumed.
3. A bar that OPENS through the stop fills at that open (minus slippage), so a
   gap costs more than 1R — the engine never pretends a stop is a guarantee.
4. A take-profit is a resting limit: it fills at its price only when the market
   trades THROUGH it by `tp_through` (queue-position proxy). Touching is not a fill.
5. Stop and time exits are market orders and pay slippage. Positions are closed
   at the close of the day's last pre-flatten bar; nothing is held overnight.
6. One position at a time. Signals arriving while in a position are counted as
   `overlap` and dropped — never queued, never pyramided.
7. Size = floor(equity x risk% / risk-per-unit), where risk-per-unit is the stop
   distance plus the exit slippage, commissions and fees. A size below the
   instrument minimum is REJECTED and counted. Risk never exceeds max_risk%.
8. After the daily loss limit or N consecutive losses, no new entries that day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from .config import Account, Instrument

STOP, TARGET, TIME, FLATTEN, GAP_STOP = 1, 2, 3, 4, 5
REASONS = {STOP: "stop", TARGET: "target", TIME: "time", FLATTEN: "flatten", GAP_STOP: "gap_stop"}


@dataclass
class Signals:
    """Entry intents at 1-minute resolution.

    idx    positional index of the 1-minute bar whose CLOSE formed the signal
    side   +1 long / -1 short
    stop   stop distance in points (> 0)
    tp     target distance in points (<= 0 means no fixed target)
    hold   maximum holding time in 1-minute bars
    """
    idx: np.ndarray
    side: np.ndarray
    stop: np.ndarray
    tp: np.ndarray
    hold: np.ndarray
    meta: pd.DataFrame | None = None   # features at signal time, row-aligned

    def __post_init__(self):
        self.idx = np.asarray(self.idx, np.int64)
        self.side = np.asarray(self.side, np.int64)
        self.stop = np.asarray(self.stop, np.float64)
        self.tp = np.asarray(self.tp, np.float64)
        self.hold = np.asarray(self.hold, np.int64)
        order = np.argsort(self.idx, kind="stable")
        if not np.all(order == np.arange(len(order))):
            self.idx, self.side, self.stop, self.tp, self.hold = (
                a[order] for a in (self.idx, self.side, self.stop, self.tp, self.hold))
            if self.meta is not None:
                self.meta = self.meta.iloc[order].reset_index(drop=True)
        if len(self.stop) and not np.all(self.stop > 0):
            raise ValueError("every stop distance must be > 0")

    def __len__(self):
        return len(self.idx)

    def subset(self, mask: np.ndarray) -> "Signals":
        meta = None if self.meta is None else self.meta[mask].reset_index(drop=True)
        return Signals(self.idx[mask], self.side[mask], self.stop[mask], self.tp[mask],
                       self.hold[mask], meta)


@dataclass
class Market1m:
    """Arrays the simulator needs, precomputed once per (data, instrument)."""
    index: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    tday: np.ndarray
    flatten: np.ndarray
    entry_ok: np.ndarray
    adverse: np.ndarray      # slippage + half-spread per side, points
    through: np.ndarray      # trade-through required for a limit fill, points
    bucket: np.ndarray
    rth: np.ndarray

    @classmethod
    def build(cls, m1: pd.DataFrame, inst: Instrument) -> "Market1m":
        c = m1["close"].to_numpy(float)
        rth = m1["rth"].to_numpy(bool)
        ticks = np.where(rth, inst.slip_ticks_rth, inst.slip_ticks_eth)
        half = np.where(rth, inst.half_spread_rth, inst.half_spread_eth)
        adverse = ticks * inst.tick_size + inst.slip_bps_side * 1e-4 * c + half
        through = inst.tp_through_ticks * inst.tick_size + inst.tp_through_bps * 1e-4 * c + half
        return cls(m1.index, m1["open"].to_numpy(float), m1["high"].to_numpy(float),
                   m1["low"].to_numpy(float), c, m1["tday"].to_numpy(np.int64),
                   m1["flatten"].to_numpy(bool), m1["entry_ok"].to_numpy(bool),
                   adverse.astype(float), through.astype(float),
                   m1["bucket"].to_numpy(np.int64), rth)


@njit(cache=True)
def _resolve(i0, side, fill, stop_px, tp_px, hold, o, h, l, c, flatten, adverse, through):
    """Walk forward from the entry bar to the first exit. Returns (bar, price, reason)."""
    n = len(o)
    for i in range(i0, n):
        if i > i0:
            if side > 0 and o[i] <= stop_px:
                return i, o[i] - adverse[i], GAP_STOP
            if side < 0 and o[i] >= stop_px:
                return i, o[i] + adverse[i], GAP_STOP
        if side > 0:
            hit_sl = l[i] <= stop_px
            hit_tp = h[i] >= tp_px + through[i]
        else:
            hit_sl = h[i] >= stop_px
            hit_tp = l[i] <= tp_px - through[i]
        if hit_sl:  # also wins any tie with the target: conservative
            return i, stop_px - side * adverse[i], STOP
        if hit_tp:
            return i, tp_px, TARGET
        if i - i0 + 1 >= hold:
            return i, c[i] - side * adverse[i], TIME
        if flatten[i]:
            return i, c[i] - side * adverse[i], FLATTEN
    return n - 1, c[n - 1] - side * adverse[n - 1], FLATTEN


@njit(cache=True)
def _risk_unit(stop_pts, fill, exit_adverse, pv, comm, fee_bps):
    """$ lost per unit if the stop fills as assumed: distance + exit costs + both commissions/fees."""
    return (stop_pts + exit_adverse) * pv + 2.0 * comm + 2.0 * fee_bps * 1e-4 * fill * pv


@njit(cache=True)
def _simulate(sig_idx, sig_side, sig_stop, sig_tp, sig_hold,
              o, h, l, c, tday, flatten, entry_ok, adverse, through,
              pv, comm, fee_bps, qty_step, min_qty, max_qty,
              eq0, risk_pct, max_risk_pct, fractional, compounding,
              daily_loss_pct, max_consec):
    n = len(o)
    m = len(sig_idx)
    out_k = np.full(m, -1, np.int64)
    out_entry = np.zeros(m, np.int64)
    out_exit = np.zeros(m, np.int64)
    out_fill = np.zeros(m)
    out_px = np.zeros(m)
    out_qty = np.zeros(m)
    out_pnl = np.zeros(m)
    out_risk = np.zeros(m)
    out_cost = np.zeros(m)
    out_reason = np.zeros(m, np.int64)
    out_eq = np.zeros(m)
    counts = np.zeros(5, np.int64)  # overlap, session, rejected_size, blocked_risk, filled
    equity = eq0
    free_from = -1
    day = -1
    day_pnl = 0.0
    day_eq = eq0
    consec = 0
    blocked = False
    nt = 0
    for k in range(m):
        t = sig_idx[k]
        i = t + 1
        if i >= n:
            break
        if t < free_from:
            counts[0] += 1
            continue
        if tday[i] != tday[t] or not entry_ok[i] or flatten[i]:
            counts[1] += 1
            continue
        if tday[i] != day:
            day = tday[i]
            day_pnl = 0.0
            day_eq = equity
            consec = 0
            blocked = False
        if blocked:
            counts[3] += 1
            continue
        side = sig_side[k]
        fill = o[i] + side * adverse[i]
        ru = _risk_unit(sig_stop[k], fill, adverse[i], pv, comm, fee_bps)
        base = eq0 if (fractional or not compounding) else equity
        budget = base * risk_pct
        if fractional:
            qty = budget / ru
        else:
            qty = np.floor(budget / ru / qty_step + 1e-9) * qty_step
            if qty > max_qty:
                qty = max_qty
            if qty < min_qty - 1e-12:
                counts[2] += 1
                continue
            if qty * ru > base * max_risk_pct + 1e-9:   # unreachable by construction; kept as a guard
                counts[2] += 1
                continue
        stop_px = fill - side * sig_stop[k]
        if sig_tp[k] > 0:
            tp_px = fill + side * sig_tp[k]
        else:
            tp_px = np.inf if side > 0 else -np.inf
        e, px, reason = _resolve(i, side, fill, stop_px, tp_px, sig_hold[k],
                                 o, h, l, c, flatten, adverse, through)
        costs = qty * (2.0 * comm + fee_bps * 1e-4 * (fill + px) * pv)
        pnl = side * (px - fill) * pv * qty - costs
        out_k[nt] = k
        out_entry[nt] = i
        out_exit[nt] = e
        out_fill[nt] = fill
        out_px[nt] = px
        out_qty[nt] = qty
        out_pnl[nt] = pnl
        out_risk[nt] = qty * ru
        out_cost[nt] = costs + qty * pv * (adverse[i] + adverse[e] * (reason != TARGET))
        out_reason[nt] = reason
        if not fractional and compounding:
            equity += pnl
        out_eq[nt] = equity
        nt += 1
        counts[4] += 1
        day_pnl += pnl
        consec = consec + 1 if pnl < 0 else 0
        if day_pnl <= -daily_loss_pct * day_eq or consec >= max_consec:
            blocked = True
        free_from = e
    return (out_k[:nt], out_entry[:nt], out_exit[:nt], out_fill[:nt], out_px[:nt], out_qty[:nt],
            out_pnl[:nt], out_risk[:nt], out_cost[:nt], out_reason[:nt], out_eq[:nt], counts)


@njit(cache=True)
def _label(sig_idx, sig_side, sig_stop, sig_tp, sig_hold, o, h, l, c, tday, flatten, entry_ok,
           adverse, through, pv, comm, fee_bps):
    """Outcome of every signal on its own (no position overlap): R per unit risk."""
    n = len(o)
    m = len(sig_idx)
    valid = np.zeros(m, np.bool_)
    r = np.zeros(m)
    reason = np.zeros(m, np.int64)
    exit_i = np.zeros(m, np.int64)
    for k in range(m):
        t = sig_idx[k]
        i = t + 1
        if i >= n or tday[i] != tday[t] or not entry_ok[i] or flatten[i]:
            continue
        side = sig_side[k]
        fill = o[i] + side * adverse[i]
        ru = _risk_unit(sig_stop[k], fill, adverse[i], pv, comm, fee_bps)
        stop_px = fill - side * sig_stop[k]
        if sig_tp[k] > 0:
            tp_px = fill + side * sig_tp[k]
        else:
            tp_px = np.inf if side > 0 else -np.inf
        e, px, rs = _resolve(i, side, fill, stop_px, tp_px, sig_hold[k],
                             o, h, l, c, flatten, adverse, through)
        pnl = side * (px - fill) * pv - 2.0 * comm - fee_bps * 1e-4 * (fill + px) * pv
        valid[k] = True
        r[k] = pnl / ru
        reason[k] = rs
        exit_i[k] = e
    return valid, r, reason, exit_i


@dataclass
class Result:
    trades: pd.DataFrame
    counts: dict
    account: Account
    instrument: Instrument
    fractional: bool
    start_tday: int = 0
    end_tday: int = 0
    days: np.ndarray = field(default_factory=lambda: np.array([], np.int64))

    def daily_pnl(self) -> pd.Series:
        """P&L per trading day over every day in the window (zero on no-trade days)."""
        s = (self.trades.groupby("exit_tday")["pnl"].sum() if len(self.trades)
             else pd.Series(dtype=float))
        return s.reindex(self.days, fill_value=0.0)

    def daily_r(self) -> pd.Series:
        s = (self.trades.groupby("exit_tday")["r"].sum() if len(self.trades)
             else pd.Series(dtype=float))
        return s.reindex(self.days, fill_value=0.0)


def simulate(mk: Market1m, sig: Signals, inst: Instrument, acct: Account = Account(),
             fractional: bool = False, window: tuple[int, int] | None = None) -> Result:
    """Run the signal stream through the account. `window` = (first, last) tday kept.

    fractional=True sizes exactly (no lot rounding, no compounding): the result is
    the strategy's edge in R, independent of account size. fractional=False is
    what the actual account would have done with whole contracts.
    """
    s = sig
    if window is not None:
        td = mk.tday[np.minimum(s.idx, len(mk.tday) - 1)]
        s = s.subset((td >= window[0]) & (td <= window[1]))
    res = _simulate(s.idx, s.side, s.stop, s.tp, s.hold,
                    mk.o, mk.h, mk.l, mk.c, mk.tday, mk.flatten, mk.entry_ok, mk.adverse, mk.through,
                    inst.point_value, inst.commission_side, inst.fee_bps_side,
                    inst.qty_step, inst.min_qty, inst.max_qty,
                    acct.equity, acct.risk_pct, acct.max_risk_pct, fractional, acct.compounding,
                    acct.daily_loss_limit_pct, acct.max_consecutive_losses)
    k, ent, ex, fill, px, qty, pnl, risk, cost, reason, eq, counts = res
    tr = pd.DataFrame({
        "sig_k": k, "entry_i": ent, "exit_i": ex,
        "entry_time": mk.index[ent], "exit_time": mk.index[ex],
        "side": s.side[k], "entry": fill, "exit": px, "qty": qty,
        "stop_pts": s.stop[k], "tp_pts": s.tp[k], "pnl": pnl, "risk": risk,
        "r": np.where(risk > 0, pnl / np.where(risk > 0, risk, 1.0), 0.0),
        "cost": cost, "reason": [REASONS[x] for x in reason], "equity": eq,
        "entry_tday": mk.tday[ent], "exit_tday": mk.tday[ex],
        "bucket": mk.bucket[ent], "rth": mk.rth[ent],
        "minutes": ex - ent + 1,
    })
    if s.meta is not None and len(tr):
        tr = pd.concat([tr, s.meta.iloc[k].reset_index(drop=True).add_prefix("f_")], axis=1)
    if window is not None:
        lo, hi = window
    else:
        lo, hi = int(mk.tday.min()), int(mk.tday.max())
    days = np.unique(mk.tday[(mk.tday >= lo) & (mk.tday <= hi)])
    names = ("overlap", "session", "rejected_size", "blocked_risk", "filled")
    return Result(tr, {n: int(v) for n, v in zip(names, counts)} | {"signals": len(s)},
                  acct, inst, fractional, lo, hi, days)


def label(mk: Market1m, sig: Signals, inst: Instrument) -> pd.DataFrame:
    """Per-signal outcome, each signal evaluated independently (triple-barrier style).

    `hit_tp` is the target-before-stop label used to fit probability models.
    """
    valid, r, reason, exit_i = _label(sig.idx, sig.side, sig.stop, sig.tp, sig.hold,
                                      mk.o, mk.h, mk.l, mk.c, mk.tday, mk.flatten, mk.entry_ok,
                                      mk.adverse, mk.through,
                                      inst.point_value, inst.commission_side, inst.fee_bps_side)
    out = pd.DataFrame({"valid": valid, "r": r, "reason": reason, "exit_i": exit_i,
                        "tday": mk.tday[np.minimum(sig.idx + 1, len(mk.tday) - 1)],
                        "side": sig.side, "hit_tp": reason == TARGET})
    return out
