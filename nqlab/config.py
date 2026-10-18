"""Instrument specifications, cost models, account rules and the research protocol.

Everything a backtest result depends on lives here, so a result can always be
traced to the exact assumptions that produced it. Costs are deliberately on the
conservative side of what a retail account pays; change them in one place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

Kind = Literal["future", "cfd", "crypto"]


@dataclass(frozen=True)
class Instrument:
    """Contract economics. All prices are in index points (or USD for BTC).

    point_value       $ P&L per 1.0 price move per unit (contract, lot or coin)
    commission_side   $ per unit per side (futures: exchange + NFA + broker)
    fee_bps_side      proportional fee per side, basis points of notional (crypto)
    half_spread_*     half the quoted spread, paid on each side (CFDs, whose price
                      feed here is futures trades, i.e. roughly mid)
    slip_ticks_*      adverse slippage on market and stop orders, in ticks
    slip_bps_side     extra proportional slippage per side (crypto, where a tick
                      is negligible against price and the price level moves 10x)
    tp_through_*      a resting limit only counts as filled when price trades
                      THROUGH it by this many ticks / bps (queue-position proxy)
    """
    symbol: str
    kind: Kind
    tick_size: float
    point_value: float
    commission_side: float = 0.0
    fee_bps_side: float = 0.0
    half_spread_rth: float = 0.0
    half_spread_eth: float = 0.0
    slip_ticks_rth: float = 1.0
    slip_ticks_eth: float = 2.0
    slip_bps_side: float = 0.0
    tp_through_ticks: float = 1.0
    tp_through_bps: float = 0.0
    qty_step: float = 1.0
    min_qty: float = 1.0
    max_qty: float = 10.0
    exchange: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# CME E-mini Nasdaq-100. $20/pt, 0.25 tick = $5. Commission ~$2.10/side all-in.
NQ = Instrument("NQ", "future", tick_size=0.25, point_value=20.0, commission_side=2.10,
                slip_ticks_rth=1.0, slip_ticks_eth=2.0, tp_through_ticks=1.0,
                qty_step=1.0, min_qty=1.0, max_qty=2.0, exchange="CME Globex")

# CME Micro E-mini Nasdaq-100. $2/pt, 0.25 tick = $0.50. ~$0.62/side all-in
# (exchange ~$0.35 + NFA $0.02 + broker ~$0.25). The only CME contract a $5k
# account can size at 0.5% risk.
MNQ = Instrument("MNQ", "future", tick_size=0.25, point_value=2.0, commission_side=0.62,
                 slip_ticks_rth=1.0, slip_ticks_eth=2.0, tp_through_ticks=1.0,
                 qty_step=1.0, min_qty=1.0, max_qty=20.0, exchange="CME Globex")

# Broker US100 / NAS100 CFD — MODELLED, not observed. Price path = NQ futures
# (the CFD tracks the cash index; intraday point moves are near-identical, the
# level differs by the futures basis). Typical retail terms: $1/pt per lot,
# 0.1-lot minimum, no commission, spread ~1.5 pts RTH / ~3 pts overnight.
# Holding past the broker's daily rollover incurs financing; this system is flat
# before it, so financing is zero by construction.
US100_CFD = Instrument("US100", "cfd", tick_size=0.1, point_value=1.0, commission_side=0.0,
                       half_spread_rth=0.75, half_spread_eth=1.5,
                       slip_ticks_rth=2.0, slip_ticks_eth=4.0, tp_through_ticks=0.0,
                       qty_step=0.1, min_qty=0.1, max_qty=50.0, exchange="OTC broker")

# BTC. Data is Bitstamp BTC/USD spot; costs model a BTC/USDT USD-M perpetual at
# the base taker tier: 5 bps/side fee + 1 bp/side slippage. Funding is ignored
# because positions are flat within the day (funding prints every 8h; an
# intraday hold can straddle one — a small, sign-random cost).
BTC_PERP = Instrument("BTCUSDT", "crypto", tick_size=0.1, point_value=1.0, fee_bps_side=5.0,
                      slip_ticks_rth=0.0, slip_ticks_eth=0.0, slip_bps_side=1.0,
                      tp_through_ticks=0.0, tp_through_bps=0.5,
                      qty_step=0.001, min_qty=0.001, max_qty=5.0, exchange="crypto perp (modelled)")

INSTRUMENTS = {i.symbol: i for i in (NQ, MNQ, US100_CFD, BTC_PERP)}


@dataclass(frozen=True)
class SessionSpec:
    """Trading-day geometry in exchange-local time (naive timestamps).

    A trading day starts at `day_start` on the previous calendar day (CME: 18:00
    ET) and is labelled by the calendar date it ends on. Positions are flattened
    at the close of the last bar starting before `flatten_at`; no entry is taken
    in the `no_entry_last_min` minutes before that, nor in the first
    `no_entry_first_min` minutes after the open (thin, gappy books).
    """
    tz: str
    day_start_min: int          # minutes after midnight the trading day starts
    flatten_min: int            # minutes after midnight: flat before this
    rth_start_min: int
    rth_end_min: int
    no_entry_first_min: int = 15
    no_entry_last_min: int = 10
    buckets: tuple = ()         # (name, start_min, end_min) for reporting
    h1_offset_min: int = 0      # alignment of 60-minute bars


NQ_SESSION = SessionSpec(
    tz="America/New_York", day_start_min=18 * 60, flatten_min=16 * 60 + 50,
    rth_start_min=9 * 60 + 30, rth_end_min=16 * 60,
    buckets=(("ASIA", 18 * 60, 3 * 60), ("LONDON", 3 * 60, 9 * 60 + 30),
             ("RTH_OPEN", 9 * 60 + 30, 10 * 60 + 30), ("RTH_MID", 10 * 60 + 30, 14 * 60),
             ("RTH_CLOSE", 14 * 60, 16 * 60), ("POST", 16 * 60, 17 * 60)),
    h1_offset_min=30,
)

BTC_SESSION = SessionSpec(
    tz="UTC", day_start_min=0, flatten_min=23 * 60 + 50,
    rth_start_min=13 * 60 + 30, rth_end_min=20 * 60,   # "RTH" = US equity hours
    no_entry_first_min=0, no_entry_last_min=10,
    buckets=(("ASIA", 0, 8 * 60), ("EUROPE", 8 * 60, 13 * 60 + 30),
             ("US", 13 * 60 + 30, 20 * 60), ("LATE", 20 * 60, 24 * 60)),
)


@dataclass(frozen=True)
class Account:
    """Account and risk-governor settings (the user's defaults)."""
    equity: float = 5_000.0
    risk_pct: float = 0.005            # 0.5% of equity between entry and stop
    max_risk_pct: float = 0.01         # hard ceiling, never exceeded by sizing
    max_positions: int = 1
    daily_loss_limit_pct: float = 0.02  # stop trading for the day at -2%
    max_consecutive_losses: int = 4     # pause for the rest of the day
    compounding: bool = True


@dataclass(frozen=True)
class Split:
    name: str
    start: str      # inclusive trading-day date
    end: str        # inclusive trading-day date


# The research protocol. Development data (NQ-A) is split chronologically;
# the holdout is a different vendor (NQ-B) over a later period. Walk-forward runs
# inside TRAIN+VALIDATION only. TEST and HOLDOUT are evaluated once, with a
# configuration frozen and committed beforehand.
NQ_SPLITS = (
    Split("train", "2023-01-03", "2024-06-30"),
    Split("validation", "2024-07-01", "2025-03-31"),
    Split("test", "2025-04-01", "2025-12-11"),
)
NQ_HOLDOUT = Split("holdout", "2026-01-27", "2026-04-15")   # first 5 NQ-B sessions = feature warm-up

BTC_SPLITS = (
    Split("train", "2021-01-01", "2023-06-30"),
    Split("validation", "2023-07-01", "2024-06-30"),
    Split("test", "2024-07-01", "2025-06-30"),
)
BTC_HOLDOUT = Split("holdout", "2025-07-01", "2026-10-16")

TIMEFRAMES = (1, 5, 15, 60)   # minutes


@dataclass(frozen=True)
class Market:
    """Everything needed to run one market through the pipeline."""
    name: str
    instrument: Instrument
    session: SessionSpec
    splits: tuple
    holdout: Split
    account: Account = field(default_factory=Account)

    def with_instrument(self, inst: Instrument) -> "Market":
        return replace(self, instrument=inst)


NQ_MARKET = Market("NQ", MNQ, NQ_SESSION, NQ_SPLITS, NQ_HOLDOUT)
BTC_MARKET = Market("BTC", BTC_PERP, BTC_SESSION, BTC_SPLITS, BTC_HOLDOUT)
