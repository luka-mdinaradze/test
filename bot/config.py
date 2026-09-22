"""Every tunable in one place. Nothing in the bot reads a magic number.

The defaults encode the design rules for a $100 account:
  - ONE strategy, not three. Complexity is how small accounts die.
  - Maker-only entries. Cuts cost per trade from ~0.096R to ~0.032R (3x).
  - Wide ATR stops, low frequency. Friction is the binding constraint.
  - Survival gates that halt the bot before the account is gone.
  - A paper phase the bot cannot skip on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TradingConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    quote_asset: str = "USDT"

    # --- sizing ---------------------------------------------------------
    risk_pct: float = 0.01            # fraction of equity risked entry->stop
    probation_risk_pct: float = 0.005  # first N live trades run at half risk
    probation_trades: int = 20
    max_leverage: float = 1.0          # spot, no borrowing. Deliberate.
    min_notional: float = 5.0          # venue minimum order value
    qty_step: float = 1e-5
    price_tick: float = 0.01

    # --- strategy (S1 trend breakout) -----------------------------------
    donchian: int = 55
    ema_trend: int = 200
    adx_period: int = 14
    adx_min: float = 20.0
    atr_period: int = 14
    stop_atr: float = 2.0
    trail_atr: float = 3.0
    vol_pct_window: int = 500
    vol_pct_min: float = 0.35
    max_bars_in_trade: int = 240
    allow_short: bool = False          # spot cannot short. Long-only.

    # --- execution ------------------------------------------------------
    maker_only: bool = True            # post-only limit entries
    entry_offset_ticks: int = 1        # how far inside the book to post
    entry_timeout_bars: int = 2        # cancel an unfilled entry after N bars
    stop_on_exchange: bool = True      # THE critical safety property

    # --- risk gates (any breach halts trading) --------------------------
    daily_loss_limit_R: float = 3.0
    max_drawdown_pct: float = 0.20     # from equity high-water mark
    max_consecutive_losses: int = 5
    loss_streak_derisk: int = 3        # halve risk after this many losses
    max_open_positions: int = 1
    max_api_errors: int = 10

    # --- paper -> live promotion gate -----------------------------------
    min_paper_trades: int = 40
    min_paper_expectancy_R: float = 0.05
    max_slippage_bps_observed: float = 15.0

    # --- costs assumed (used for paper fills and expectancy checks) -----
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 10.0
    slippage_bps: float = 2.0

    # --- operations -----------------------------------------------------
    poll_seconds: int = 30
    state_path: str = "bot_state.json"
    log_path: str = "bot.log"
    bars_per_year: int = 8760

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = TradingConfig()
