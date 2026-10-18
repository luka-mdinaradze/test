"""Bot configuration: one TOML file describes one bot.

API keys are NEVER read from the file — only from the environment variables the
file names — so a config can be committed or shared without leaking secrets.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODES = ("backtest", "paper", "live")
BROKERS = ("paper", "ccxt", "ibkr")
FEEDS = ("csv", "ccxt", "ibkr")
TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


@dataclass
class BotSection:
    name: str = "my-bot"
    mode: str = "paper"              # backtest | paper | live
    state_dir: str = "runs/my-bot"
    history_bars: int = 500          # bars of history handed to the strategy
    poll_seconds: int = 10           # live feeds: how often to check for a new bar


@dataclass
class MarketSection:
    broker: str = "paper"            # paper | ccxt | ibkr
    feed: str = "csv"                # csv | ccxt | ibkr
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "15m"
    csv_path: str = ""
    csv_timezone: str = "UTC"        # timezone of the CSV timestamps
    start: str = ""                  # optional backtest window (YYYY-MM-DD)
    end: str = ""


@dataclass
class InstrumentSection:
    point_value: float = 1.0         # $ P&L per 1.0 price move per unit (MNQ = 2, NQ = 20, BTC = 1)
    tick_size: float = 0.1
    qty_step: float = 0.001          # MNQ/NQ = 1 (whole contracts)
    min_qty: float = 0.001
    max_qty: float = 1.0
    commission_per_unit: float = 0.0  # $ per unit per side (MNQ ~0.62)
    fee_bps: float = 5.0             # % of notional per side, in basis points (crypto taker ~5)
    slippage_ticks: float = 0.0      # adverse slippage on market/stop orders
    slippage_bps: float = 1.0
    target_through_ticks: float = 0.0  # paper: a target fills only if traded through by this much
    allow_short: bool = True


@dataclass
class AccountSection:
    starting_equity: float = 5000.0  # backtest / paper only; live reads the venue
    risk_pct: float = 0.5            # % of equity between entry and stop
    max_risk_pct: float = 1.0        # hard ceiling, never exceeded
    max_positions: int = 1
    daily_loss_limit_pct: float = 2.0
    max_consecutive_losses: int = 4  # pause for the rest of the day
    max_trades_per_day: int = 10
    max_drawdown_pct: float = 15.0   # from the equity peak: HARD halt, a human must reset


@dataclass
class SessionSection:
    timezone: str = "UTC"
    trade_start: str = "00:00"       # entries allowed from (local time)
    trade_end: str = "23:59"         # ... until (exclusive)
    flatten_at: str = ""             # close everything before this time daily ("" = never)
    weekdays: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])   # Mon=0


@dataclass
class StrategySection:
    name: str = "template"           # a module in tradebot/strategies, or "package.module:Class"
    params: dict = field(default_factory=dict)


@dataclass
class CCXTSection:
    exchange: str = "binanceusdm"
    sandbox: bool = True             # testnet / demo trading
    api_key_env: str = "TRADEBOT_API_KEY"
    api_secret_env: str = "TRADEBOT_API_SECRET"
    options: dict = field(default_factory=dict)


@dataclass
class IBKRSection:
    host: str = "127.0.0.1"
    port: int = 7497                 # 7497 = TWS paper, 4002 = Gateway paper; live ports 7496 / 4001
    client_id: int = 17
    symbol: str = "MNQ"
    exchange: str = "CME"
    currency: str = "USD"
    use_rth: bool = False


@dataclass
class Config:
    bot: BotSection = field(default_factory=BotSection)
    market: MarketSection = field(default_factory=MarketSection)
    instrument: InstrumentSection = field(default_factory=InstrumentSection)
    account: AccountSection = field(default_factory=AccountSection)
    session: SessionSection = field(default_factory=SessionSection)
    strategy: StrategySection = field(default_factory=StrategySection)
    ccxt: CCXTSection = field(default_factory=CCXTSection)
    ibkr: IBKRSection = field(default_factory=IBKRSection)
    path: str = ""

    @property
    def tf_minutes(self) -> int:
        return TF_MINUTES[self.market.timeframe]

    @property
    def state_dir(self) -> Path:
        p = Path(self.bot.state_dir)
        return p if p.is_absolute() or not self.path else base_dir(self.path) / p

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        e = []
        if self.bot.mode not in MODES:
            e.append(f"bot.mode must be one of {MODES}")
        if self.market.broker not in BROKERS:
            e.append(f"market.broker must be one of {BROKERS}")
        if self.market.feed not in FEEDS:
            e.append(f"market.feed must be one of {FEEDS}")
        if self.market.timeframe not in TF_MINUTES:
            e.append(f"market.timeframe must be one of {list(TF_MINUTES)}")
        if self.bot.mode == "backtest" and (self.market.broker != "paper" or self.market.feed != "csv"):
            e.append("backtest mode needs broker = 'paper' and feed = 'csv'")
        if self.bot.mode == "paper" and self.market.broker != "paper":
            e.append("paper mode needs broker = 'paper' (live data can still come from a ccxt/ibkr feed)")
        if self.bot.mode == "live" and self.market.broker == "paper":
            e.append("live mode needs a real broker (ccxt or ibkr)")
        if self.market.feed == "csv" and not self.market.csv_path:
            e.append("feed = 'csv' needs market.csv_path")
        a = self.account
        if not (0 < a.risk_pct <= a.max_risk_pct):
            e.append("account.risk_pct must be > 0 and <= account.max_risk_pct")
        if a.max_risk_pct > 2.0:
            e.append("account.max_risk_pct above 2% is refused by design")
        if a.max_positions != 1:
            e.append("this template supports exactly one open position (account.max_positions = 1)")
        i = self.instrument
        if i.min_qty <= 0 or i.qty_step <= 0 or i.max_qty < i.min_qty or i.point_value <= 0:
            e.append("instrument sizes (min_qty, qty_step, max_qty, point_value) are inconsistent")
        return e


def base_dir(cfg_path: str | Path) -> Path:
    """Relative paths in a config resolve against the project root: the config's
    folder, or its parent when the config lives in a `configs/` folder."""
    p = Path(cfg_path).resolve().parent
    return p.parent if p.name == "configs" else p


def _section(cls, raw: dict | None):
    raw = dict(raw or {})
    known = cls.__dataclass_fields__
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__.replace('Section', '').lower()}]: {sorted(unknown)}")
    return cls(**raw)


def load(path: str | Path) -> Config:
    raw = tomllib.loads(Path(path).read_text())
    cfg = Config(
        bot=_section(BotSection, raw.get("bot")),
        market=_section(MarketSection, raw.get("market")),
        instrument=_section(InstrumentSection, raw.get("instrument")),
        account=_section(AccountSection, raw.get("account")),
        session=_section(SessionSection, raw.get("session")),
        strategy=_section(StrategySection, raw.get("strategy")),
        ccxt=_section(CCXTSection, raw.get("ccxt")),
        ibkr=_section(IBKRSection, raw.get("ibkr")),
        path=str(path),
    )
    if cfg.market.csv_path and not Path(cfg.market.csv_path).is_absolute():
        cfg.market.csv_path = str(base_dir(path) / cfg.market.csv_path)
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(errs))
    return cfg


def secret(env_name: str) -> str:
    v = os.environ.get(env_name, "")
    if not v:
        raise RuntimeError(f"environment variable {env_name} is not set (API keys are never read from config files)")
    return v
