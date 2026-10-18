"""Broker adapters. All speak the `Broker` interface in base.py."""
from __future__ import annotations


def make_broker(cfg):
    kind = cfg.market.broker
    if kind == "paper":
        from .paper import PaperBroker
        return PaperBroker(cfg.instrument, cfg.account.starting_equity)
    if kind == "ccxt":
        from .ccxt_broker import CCXTBroker
        return CCXTBroker(cfg)
    if kind == "ibkr":
        from .ibkr import IBKRBroker
        return IBKRBroker(cfg)
    raise ValueError(kind)
