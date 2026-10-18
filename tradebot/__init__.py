"""tradebot — a strategy-agnostic trading bot platform.

The platform owns everything that is the same for every strategy: data feeds,
order routing, position sizing, the risk governor, session rules, state,
journaling and the run loop. A strategy is one small class with one method,
`on_bar`, that looks at closed bars and answers LONG, SHORT, EXIT or nothing.

Backtest, paper and live trading run through the same loop, so what is tested
is what runs. See TRADEBOT.md.
"""
__version__ = "0.1.0"
