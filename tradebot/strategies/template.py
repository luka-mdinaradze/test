"""TEMPLATE STRATEGY — the rules go here. Until they do, it never trades.

How to fill it in:
  1. Put every tunable number in `defaults`; override them per bot in the
     config's [strategy.params] section.
  2. Set `warmup` to the number of bars your longest indicator needs.
  3. In `on_bar`, compute indicators from ctx.close / ctx.high / ctx.low /
     ctx.volume (closed bars only, newest last) and return:
        Signal.long(stop=..., target=..., max_bars=..., reason="...")
        Signal.short(...)
        Exit("why")        — only meaningful while a position is open
        None               — do nothing this bar
     stop / target are DISTANCES in price units from the fill price.
  4. Backtest it:  python3 -m tradebot backtest --config configs/backtest_csv.toml
  5. Paper-trade it on live data before anything else.

What NOT to do here: size positions, place orders, check the clock or the
account. The platform does all of that, the same way in backtest and live.
"""
from __future__ import annotations

from tradebot import indicators as ind
from tradebot.models import Exit, Signal  # noqa: F401  (used once the rules are written)
from tradebot.strategy import Context, Strategy


class TemplateStrategy(Strategy):
    name = "template"
    warmup = 50
    defaults = {
        "atr_len": 14,
        "stop_atr": 1.5,        # stop = 1.5 x ATR
        "target_rr": 2.0,       # target = 2 x stop
        "max_bars": 40,         # time exit
    }

    def on_bar(self, ctx: Context):
        p = self.params
        atr = ind.atr(ctx.high, ctx.low, ctx.close, p["atr_len"]).iloc[-1]
        if not atr > 0:
            return None
        stop = p["stop_atr"] * atr
        target = p["target_rr"] * stop

        if ctx.position is not None:
            # --- RULES FOR AN OPEN POSITION -----------------------------------
            # e.g.  if <condition>: return Exit("reason")
            return None

        # --- ENTRY RULES -------------------------------------------------------
        # e.g.  if <long condition>:
        #           return Signal.long(stop=stop, target=target, max_bars=p["max_bars"], reason="...")
        #       if <short condition>:
        #           return Signal.short(stop=stop, target=target, max_bars=p["max_bars"], reason="...")
        return None
