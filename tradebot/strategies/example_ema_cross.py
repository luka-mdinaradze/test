"""EXAMPLE ONLY — shows the strategy API working end to end.

An EMA crossover with an ATR stop and a fixed reward:risk target. It is here so
the platform can be exercised in a backtest and in paper mode. It has NOT been
validated and is not a recommendation; the research in this repository found
no such rule with a reliable edge after costs.
"""
from __future__ import annotations

from tradebot import indicators as ind
from tradebot.models import Exit, Signal
from tradebot.strategy import Context, Strategy


class EmaCross(Strategy):
    name = "example_ema_cross"
    warmup = 60
    defaults = {"fast": 12, "slow": 48, "atr_len": 14, "stop_atr": 1.5, "target_rr": 2.0, "max_bars": 48}

    def on_bar(self, ctx: Context):
        p = self.params
        fast, slow = ind.ema(ctx.close, p["fast"]), ind.ema(ctx.close, p["slow"])
        atr = ind.atr(ctx.high, ctx.low, ctx.close, p["atr_len"]).iloc[-1]
        if not atr > 0:
            return None
        up, down = ind.crossed_above(fast, slow), ind.crossed_below(fast, slow)
        ctx.note(f"EMA{p['fast']}={fast.iloc[-1]:.2f} EMA{p['slow']}={slow.iloc[-1]:.2f} ATR={atr:.2f}")

        if ctx.position is not None:
            if (ctx.position.side > 0 and down) or (ctx.position.side < 0 and up):
                return Exit("opposite crossover")
            return None

        stop = p["stop_atr"] * atr
        if up:
            return Signal.long(stop=stop, target=p["target_rr"] * stop, max_bars=p["max_bars"],
                               reason=f"EMA{p['fast']} crossed above EMA{p['slow']}")
        if down:
            return Signal.short(stop=stop, target=p["target_rr"] * stop, max_bars=p["max_bars"],
                                reason=f"EMA{p['fast']} crossed below EMA{p['slow']}")
        return None
