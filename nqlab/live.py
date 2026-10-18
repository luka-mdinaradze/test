"""Live signal generation in the required format, from the frozen configuration.

    python3 -m nqlab.live --csv recent_1m.csv               # your latest 1-minute bars
    python3 -m nqlab.live --demo                            # last bars of the holdout file
    python3 -m nqlab.live --demo --candidate A              # research view of a NOT-validated candidate

The CSV needs columns timestamp,open,high,low,close,volume (1-minute bars, bar-OPEN
timestamps). Timestamps are interpreted as exchange-local (ET for NQ) unless
--utc is given. Give at least ~10 trading days so every feature is warmed up.

What it will and will not say:
* SIGNAL is LONG/SHORT only if the frozen configuration names a SELECTED strategy
  (one that passed every pre-registered gate) and that strategy fires on the last
  closed bar. Otherwise it is NO TRADE.
* ESTIMATED WIN PROBABILITY and SIGNAL CONFIDENCE are printed only for a model
  that passed out-of-sample calibration. None did, so they read 'unavailable'.
  In --candidate mode the historical win FREQUENCY (share of trades with P&L > 0)
  on unseen data is shown, labelled as a frequency, not a forecast.
* POSITION SIZE is whole contracts within the risk budget; if one contract
  exceeds it, the size is 0 and the reason is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features, regime
from .config import BTC_MARKET, MNQ, NQ_MARKET, Market
from .research import RESULTS, Lab
from .strategies import from_candidate

CACHE = data.CACHE


def train_profiles(market: Market, tf: int) -> features.Profiles:
    """Time-of-day profiles fitted on the TRAINING split, cached after first use."""
    p = CACHE / f"profiles_{market.name}_{tf}.npz"
    if p.exists():
        z = np.load(p)
        return features.Profiles(tf, z["vol"], z["volume"], int(z["slots"]))
    lab = Lab.build(market, "dev", upto=market.splits[0].end, timeframes=(tf,))
    prof = lab.profiles[tf]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(p, vol=prof.vol, volume=prof.volume, slots=prof.slots)
    return prof


def regime_thresholds(market: Market, tf: int):
    p = CACHE / f"rules_{market.name}_{tf}.json"
    if p.exists():
        return regime.RuleThresholds(**{k: (tuple(v) if k == "vol_q" else v) for k, v in json.loads(p.read_text()).items()})
    lab = Lab.build(market, "dev", upto=market.splits[0].end, timeframes=(tf,))
    th = regime.fit_rules(lab.feats[tf], lab.mask(tf, lab.split("train")))
    p.write_text(json.dumps(th.__dict__))
    return th


def load_csv(path: str, market: Market, utc: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df[df.columns[0]])
    if utc:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(market.session.tz).dt.tz_localize(None)
    df.index = pd.DatetimeIndex(ts, name="ts")
    return df[data.OHLCV].astype(float).sort_index()


def _hist(market: Market, fam: str) -> dict:
    p = RESULTS / f"05_final_{market.name}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    rows = [r for r in d["comparison"] if r["candidate"] == fam and r["period"] in ("test", "holdout")]
    if not rows:
        return {}
    n = sum(r["trades"] for r in rows)
    mean = sum((r["exp_r"] or 0) * r["trades"] for r in rows) / max(n, 1)
    wr = sum((r["win_rate"] or 0) * r["trades"] for r in rows) / max(n, 1)
    return {"n": n, "exp_r": mean, "win_freq": wr}


def generate(m1raw: pd.DataFrame, market: Market, candidate: str | None = None) -> str:
    frozen = json.loads((RESULTS / f"frozen_config_{market.name}.json").read_text())
    selected = frozen.get("selected")
    fam = candidate or selected
    cand = frozen["candidates"].get(fam) if fam else None
    tf = cand["tf"] if cand else 15
    inst = MNQ if market.name == "NQ" else market.instrument
    m1 = data.prepare(m1raw, market.session)
    tfd = data.resample(m1, tf, market.session)
    prof = train_profiles(market, tf)
    f = features.build(tfd, prof)
    th = regime_thresholds(market, tf)
    last = len(f) - 1
    bar_time = f.index[last]
    close = float(tfd["close"].iloc[last])
    rth = bool(f["rth"].iloc[last])
    vt = ["low", "normal", "high", "extreme"][int(regime.vol_tercile(f, th)[last])]
    rr_lbl = {1: "TREND", -1: "RANGE", 0: "NEUTRAL"}[int(regime.rule_regime(f, th, "vr")[last])]
    regime_txt = (f"session={'RTH' if rth else 'ETH'}; volatility={vt} (deseasonalised level vs 10-day median, "
                  f"train terciles); variance-ratio label={rr_lbl} (not predictive out of sample — informational)")
    lines = {"INSTRUMENT": f"{inst.symbol} ({'CME Micro E-mini Nasdaq-100' if inst.symbol == 'MNQ' else inst.exchange})",
             "TIMEFRAME": f"{tf} minutes (last closed bar {bar_time:%Y-%m-%d %H:%M}-"
                          f"{bar_time + pd.Timedelta(minutes=tf):%H:%M} {market.session.tz})"}
    fired, side = False, 0
    if cand:
        sig = from_candidate(f, tf, cand, th)
        hit = np.flatnonzero(sig.meta["bar"].to_numpy() == last)
        if len(hit):
            fired, k = True, hit[0]
            side = int(sig.side[k])
            stop_pts, tp_pts = float(sig.stop[k]), float(sig.tp[k])
            hold_min = int(sig.hold[k])
            zval = float(sig.meta["z_entry"].iloc[k])
    validated = selected is not None and fam == selected
    sess = market.session
    flat_d = (sess.flatten_min - sess.day_start_min) % 1440 or 1440
    next_dmin = int(m1["dmin"].iloc[-1]) + 1
    late = fired and not (sess.no_entry_first_min <= next_dmin < flat_d - sess.no_entry_last_min)
    if late:
        fired = False
    if fired:
        entry = close
        tick = inst.tick_size
        stop = round((entry - side * stop_pts) / tick) * tick
        tp = round((entry + side * tp_pts) / tick) * tick if tp_pts > 0 else float("nan")
        adverse = inst.slip_ticks_rth * tick if rth else inst.slip_ticks_eth * tick
        per_contract = (stop_pts + adverse) * inst.point_value + 2 * inst.commission_side
        budget = market.account.equity * market.account.risk_pct
        qty = int(math.floor(budget / per_contract)) if per_contract > 0 else 0
        qty = min(qty, int(inst.max_qty))
        hist = _hist(market, fam)
        signal = ("LONG" if side > 0 else "SHORT") if validated else f"NO TRADE (candidate {fam} would be "\
            f"{'LONG' if side > 0 else 'SHORT'} — NOT VALIDATED, research view only)"
        lines.update({
            "SIGNAL": signal,
            "ENTRY PRICE": f"~{entry:.2f} — market order at the next bar's open (last close {close:.2f})",
            "STOP-LOSS": f"{stop:.2f} ({stop_pts:.2f} pts = {cand.get('params', cand.get('A', {})).get('k', '?')} x ATR14)",
            "TAKE-PROFIT": f"{tp:.2f} ({tp_pts:.2f} pts)" if tp_pts > 0 else "none (time/flatten exit)",
            "RISK-TO-REWARD": f"1:{tp_pts / stop_pts:.2f}" if tp_pts > 0 else "n/a",
            "ESTIMATED WIN PROBABILITY": "unavailable — no probability model passed out-of-sample calibration "
                                         "(Brier skill <= 0.04, AUC ~0.5)" + (
                f". Historical win FREQUENCY (P&L > 0) on unseen data: {100 * hist['win_freq']:.1f}% over "
                f"{hist['n']} trades — a frequency, not a calibrated forecast" if hist else ""),
            "EXPECTED VALUE": (f"not validated. Unseen-data average {hist['exp_r']:+.3f} R/trade after costs "
                               f"(= ${hist['exp_r'] * budget:+.2f} at ${budget:.0f}/R)" if hist else "not validated"),
            "POSITION SIZE": (f"{qty} contract(s), risk ${qty * per_contract:.2f} of ${budget:.2f} budget" if qty >= 1 else
                              f"0 — one contract risks ${per_contract:.2f} > ${budget:.2f} budget "
                              f"({100 * market.account.risk_pct:.1f}% of ${market.account.equity:,.0f}); trade NOT allowed"),
            "MARKET REGIME": regime_txt,
            "SIGNAL CONFIDENCE": "unavailable — no confidence model was validated",
            "REASON FOR SIGNAL": f"{cand['key']}: entry statistic {zval:+.2f} crossed its threshold at this bar's close",
            "INVALIDATION CONDITION": (f"stop {stop:.2f} trades; or {hold_min} minutes elapse (holding cap); "
                                       "or session flatten "
                                       f"{market.session.flatten_min // 60:02d}:{market.session.flatten_min % 60:02d}; "
                                       "or the entry bar opens beyond the stop (skip)"),
        })
    else:
        why = (frozen["verdict_dev"] + "; final verdict on unseen data: NOT VALIDATED" if not selected
               else f"selected strategy {selected} has no fresh trigger on the last closed bar")
        if late:
            why += ". A trigger fired, but inside the no-entry window before the session flatten"
        if cand:
            zc = cand.get("params", cand.get("A", {}))
            col = f"zmom_{zc['w']}" if "tau" in zc else f"zbb_{zc['w']}"
            if col in f:
                why += f". Candidate {fam} statistic {col} = {float(f[col].iloc[last]):+.2f}"
        lines.update({"SIGNAL": "NO TRADE", "ENTRY PRICE": "-", "STOP-LOSS": "-", "TAKE-PROFIT": "-",
                      "RISK-TO-REWARD": "-",
                      "ESTIMATED WIN PROBABILITY": "unavailable — no probability model passed out-of-sample calibration",
                      "EXPECTED VALUE": "unavailable — no strategy has a validated positive expectancy after costs",
                      "POSITION SIZE": "0 contracts", "MARKET REGIME": regime_txt,
                      "SIGNAL CONFIDENCE": "unavailable — not validated", "REASON FOR SIGNAL": why,
                      "INVALIDATION CONDITION": "-"})
    order = ["INSTRUMENT", "SIGNAL", "TIMEFRAME", "ENTRY PRICE", "STOP-LOSS", "TAKE-PROFIT", "RISK-TO-REWARD",
             "ESTIMATED WIN PROBABILITY", "EXPECTED VALUE", "POSITION SIZE", "MARKET REGIME", "SIGNAL CONFIDENCE",
             "REASON FOR SIGNAL", "INVALIDATION CONDITION"]
    return "\n".join(f"{k}: {lines[k]}" for k in order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="NQ", choices=["NQ", "BTC"])
    ap.add_argument("--csv", help="recent 1-minute bars")
    ap.add_argument("--utc", action="store_true", help="CSV timestamps are UTC")
    ap.add_argument("--demo", action="store_true", help="use the last bars of the holdout data")
    ap.add_argument("--candidate", choices=["A", "B", "C"], help="research view of a frozen (not validated) candidate")
    ap.add_argument("--until", help="demo only: pretend 'now' is this timestamp (exchange time)")
    a = ap.parse_args()
    market = NQ_MARKET if a.market == "NQ" else BTC_MARKET
    if a.csv:
        raw = load_csv(a.csv, market, a.utc)
    elif a.demo:
        raw = data.load_nq_b() if market.name == "NQ" else data.load_btc()
        if a.until:
            raw = raw.loc[: a.until]
        raw = raw.iloc[-25_000:]
    else:
        sys.exit("give --csv FILE or --demo")
    print(generate(raw, market, a.candidate))


if __name__ == "__main__":
    main()
