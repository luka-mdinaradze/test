"""Experiment 4 — strategy development and selection (TRAIN + VALIDATION only).

    python3 experiments/e04_strategies.py [--market NQ|BTC]

1. Every configuration of A (108) and B (108) at 1/5/15/60 minutes, in R after costs
2. Strategy C at every timeframe, two variants:
     C1 textbook — variance-ratio regime (TREND->A, RANGE->B), blocked in CUSUM
        transitions and extreme volatility
     C2 evidence-based — session router fitted on TRAIN (RTH and ETH each routed to
        A, B or nothing by training expectancy)
3. Entry-method study (Module 4): each trigger alone with identical exits
4. Walk-forward of the selection PROCEDURE: 12-month in-sample, next quarter out
5. Finalists (best A, B, C): bootstrap CI, random-entry null, deflated Sharpe,
   plateau, exits (RR sweep, time-only, mean target), cost sensitivity,
   component ablation (session, transition block, vol block, probability filter)
6. Selection by the pre-registered rule; writes results/frozen_config_<market>.json

Test and holdout data are not loaded (the lab stops at the last validation day).
"""
import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import probability, regime, selection  # noqa: E402
from nqlab.config import BTC_MARKET, NQ_MARKET, Split  # noqa: E402
from nqlab.engine import Signals  # noqa: E402
from nqlab.research import RESULTS, Lab, md_table, save_json, save_md  # noqa: E402
from nqlab.strategies import (A_GRID, B_GRID, AParams, BParams, entry_methods, grid,  # noqa: E402
                              strategy_a, strategy_b, strategy_c)
from nqlab.validation import day_block_bootstrap, deflated_sharpe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("e04")
KEEP = ["sig_k", "entry_i", "exit_i", "side", "r", "pnl", "risk", "cost", "reason",
        "entry_tday", "exit_tday", "bucket", "rth", "minutes"]


def tdi(s):
    return int(s.replace("-", ""))


def build_sig(lab, tf, fam, p, f=None):
    f = lab.feats[tf] if f is None else f
    return strategy_a(f, tf, p) if fam == "A" else strategy_b(f, tf, p)


def daily_sr(tr, days):
    d = tr.groupby("exit_tday")["r"].sum().reindex(days, fill_value=0.0).to_numpy()
    return d, (d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan)


def row_stats(key, fam, tf, params, tr, tw, vw, val_days):
    st_tr = selection.stats_for(tr, *tw)
    st_va = selection.stats_for(tr, *vw)
    vtr = tr[(tr["exit_tday"] >= vw[0]) & (tr["exit_tday"] <= vw[1])]
    _, sr = daily_sr(vtr, val_days)
    return {"key": key, "family": fam, "tf": tf, **params,
            **{f"train_{k}": v for k, v in st_tr.items()}, **{f"val_{k}": v for k, v in st_va.items()},
            "val_daily_sr": sr,
            "val_cost_r": float((vtr["cost"] / vtr["risk"]).mean()) if len(vtr) else np.nan,
            "val_trades_per_day": len(vtr) / len(val_days)}


def random_null(lab, tf, sig, split, n_rep=300, seed=0):
    """Same number of entries at random bars of the same split and timeframe.
    Each random entry inherits one real signal's side, stop multiple (x ATR at the
    new bar), target ratio and holding cap, so only the TIMING is randomised.
    Returns (strategy mean R, null means, p = P(null >= strategy))."""
    f = lab.feats[tf]
    m = lab.mask(tf, split) & np.isfinite(f["atr"].to_numpy())
    end = f["end_idx"].to_numpy()
    nxt = np.minimum(end + 1, len(lab.mk.entry_ok) - 1)
    m &= lab.mk.entry_ok[nxt] & (lab.mk.tday[nxt] == f["tday"].to_numpy())
    td = lab.mk.tday[np.minimum(sig.idx + 1, len(lab.mk.tday) - 1)]
    s = sig.subset((td >= tdi(split.start)) & (td <= tdi(split.end)))
    lb = lab.label(s)
    actual = float(lb.loc[lb["valid"], "r"].mean())
    k_atr = s.meta["stop_atr"].to_numpy(float)
    rr = s.meta["rr"].to_numpy(float)
    cand = np.flatnonzero(m)
    rng = np.random.default_rng(seed)
    atr = f["atr"].to_numpy()
    nulls = np.empty(n_rep)
    n = min(len(s), len(cand))
    for i in range(n_rep):
        bars = np.sort(rng.choice(cand, size=n, replace=False))
        j = rng.permutation(len(s))[:n]
        st = k_atr[j] * atr[bars]
        rs = Signals(end[bars], s.side[j], st, rr[j] * st, s.hold[j])
        l2 = lab.label(rs)
        nulls[i] = l2.loc[l2["valid"], "r"].mean()
    return actual, nulls, float((nulls >= actual).mean())


def main(market, quick=False):
    tr_s, va_s = market.splits[0], market.splits[1]
    lab = Lab.build(market, "dev", upto=va_s.end)
    dev = Split("dev", tr_s.start, va_s.end)
    tw, vw = (tdi(tr_s.start), tdi(tr_s.end)), (tdi(va_s.start), tdi(va_s.end))
    val_days = np.unique(lab.mk.tday[(lab.mk.tday >= vw[0]) & (lab.mk.tday <= vw[1])])
    tfs = sorted(lab.feats)
    out = {"market": market.name, "instrument": market.instrument.symbol}
    md = [f"# Experiment 4 — Strategy development ({market.name}, instrument {market.instrument.symbol})\n",
          "All figures are R per trade AFTER costs (exact fractional sizing, risk governor off).",
          "`score` = mean - 1.96 day-clustered SE (lower edge of a 95% interval).\n"]

    # 1. grids ---------------------------------------------------------------------------
    store, rows = {}, []
    ga = {k: v[:2] for k, v in A_GRID.items()} if quick else A_GRID
    gb = {k: v[:2] for k, v in B_GRID.items()} if quick else B_GRID
    for tf in tfs:
        for fam, g, cls in (("A", ga, AParams), ("B", gb, BParams)):
            for p in grid(g, cls):
                key = f"{tf}m {p.key()}"
                res = lab.run(build_sig(lab, tf, fam, p), dev)
                t = res.trades[KEEP]
                store[key] = t
                rows.append(row_stats(key, fam, tf, asdict(p), t, tw, vw, val_days))
        log.info("grid tf=%d done (%d configs so far)", tf, len(rows))
    table = pd.DataFrame(rows)
    elig = []
    for fam, g in (("A", ga), ("B", gb)):
        elig.append(selection.eligibility(table[table["family"] == fam], g))
    table = pd.concat(elig, ignore_index=True)
    table.to_csv(RESULTS / f"04_grid_{market.name}.csv", index=False)

    summ = table.groupby(["family", "tf"]).agg(
        configs=("key", "size"), train_pos=("train_mean", lambda x: (x > 0).mean()),
        val_pos=("val_mean", lambda x: (x > 0).mean()), both_pos=("eligible", lambda x: np.nan),
        eligible=("eligible", "sum"), best_val_score=("val_score", "max"),
        median_val_mean=("val_mean", "median"), median_cost_r=("val_cost_r", "median"),
        median_trades_day=("val_trades_per_day", "median"))
    summ["both_pos"] = table.assign(b=(table["train_mean"] > 0) & (table["val_mean"] > 0)).groupby(
        ["family", "tf"])["b"].mean()
    out["grid_summary"] = summ.reset_index().to_dict("records")
    md += ["## 1. Grid overview (share of configurations positive after costs)\n", md_table(summ, ".3f"), ""]
    cols = ["key", "train_n", "train_mean", "val_n", "val_mean", "val_score", "val_pf", "plateau", "eligible"]
    md += ["### Top 8 per family by validation score\n"]
    for fam in ("A", "B"):
        md += [md_table(table[table["family"] == fam].sort_values("val_score", ascending=False)[cols].head(8),
                        ".3f", index=False), ""]

    # 2. strategy C ------------------------------------------------------------------------
    crow = []
    cdefs = {}
    for tf in tfs:
        f = lab.feats[tf]
        sub = table[table["tf"] == tf]
        ba = selection.pick(sub[sub["family"] == "A"], require_eligible=False)
        bb = selection.pick(sub[sub["family"] == "B"], require_eligible=False)
        if ba is None or bb is None:
            continue
        pa = AParams(int(ba["w"]), float(ba["tau"]), float(ba["k"]), float(ba["rr"]))
        pb = BParams(int(bb["w"]), float(bb["z"]), float(bb["k"]), float(bb["rr"]))
        th = regime.fit_rules(f, lab.mask(tf, tr_s))
        blocked = (regime.change_points(f) == 1) | (regime.vol_tercile(f, th) == 3)
        c1 = strategy_c(f, tf, pa, pb, regime.rule_regime(f, th, "vr"), blocked)
        route = selection.session_router(store[ba["key"]], store[bb["key"]], *tw)
        lbl = np.where(f["rth"].to_numpy(),
                       {"A": regime.TREND, "B": regime.RANGE, "NONE": regime.NEUTRAL}[route["RTH"]],
                       {"A": regime.TREND, "B": regime.RANGE, "NONE": regime.NEUTRAL}[route["ETH"]])
        c2 = strategy_c(f, tf, pa, pb, lbl)
        for name, sig, extra in (("C1", c1, {"regime": "variance-ratio rule, CUSUM+extreme-vol block"}),
                                 ("C2", c2, {"regime": f"session router RTH->{route['RTH']} ETH->{route['ETH']}"})):
            key = f"{tf}m {name} [{pa.key()} | {pb.key()}]"
            t = lab.run(sig, dev).trades[KEEP] if len(sig) else pd.DataFrame(columns=KEEP)
            store[key] = t
            r = row_stats(key, name, tf, {}, t, tw, vw, val_days)
            r.update(extra)
            r["eligible"] = bool(r["train_mean"] > 0 and r["val_mean"] > 0 and r["val_n"] >= selection.MIN_VAL_TRADES)
            crow.append(r)
            cdefs[key] = {"variant": name, "tf": tf, "A": asdict(pa), "B": asdict(pb), "route": route if name == "C2" else None}
    ctab = pd.DataFrame(crow)
    out["strategy_c"] = ctab.to_dict("records")
    md += ["## 2. Strategy C (regime-adaptive)\n",
           md_table(ctab[["key", "regime", "train_n", "train_mean", "val_n", "val_mean", "val_score", "eligible"]],
                    ".3f", index=False), ""]

    # 3. entry methods -----------------------------------------------------------------------
    erow = []
    for tf in tfs:
        f = lab.feats[tf]
        for name, sig in entry_methods(f, lab.vwap_dev, tf, lab.mask(tf, tr_s)).items():
            t = lab.run(sig, dev).trades
            a, b = selection.stats_for(t, *tw), selection.stats_for(t, *vw)
            erow.append({"tf": tf, "method": name, "train_n": a["n"], "train_mean": a["mean"], "train_t": a["t"],
                         "val_n": b["n"], "val_mean": b["mean"], "val_t": b["t"]})
    etab = pd.DataFrame(erow)
    out["entry_methods"] = etab.to_dict("records")
    md += ["## 3. Entry methods in isolation (identical exits: 1.5 ATR stop, 2R target, 20-bar cap)\n",
           "Order-flow imbalance is not in this table: no source provides it for the development period.\n",
           md_table(etab.set_index(["tf", "method"]), ".3f"), ""]

    # 4. walk-forward --------------------------------------------------------------------------
    folds = selection.quarterly_folds(tr_s.start, va_s.end)
    wf = {}
    for fam in ("A", "B", "A or B"):
        keys = [k for k in table["key"] if (fam == "A or B") or table.loc[table["key"] == k, "family"].iloc[0] == fam]
        oos, ch = selection.walk_forward(store, keys, folds)
        bs = day_block_bootstrap(oos["r"].to_numpy(), oos["exit_tday"].to_numpy(), 10000) if len(oos) else {}
        wf[fam] = {"choices": ch.to_dict("records"), "oos_trades": len(oos), "oos_mean": bs.get("mean"),
                   "ci": [bs.get("lo"), bs.get("hi")], "p": bs.get("p"),
                   "oos_total_r": float(oos["r"].sum()) if len(oos) else 0.0}
        md += [f"## 4. Walk-forward — family {fam} (12m in-sample -> next quarter)\n",
               md_table(ch, ".3f", index=False),
               f"\nOOS: {len(oos)} trades, mean {bs.get('mean', np.nan):.4f} R, 95% CI "
               f"[{bs.get('lo', np.nan):.4f}, {bs.get('hi', np.nan):.4f}], p(mean<=0) = {bs.get('p', np.nan):.3f}, "
               f"total {wf[fam]['oos_total_r']:.1f} R\n"]
    out["walk_forward"] = wf

    # 5. finalists ------------------------------------------------------------------------------
    n_trials = len(table) + len(ctab) + len(etab)
    var_sr = float(np.nanvar(pd.concat([table["val_daily_sr"], ctab["val_daily_sr"]])))
    finals = {}
    for fam in ("A", "B"):
        sub = table[table["family"] == fam]
        row = selection.pick(sub) if selection.pick(sub) is not None else selection.pick(sub, require_eligible=False)
        finals[fam] = row
    finals["C"] = selection.pick(ctab) if selection.pick(ctab) is not None else selection.pick(ctab, require_eligible=False)
    fin_rows, extra_md = [], []
    for fam, row in finals.items():
        if row is None:
            continue
        key, tf = row["key"], int(row["tf"])
        t = store[key]
        v = t[(t["exit_tday"] >= vw[0]) & (t["exit_tday"] <= vw[1])]
        bs = day_block_bootstrap(v["r"].to_numpy(), v["exit_tday"].to_numpy(), 10000)
        dr, _ = daily_sr(v, val_days)
        dsr = deflated_sharpe(dr, n_trials, var_sr)
        if fam in ("A", "B"):
            p = (AParams if fam == "A" else BParams)(**{k: (int(row[k]) if k == "w" else float(row[k]))
                                                         for k in (A_GRID if fam == "A" else B_GRID)})
            sig = build_sig(lab, tf, fam, p)
        else:
            p = None
            sig = c_signals(lab, tf, cdefs[key], tw)
        act, nulls, pnull = random_null(lab, tf, sig, va_s)
        gates = {
            "D1 validation mean > 0": bool(row["val_mean"] > 0),
            "D2 train mean > 0": bool(row["train_mean"] > 0),
            "D3 plateau >= 60%": bool(row.get("plateau", np.nan) >= selection.PLATEAU_SHARE) if fam != "C" else None,
            # C has no grid of its own; its walk-forward evidence is the adaptive 'A or B' procedure
            "D4 walk-forward OOS mean > 0": bool((wf.get(fam, wf["A or B"])["oos_mean"] or -1) > 0),
            "D5 beats random entries (p<0.05)": bool(pnull < 0.05) if np.isfinite(pnull) else None,
        }
        passed = all(g for g in gates.values() if g is not None)
        fin_rows.append({"family": fam, "key": key, "tf": tf, "val_n": len(v), "val_score": row["val_score"],
                         "val_mean": bs["mean"],
                         "ci_lo": bs["lo"], "ci_hi": bs["hi"], "p_boot": bs["p"],
                         "train_mean": row["train_mean"], "plateau": row.get("plateau", np.nan),
                         "null_mean": float(np.nanmean(nulls)), "null_p": pnull,
                         "dsr": dsr["dsr"], "sr_ann": dsr.get("sr_ann"), "sr0_ann": dsr.get("sr0_ann"),
                         **gates, "passes_dev": passed})
        if p is not None:
            extra_md += finalist_studies(lab, tf, fam, p, dev, tw, vw, store, key)
    ftab = pd.DataFrame(fin_rows)
    out["finalists"] = ftab.to_dict("records")
    out["dsr_inputs"] = {"n_trials": n_trials, "var_daily_sr": var_sr}
    md += ["## 5. Finalists (validation)\n",
           f"Deflated Sharpe uses {n_trials} trials and the cross-config variance of daily Sharpe ({var_sr:.5f}).\n",
           md_table(ftab.set_index("family").T, ".4f"), ""] + extra_md

    # 6. selection -------------------------------------------------------------------------------
    passing = ftab[ftab["passes_dev"]]
    chosen = passing.sort_values("val_score", ascending=False).iloc[0] if len(passing) else None
    frozen = {
        "market": market.name, "instrument": market.instrument.to_dict(),
        "account": asdict(market.account), "session": asdict(market.session),
        "frozen_from_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                             cwd=Path(__file__).resolve().parent.parent).stdout.strip(),
        "selected": None if chosen is None else chosen["family"],
        "verdict_dev": "PASSED development gates" if chosen is not None else
                       "NO strategy passed the development gates -> system outputs NO TRADE",
        "candidates": {},
    }
    for fam, row in finals.items():
        if row is None:
            continue
        ent = {"key": row["key"], "tf": int(row["tf"]), "passes_dev":
               bool(ftab.loc[ftab["family"] == fam, "passes_dev"].iloc[0])}
        if fam in ("A", "B"):
            ent["params"] = {k: (int(row[k]) if k == "w" else float(row[k])) for k in (A_GRID if fam == "A" else B_GRID)}
        else:
            ent.update(cdefs[row["key"]])
        frozen["candidates"][fam] = ent
    save_json(frozen, f"frozen_config_{market.name}.json")
    md += ["## 6. Selection\n", f"**{frozen['verdict_dev']}**" + (f" — selected: {frozen['selected']}" if chosen is not None else ""),
           "", "```json", json.dumps(frozen["candidates"], indent=2), "```"]
    save_json(out, f"04_strategies_{market.name}.json")
    save_md("\n".join(md), f"04_strategies_{market.name}.md")
    print("\n".join(md))


def c_signals(lab, tf, cd, tw):
    """Rebuild a Strategy C signal stream from its stored definition."""
    f = lab.feats[tf]
    pa, pb = AParams(**cd["A"]), BParams(**cd["B"])
    th = regime.fit_rules(f, lab.mask(tf, lab.split("train")))
    if cd["variant"] == "C1":
        blocked = (regime.change_points(f) == 1) | (regime.vol_tercile(f, th) == 3)
        return strategy_c(f, tf, pa, pb, regime.rule_regime(f, th, "vr"), blocked)
    to = {"A": regime.TREND, "B": regime.RANGE, "NONE": regime.NEUTRAL}
    lbl = np.where(f["rth"].to_numpy(), to[cd["route"]["RTH"]], to[cd["route"]["ETH"]])
    return strategy_c(f, tf, pa, pb, lbl)


def finalist_studies(lab, tf, fam, p, dev, tw, vw, store, key):
    """Exits (Module 5), cost sensitivity (Module 9) and component ablation."""
    f = lab.feats[tf]
    base = store[key]
    md = [f"### Finalist {fam}: {key}\n"]
    vb = selection.stats_for(base, *vw)

    # exits
    rows = []
    for name, mod in (("fixed RR (selected)", None), ("time exit only (no target)", "notp"),
                      ("target = distance to rolling mean", "mean"), ("stop from seasonal vol forecast", "seasonal")):
        if mod == "mean" and fam != "B":
            continue
        ff = f
        if mod == "seasonal":
            trm = lab.mask(tf, lab.split("train"))
            c = np.nanmedian(f["atr"].to_numpy()[trm]) / np.nanmedian(f["sigma_next"].to_numpy()[trm])
            ff = f.copy()
            ff["atr"] = c * f["sigma_next"].to_numpy()
        sig = build_sig(lab, tf, fam, p, ff)
        if mod == "notp":
            sig.tp[:] = 0.0
        if mod == "mean":
            sma = pd.Series(f["pc"].to_numpy()).rolling(p.w).mean().to_numpy()
            d = np.abs(f["pc"].to_numpy() - sma)[sig.meta["bar"].to_numpy()]
            sig.tp[:] = np.maximum(d, lab.market.instrument.tick_size * 2)
        t = lab.run(sig, dev).trades
        a, b = selection.stats_for(t, *tw), selection.stats_for(t, *vw)
        rows.append({"exit": name, "train_mean": a["mean"], "val_n": b["n"], "val_mean": b["mean"], "val_score": b["score"]})
    md += ["Exits (Module 5):\n", md_table(pd.DataFrame(rows), ".4f", index=False), ""]

    # cost sensitivity
    inst = lab.market.instrument
    rows = []
    for name, i2 in (("as modelled", inst),
                     ("zero slippage", replace(inst, slip_ticks_rth=0, slip_ticks_eth=0, slip_bps_side=0)),
                     ("slippage x2", replace(inst, slip_ticks_rth=inst.slip_ticks_rth * 2, slip_ticks_eth=inst.slip_ticks_eth * 2,
                                             slip_bps_side=inst.slip_bps_side * 2)),
                     ("commission/fees x2", replace(inst, commission_side=inst.commission_side * 2, fee_bps_side=inst.fee_bps_side * 2)),
                     ("target needs 0 ticks through", replace(inst, tp_through_ticks=0, tp_through_bps=0)),
                     ("target needs 2x through", replace(inst, tp_through_ticks=inst.tp_through_ticks * 2, tp_through_bps=inst.tp_through_bps * 2)),
                     ("frictionless", replace(inst, slip_ticks_rth=0, slip_ticks_eth=0, slip_bps_side=0, commission_side=0,
                                              fee_bps_side=0, tp_through_ticks=0, tp_through_bps=0))):
        t = lab.run(build_sig(lab, tf, fam, p), dev, inst=i2).trades
        b = selection.stats_for(t, *vw)
        rows.append({"costs": name, "val_mean": b["mean"], "val_score": b["score"], "val_total_r": b["total_r"]})
    md += ["Cost sensitivity (Module 9):\n", md_table(pd.DataFrame(rows), ".4f", index=False), ""]

    # ablation
    sig = build_sig(lab, tf, fam, p)
    bars = sig.meta["bar"].to_numpy()
    th = regime.fit_rules(f, lab.mask(tf, lab.split("train")))
    lb = lab.label(sig)
    d = pd.concat([sig.meta, lb[["valid", "r", "hit_tp", "tday", "exit_i"]]], axis=1)
    ins = (d["tday"] >= tw[0]) & (d["tday"] <= tw[1]) & d["valid"]
    comps = {"RTH only": f["rth"].to_numpy()[bars],
             "block CUSUM transitions": regime.change_points(f)[bars] == 0,
             "block extreme volatility": regime.vol_tercile(f, th)[bars] != 3}
    if ins.sum() > 300:
        fm = probability.fit("logistic", d[ins], d.loc[ins, "hit_tp"].to_numpy().astype(float))
        pr = fm.predict(d)
        r_win = d.loc[ins & d["hit_tp"], "r"].mean()
        r_non = d.loc[ins & ~d["hit_tp"], "r"].mean()
        comps["probability filter (logistic, p >= break-even+2pp)"] = pr >= (-r_non / (r_win - r_non) + 0.02)
    rows = [{"component": "none (base)", "train_mean": selection.stats_for(base, *tw)["mean"],
             "val_n": vb["n"], "val_mean": vb["mean"], "val_score": vb["score"]}]
    for name, keep in comps.items():
        t = lab.run(sig.subset(keep), dev).trades
        a, b = selection.stats_for(t, *tw), selection.stats_for(t, *vw)
        rows.append({"component": name, "train_mean": a["mean"], "val_n": b["n"], "val_mean": b["mean"], "val_score": b["score"]})
    ab = pd.DataFrame(rows)
    ab["helps_both"] = (ab["train_mean"] > ab["train_mean"].iloc[0]) & (ab["val_mean"] > ab["val_mean"].iloc[0])
    md += ["Component ablation (a component earns its place only if it improves train AND validation):\n",
           md_table(ab, ".4f", index=False), ""]
    return md


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="NQ", choices=["NQ", "BTC"])
    ap.add_argument("--quick", action="store_true", help="2 values per parameter (plumbing check)")
    a = ap.parse_args()
    main(NQ_MARKET if a.market == "NQ" else BTC_MARKET, a.quick)
