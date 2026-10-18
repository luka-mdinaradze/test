"""Experiment 3 — do the sophisticated models add out-of-sample value? (fit TRAIN, score VALIDATION)

    python3 experiments/e03_models.py [--market NQ|BTC]

M1  Volatility: ATR vs EWMA vs seasonal EWMA vs GARCH(1,1) — next-bar forecast (QLIKE)
M2  Do better vol forecasts make better stops? (same average stop width, ATR vs seasonal)
M3  Regimes: HMM (forward-filtered), variance-ratio rule, efficiency-ratio rule,
    session, volatility tercile, CUSUM transitions — scored against what the
    NEXT bars actually did (realised forward variance ratio and volatility)
M4  P(target before stop): logistic, gradient boosting, Bayesian beta-binomial —
    Brier skill, AUC, calibration, and whether filtering on it raises expectancy
M5  Order flow (estimated, tick rule on 1-second bars) — NQ only, 5 weeks
Reference configurations (fixed before looking at validation):
    A: w=20 tau=2.0 k=1.5 rr=2.0      B: w=20 z=2.0 k=1.5 rr=2.0
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import probability, regime, volatility  # noqa: E402
from nqlab.config import BTC_MARKET, NQ_MARKET  # noqa: E402
from nqlab.data import RAW  # noqa: E402
from nqlab.engine import Signals  # noqa: E402
from nqlab.research import Lab, md_table, save_json, save_md  # noqa: E402
from nqlab.strategies import AParams, BParams, strategy_a, strategy_b  # noqa: E402
from nqlab.validation import cluster_mean_test, day_block_bootstrap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
REF_A = AParams(20, 2.0, 1.5, 2.0)
REF_B = BParams(20, 2.0, 1.5, 2.0)


def labelled(lab, tf, sig: Signals):
    lb = lab.label(sig)
    meta = sig.meta.copy()
    df = pd.concat([meta, lb[["valid", "r", "hit_tp", "tday", "exit_i"]]], axis=1)
    df["idx"] = sig.idx
    return df[df["valid"]].reset_index(drop=True)


def main(market):
    lab = Lab.build(market, "dev", upto=market.splits[1].end)
    tr, va = lab.split("train"), lab.split("validation")
    out = {"market": market.name}
    md = [f"# Experiment 3 — Model investigations ({market.name}; fit on train, scored on validation)\n"]

    # M1 volatility ----------------------------------------------------------------
    vt = []
    for tf, f in lab.feats.items():
        tab, gp = volatility.compare(f, lab.profiles[tf], lab.mask(tf, tr), lab.mask(tf, va))
        tab["tf"] = tf
        vt.append(tab.reset_index())
        out.setdefault("garch", {})[tf] = gp
    vtab = pd.concat(vt).set_index(["tf", "model"])
    out["vol_forecast"] = vtab.reset_index().to_dict("records")
    md += ["## M1 Next-bar volatility forecasts (QLIKE, lower is better; validation)\n",
           "Each model is rescaled by one constant fitted on train. `qlike_gain_vs_atr_pct` > 0 = better than ATR.\n",
           md_table(vtab, ".4f"), "",
           "GARCH(1,1) on deseasonalised changes (train fit): " + ", ".join(
               f"{tf}m alpha={g['alpha']:.3f} beta={g['beta']:.3f} nu={g['nu']:.1f}" for tf, g in out["garch"].items()), ""]

    # M2 stops: ATR vs seasonal forecast at equal median width --------------------------
    rows = []
    for tf, f in lab.feats.items():
        trm = lab.mask(tf, tr)
        c = np.nanmedian(f["atr"].to_numpy()[trm]) / np.nanmedian(f["sigma_next"].to_numpy()[trm])
        for fam, mk in (("A", lambda ff: strategy_a(ff, tf, REF_A)), ("B", lambda ff: strategy_b(ff, tf, REF_B))):
            for unit in ("atr", "seasonal"):
                ff = f.copy()
                if unit == "seasonal":
                    ff["atr"] = c * f["sigma_next"].to_numpy()
                d = labelled(lab, tf, mk(ff))
                d = d[(d["tday"] >= int(va.start.replace("-", ""))) & (d["tday"] <= int(va.end.replace("-", "")))]
                bs = day_block_bootstrap(d["r"].to_numpy(), d["tday"].to_numpy(), 2000)
                early = (d["exit_i"] - d["idx"] <= max(2, tf)).mean()
                rows.append({"tf": tf, "family": fam, "stop_unit": unit, "signals": len(d),
                             "exp_r": bs["mean"], "ci_lo": bs["lo"], "ci_hi": bs["hi"],
                             "r_std": d["r"].std(), "stopped_within_1_bar": early})
    st = pd.DataFrame(rows).set_index(["tf", "family", "stop_unit"])
    out["stop_unit"] = st.reset_index().to_dict("records")
    md += ["## M2 Stop width from ATR vs from the seasonal volatility forecast (validation, per-signal R after costs)\n",
           "Same median stop width on train; only the time profile of the width differs.\n", md_table(st, ".4f"), ""]

    # M3 regimes ----------------------------------------------------------------------
    rrows, hmm_info, strat_rows = [], {}, []
    for tf, f in lab.feats.items():
        trm, vam = lab.mask(tf, tr), lab.mask(tf, va)
        h = max(5, 60 // tf)
        fwd = regime.forward_realised(f, h)
        th = regime.fit_rules(f, trm)
        X = regime.hmm_obs(f)
        model = regime.fit_hmm(X[trm], 3)
        post = regime.forward_filter(model, X)
        state = np.where(np.isfinite(post[:, 0]), np.nanargmax(np.nan_to_num(post, nan=-1), axis=1), -1)
        hmm_info[tf] = {"transmat": model.transmat.round(4).tolist(),
                        "expected_duration_bars": model.expected_duration.round(1).tolist(),
                        "means": model.means.round(3).tolist()}
        classifiers = {
            "HMM-3 (filtered)": (state, {0: "calm", 1: "mid", 2: "turbulent"}),
            "variance-ratio rule": (regime.rule_regime(f, th, "vr"), {1: "TREND", -1: "RANGE", 0: "NEUTRAL"}),
            "efficiency-ratio rule": (regime.rule_regime(f, th, "er"), {1: "TREND", -1: "RANGE", 0: "NEUTRAL"}),
            "session": (np.where(f["rth"].to_numpy(), 1, -1), {1: "RTH", -1: "ETH"}),
            "vol tercile": (regime.vol_tercile(f, th), {0: "low", 1: "normal", 2: "high", 3: "extreme"}),
            "CUSUM transition": (regime.change_points(f), {0: "stable", 1: "transition"}),
        }
        for cname, (lbl, names) in classifiers.items():
            for sp, m in (("train", trm), ("validation", vam)):
                for v, nm in names.items():
                    mm = m & (lbl == v)
                    if mm.sum() < 30:
                        continue
                    t = cluster_mean_test(fwd["fwd_vr"].to_numpy()[mm] - 1, f["tday"].to_numpy()[mm])
                    vol = np.nanmean(fwd["fwd_absd"].to_numpy()[mm] / f["sigma_next"].to_numpy()[mm])
                    rrows.append({"tf": tf, "classifier": cname, "label": nm, "split": sp, "share": mm.sum() / m.sum(),
                                  "fwd_vr_minus_1": t["mean"], "t": t["t"], "fwd_vol_vs_fcst": vol})
        # strategy outcome by regime label, reference configs
        for fam, sig in (("A", strategy_a(f, tf, REF_A)), ("B", strategy_b(f, tf, REF_B))):
            d = labelled(lab, tf, sig)
            for cname in ("variance-ratio rule", "session", "vol tercile", "HMM-3 (filtered)", "CUSUM transition"):
                lbl, names = classifiers[cname]
                d["lbl"] = lbl[d["bar"].to_numpy()]
                for sp in (tr, va):
                    s = d[(d["tday"] >= int(sp.start.replace("-", ""))) & (d["tday"] <= int(sp.end.replace("-", "")))]
                    for v, g in s.groupby("lbl"):
                        strat_rows.append({"tf": tf, "family": fam, "classifier": cname, "label": names.get(v, v),
                                           "split": sp.name, "n": len(g), "exp_r": g["r"].mean()})
    rg = pd.DataFrame(rrows)
    out["regime_forward"] = rg.to_dict("records")
    out["hmm"] = hmm_info
    sr = pd.DataFrame(strat_rows)
    out["regime_strategy"] = sr.to_dict("records")
    piv = rg.pivot_table(index=["tf", "classifier", "label"], columns="split", values=["fwd_vr_minus_1", "t", "fwd_vol_vs_fcst"])
    md += ["## M3 Regime classifiers vs what happened NEXT\n",
           "`fwd_vr_minus_1`: realised variance ratio of the next h bars minus 1 (h = 60 min or 5 bars).",
           "> 0 means the next bars trended, < 0 means they reverted. A useful trend/range classifier",
           "shows TREND > 0 > RANGE in BOTH periods. `fwd_vol_vs_fcst`: next-bar realised |change| /",
           "forecast (a vol classifier is useful if this differs across labels).\n", md_table(piv, ".3f"), "",
           "### HMM (3 states, fitted on train): transition matrices and expected durations (bars)\n"]
    for tf, i in hmm_info.items():
        md.append(f"- {tf}m: durations {i['expected_duration_bars']}, diag {np.diag(np.array(i['transmat'])).round(3).tolist()}")
    piv2 = sr.pivot_table(index=["tf", "family", "classifier", "label"], columns="split", values=["exp_r", "n"])
    md += ["", "### Reference strategy expectancy (R per signal, after costs) by regime label\n", md_table(piv2, ".3f"), ""]

    # M4 probability ---------------------------------------------------------------------
    prow, filt = [], []
    for tf, f in lab.feats.items():
        th = regime.fit_rules(f, lab.mask(tf, tr))
        for fam, sig in (("A", strategy_a(f, tf, REF_A)), ("B", strategy_b(f, tf, REF_B))):
            d = labelled(lab, tf, sig)
            d["vt"] = regime.vol_tercile(f, th)[d["bar"].to_numpy()]
            ins = (d["tday"] >= int(tr.start.replace("-", ""))) & (d["tday"] <= int(tr.end.replace("-", "")))
            oos = (d["tday"] >= int(va.start.replace("-", ""))) & (d["tday"] <= int(va.end.replace("-", "")))
            if ins.sum() < 300 or oos.sum() < 100:
                continue
            y = d["hit_tp"].to_numpy().astype(float)
            base = y[ins].mean()
            r_win = d.loc[ins & (y == 1), "r"].mean()
            r_non = d.loc[ins & (y == 0), "r"].mean()
            p_be = -r_non / (r_win - r_non)
            preds = {"base rate": np.full(int(oos.sum()), base)}
            for name in ("logistic", "gbm"):
                fm = probability.fit(name, d[ins], y[ins])
                preds[name] = fm.predict(d[oos])
            grp = (d["side"].astype(str) + d["vt"].astype(str) + d["rth"].astype(str)).to_numpy()
            pb = probability.bayes_online(y, grp, d["idx"].to_numpy(), d["exit_i"].to_numpy(), base, 50.0)
            preds["bayes beta-binomial"] = pb[oos.to_numpy()]
            yo = y[oos]
            ro = d.loc[oos, "r"].to_numpy()
            for name, p in preds.items():
                sc = probability.scores(yo, p, base)
                prow.append({"tf": tf, "family": fam, "model": name, **sc})
                if name != "base rate":
                    keep = p >= p_be + 0.02
                    filt.append({"tf": tf, "family": fam, "model": name, "p_breakeven": p_be,
                                 "kept_share": keep.mean(), "exp_r_all": ro.mean(),
                                 "exp_r_kept": ro[keep].mean() if keep.any() else np.nan,
                                 "exp_r_dropped": ro[~keep].mean() if (~keep).any() else np.nan})
    pt = pd.DataFrame(prow).set_index(["tf", "family", "model"])
    ft = pd.DataFrame(filt).set_index(["tf", "family", "model"])
    out["probability"] = pt.reset_index().to_dict("records")
    out["probability_filter"] = ft.reset_index().to_dict("records")
    md += ["## M4 P(target before stop) — validation scores\n",
           "Brier skill > 0 = better than the training base rate. ECE = expected calibration error.\n",
           md_table(pt[["n", "observed_rate", "mean_pred", "brier_skill", "auc", "ece", "log_loss"]], ".4f"), "",
           "### Does filtering on the probability raise expectancy? (keep p >= break-even + 2pp)\n",
           md_table(ft, ".4f"), ""]

    # M5 order flow ------------------------------------------------------------------------
    if market.name == "NQ":
        from nqlab import microstructure
        sec = microstructure.load_seconds(RAW / "nq_b" / "NQ_tick_20260309_20260415.csv")
        m = microstructure.tick_rule_ofi(sec)
        res = microstructure.ofi_predictive_test(m)
        rth = m[(m.index.hour * 60 + m.index.minute >= 570) & (m.index.hour * 60 + m.index.minute < 960)]
        res_rth = microstructure.ofi_predictive_test(rth)
        out["ofi"] = {"all": res, "rth": res_rth}
        md += ["## M5 Estimated order-flow imbalance (tick rule on 1-second bars, NQ-B, Mar-Apr 2026)\n",
               "ESTIMATE, not measured order flow: no bid/ask or aggressor flags exist in any reachable source.\n",
               md_table(pd.DataFrame({"all hours": res, "RTH": res_rth}), ".4f"), ""]

    save_json(out, f"03_models_{market.name}.json")
    save_md("\n".join(md), f"03_models_{market.name}.md")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="NQ", choices=["NQ", "BTC"])
    a = ap.parse_args()
    main(NQ_MARKET if a.market == "NQ" else BTC_MARKET)
