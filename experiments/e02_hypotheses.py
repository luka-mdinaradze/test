"""Experiment 2 — is there statistical structure worth trading? (TRAIN and VALIDATION only)

    python3 experiments/e02_hypotheses.py [--market NQ|BTC]

H1  Serial dependence of bar changes (variance ratios, autocorrelation), by timeframe
H2  Volatility clustering (Ljung-Box on squared changes, ARCH-LM)
H3  Stationarity: price level (unit root expected), changes, z-score deviations
H4  Trend continuation: after z_mom crosses +-2, does price keep going?
H5  Mean reversion: after z_bb crosses +-2, does price come back? — by volatility regime
H6  Intraday momentum (Gao et al. 2018): first-30-min return -> last-30-min return (NQ only)

Every effect is measured in TRAIN and again in VALIDATION: structure that does
not persist into data it was not found in is not structure.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import regime, stats_tests  # noqa: E402
from nqlab.config import BTC_MARKET, NQ_MARKET  # noqa: E402
from nqlab.research import Lab, md_table, save_json, save_md  # noqa: E402
from nqlab.strategies import cross_down, cross_up  # noqa: E402
from nqlab.validation import holm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def within_day_changes(f: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    d = np.diff(f["pc"].to_numpy(), prepend=np.nan)
    first = np.r_[True, f["tday"].to_numpy()[1:] != f["tday"].to_numpy()[:-1]]
    d[first] = np.nan
    return d[mask]


def main(market):
    lab = Lab.build(market, "dev", upto=market.splits[1].end)
    out = {"market": market.name}
    md = [f"# Experiment 2 — Hypothesis tests ({market.name}, train and validation)\n"]
    splits = [lab.split("train"), lab.split("validation")]

    # H1 + H2 + H3 -----------------------------------------------------------
    rows_vr, rows_ac, rows_st = [], [], []
    for tf, f in lab.feats.items():
        for sp in splits:
            m = lab.mask(tf, sp)
            for sess, sm in (("all", np.ones(len(f), bool)), ("RTH", f["rth"].to_numpy()),
                             ("ETH", ~f["rth"].to_numpy())):
                d = within_day_changes(f, m & sm)
                d = d[np.isfinite(d)]
                if len(d) < 500:
                    continue
                for q in (2, 4, 8, 16):
                    r = stats_tests.lo_mackinlay(d, q)
                    rows_vr.append({"tf": tf, "split": sp.name, "session": sess, **r})
                if sess == "all":
                    a = stats_tests.acf_ljungbox(d)
                    arch = stats_tests.arch_lm(d)
                    rows_ac.append({"tf": tf, "split": sp.name, **a, **arch})
            if sp.name == "train":
                st_level = stats_tests.stationarity(f["pc"].to_numpy()[m])
                st_diff = stats_tests.stationarity(within_day_changes(f, m))
                st_z = stats_tests.stationarity(f["zbb_40"].to_numpy()[m])
                for name, s in (("price level P*", st_level), ("bar change", st_diff), ("z_bb(40)", st_z)):
                    rows_st.append({"tf": tf, "series": name, **s})
    vr = pd.DataFrame(rows_vr)
    vr["p_holm"] = pd.Series(holm(dict(enumerate(vr["p"]))))
    out["variance_ratio"] = vr.to_dict("records")
    out["acf"] = rows_ac
    out["stationarity"] = rows_st
    piv = vr[vr["session"] == "all"].pivot_table(index=["tf", "q"], columns="split", values=["vr", "z"])
    md += ["## H1 Serial dependence — Lo-MacKinlay variance ratios (within-day changes)\n",
           "VR>1 = trending, VR<1 = mean-reverting. z* is heteroskedasticity-robust; |z*|>1.96 ~ 5%.",
           "With 10^5 observations tiny effects are 'significant': read the size of VR-1, not just z*.\n",
           md_table(piv.round(4), ".4f"), ""]
    piv_s = vr[(vr["q"] == 4)].pivot_table(index=["tf", "session"], columns="split", values=["vr", "z"])
    md += ["### VR(4) by session\n", md_table(piv_s.round(4), ".4f"), ""]
    md += ["## H1/H2 Autocorrelation and volatility clustering\n",
           md_table(pd.DataFrame(rows_ac).set_index(["tf", "split"])[
               ["acf1", "acf2", "acf5", "se_acf", "lb_p", "lb_sq_stat", "lb_sq_p", "arch_p", "n"]], ".4f"), ""]
    md += ["## H3 Stationarity (TRAIN; ADF H0 = unit root, KPSS H0 = stationary)\n",
           md_table(pd.DataFrame(rows_st).set_index(["tf", "series"]), ".4f"), ""]

    # H4 + H5 conditional forward moves, by volatility regime ------------------
    rows_c = []
    for tf, f in lab.feats.items():
        th = regime.fit_rules(f, lab.mask(tf, splits[0]))
        vt = regime.vol_tercile(f, th)
        for sp in splits:
            m = lab.mask(tf, sp)
            for w in (20, 40):
                zm = f[f"zmom_{w}"].to_numpy()
                zb = f[f"zbb_{w}"].to_numpy()
                for h in (max(1, w // 4), w):
                    for vname, vm in (("all", np.ones(len(f), bool)), ("low vol", vt == 0),
                                      ("normal vol", vt == 1), ("high vol", vt >= 2)):
                        mm = m & vm
                        t = stats_tests.conditional_forward(f, cross_up(zm, 2.0) & mm, cross_down(zm, -2.0) & mm, h)
                        rows_c.append({"tf": tf, "split": sp.name, "test": "H4 trend: after z_mom crosses 2",
                                       "w": w, "h": h, "vol": vname, **t})
                        t = stats_tests.conditional_forward(f, cross_down(zb, -2.0) & mm, cross_up(zb, 2.0) & mm, h)
                        rows_c.append({"tf": tf, "split": sp.name, "test": "H5 revert: after z_bb crosses 2",
                                       "w": w, "h": h, "vol": vname, **t})
    cf = pd.DataFrame(rows_c)
    out["conditional"] = cf.to_dict("records")
    md += ["## H4/H5 Conditional forward moves\n",
           "`mean` = average forward change over h bars in units of sigma*sqrt(h), signed so that POSITIVE",
           "means the hypothesis worked (trend continued / deviation reverted). Day-clustered t-stats.",
           "Under no predictability mean ~ 0. Before costs.\n"]
    for test in cf["test"].unique():
        sub = cf[(cf["test"] == test) & (cf["vol"] == "all")]
        piv = sub.pivot_table(index=["tf", "w", "h"], columns="split", values=["mean", "t", "n"])
        md += [f"### {test} (all regimes)\n", md_table(piv, ".3f"), ""]
        sub = cf[(cf["test"] == test) & (cf["w"] == 20) & (cf["h"] == 20)]
        piv = sub.pivot_table(index=["tf", "vol"], columns="split", values=["mean", "t"])
        md += [f"### {test} by volatility regime (w=20, h=20)\n", md_table(piv, ".3f"), ""]

    # H6 ----------------------------------------------------------------------
    if market.name == "NQ":
        rows_m = []
        for sp in splits:
            m1 = lab.m1[(lab.m1["tday"] >= int(sp.start.replace("-", ""))) & (lab.m1["tday"] <= int(sp.end.replace("-", "")))]
            rows_m.append({"split": sp.name, **stats_tests.intraday_momentum(m1, market.session.rth_start_min,
                                                                             market.session.rth_end_min)})
        out["intraday_momentum"] = rows_m
        md += ["## H6 Intraday momentum (first 30 min + overnight -> last 30 min, daily OLS, HAC t)\n",
               md_table(pd.DataFrame(rows_m).set_index("split"), ".4f"), ""]

    save_json(out, f"02_hypotheses_{market.name}.json")
    save_md("\n".join(md), f"02_hypotheses_{market.name}.md")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="NQ", choices=["NQ", "BTC"])
    a = ap.parse_args()
    main(NQ_MARKET if a.market == "NQ" else BTC_MARKET)
