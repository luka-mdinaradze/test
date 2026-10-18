"""Experiment 1 — data integrity, cross-vendor agreement, and position-size feasibility.

    python3 experiments/e01_data.py

Writes results/01_data.json and results/01_data.md.
"""
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import data  # noqa: E402
from nqlab.config import BTC_SESSION, MNQ, NQ, NQ_MARKET, NQ_SESSION, US100_CFD  # noqa: E402
from nqlab.research import Lab, md_table, save_json, save_md  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def feasibility(lab: Lab) -> pd.DataFrame:
    """Share of TRAIN signal bars where the smallest tradeable size fits the risk budget.

    Uses ATR(14) of each timeframe as the volatility unit and stop = k x ATR.
    """
    rows = []
    acct = lab.market.account
    tr = lab.split("train")
    for tf, f in lab.feats.items():
        atr = f["atr"].to_numpy()[lab.mask(tf, tr) & f["rth"].to_numpy()]
        atr = atr[np.isfinite(atr)]
        for inst in (MNQ, NQ, US100_CFD):
            for k in (1.0, 1.5, 2.0):
                stop = k * atr
                adverse = inst.slip_ticks_rth * inst.tick_size + inst.half_spread_rth
                per_unit = (stop + adverse) * inst.point_value + 2 * inst.commission_side
                min_risk = per_unit * inst.min_qty
                rt_cost_pts = 2 * adverse + 2 * inst.commission_side / inst.point_value
                rows.append({
                    "tf": tf, "instrument": inst.symbol, "k_atr": k,
                    "atr_median_pts": float(np.median(atr)),
                    "stop_median_pts": float(np.median(stop)),
                    "min_risk_median_usd": float(np.median(min_risk)),
                    "fits_0.5pct": float((min_risk <= acct.equity * 0.005).mean()),
                    "fits_1.0pct": float((min_risk <= acct.equity * 0.01).mean()),
                    "rt_cost_R_median": float(np.median(rt_cost_pts / (stop + adverse))),
                })
    return pd.DataFrame(rows)


def main():
    out, md = {}, ["# Experiment 1 — Data integrity and feasibility\n"]
    a = data.load_nq_a()
    b = data.load_nq_b()
    btc = data.load_btc()
    reps = [data.validate(a, "NQ-A (dev, 1m)", 0.25, NQ_SESSION),
            data.validate(b, "NQ-B (holdout, 1m)", 0.25, NQ_SESSION),
            data.validate(btc, "BTC Bitstamp (1m)", None, BTC_SESSION)]
    out["integrity"] = [r.as_row() for r in reps]
    tab = pd.DataFrame([r.as_row() for r in reps]).set_index("name").drop(columns=["notes"])
    md += ["## Integrity checks (raw 1-minute bars)\n", md_table(tab.T, ".1f"), ""]
    btc_zero = float((btc["volume"] == 0).mean())
    out["btc_zero_volume_share"] = btc_zero
    md.append(f"BTC: {100 * btc_zero:.2f}% of minutes are zero-volume fills (flat bars inserted by the vendor).\n")

    x = data.validate_cross_source(a)
    out["cross_source"] = x
    md += ["## NQ-A vs NQ-B on their 2025 overlap (hourly)\n",
           f"- overlap hours: {x['overlap_hours']}",
           f"- hourly range correlation: {x['range_corr']:.4f}; hourly change correlation: {x['change_corr']:.4f}",
           f"- volume ratio (median): {x['volume_ratio_median']:.3f}; median |range difference|: {x['range_abs_diff_median']:.2f} pts",
           "- NQ-A minus NQ-B close offset by segment (constant between rolls = additive back-adjustment):\n",
           md_table(pd.DataFrame(x["offset_segments"]), ".2f", index=False), ""]

    lab = Lab.build(NQ_MARKET, "dev", upto=NQ_MARKET.splits[1].end)
    fz = feasibility(lab)
    out["feasibility"] = fz.to_dict("records")
    md += ["## Can a $5,000 account size the trade? (TRAIN, RTH bars)\n",
           "Smallest tradeable size: 1 NQ, 1 MNQ, 0.1 lot of a $1/pt CFD. `fits_x` = share of bars where that",
           "size's risk (stop + exit slippage + commissions) is within x% of $5,000. `rt_cost_R` = round-trip",
           "cost as a fraction of the stop.\n", md_table(fz, ".3f", index=False), ""]
    save_json(out, "01_data.json")
    save_md("\n".join(md), "01_data.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
