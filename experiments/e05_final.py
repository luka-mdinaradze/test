"""Experiment 5 — the frozen candidates on data they have never seen. Run ONCE.

    python3 experiments/e05_final.py [--market NQ|BTC]

Refuses to run unless results/frozen_config_<market>.json is committed and
unmodified, so nothing here can feed back into the configuration.

1. Identical-period comparison of A, B, C: train, validation, TEST, HOLDOUT (R after costs)
2. Pre-registered gates T1 (test) and H1 (holdout) and the final verdict
3. Every reported metric on the unseen periods, long/short, by session, by vol regime
4. The $5,000 account: whole MNQ contracts (0.5% and 1% risk), CFD, risk governor on
5. Monte Carlo drawdown distribution from the unseen-period trades
"""
import argparse
import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nqlab import metrics, regime  # noqa: E402
from nqlab.config import BTC_MARKET, MNQ, NQ_MARKET, US100_CFD  # noqa: E402
from nqlab.data import bucket_names  # noqa: E402
from nqlab.research import RESULTS, Lab, md_table, save_json, save_md  # noqa: E402
from nqlab.strategies import from_candidate  # noqa: E402
from nqlab.validation import day_block_bootstrap, monte_carlo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent
SHOW = ["trades", "exp_r", "win_rate", "avg_win_r", "avg_loss_r", "pf", "total_r", "gross_r", "cost_r",
        "net_usd", "return_pct", "max_dd_pct", "sharpe", "sortino", "calmar", "avg_minutes",
        "max_consec_wins", "max_consec_losses"]


def require_committed(path: Path):
    rel = path.relative_to(ROOT)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed and unmodified before the final evaluation")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", str(rel)], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def equity_chart(curves: dict, splits: list, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, s in curves.items():
        ax.plot(s.index, s.to_numpy(), label=name, lw=1.4)
    for x, lbl in splits:
        ax.axvline(x, color="0.5", lw=0.8, ls="--")
        ax.text(x, ax.get_ylim()[1], f" {lbl}", va="top", fontsize=8, color="0.35")
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_ylabel("cumulative R (after costs)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(market):
    fpath = RESULTS / f"frozen_config_{market.name}.json"
    frozen_commit = require_committed(fpath)
    frozen = json.loads(fpath.read_text())
    full = Lab.build(market, "dev")
    hold = Lab.build(market, "holdout", profiles=full.profiles) if market.name == "NQ" else full
    periods = [("train", full, full.split("train")), ("validation", full, full.split("validation")),
               ("test", full, full.split("test")), ("holdout", hold, market.holdout)]
    names = bucket_names(market.session)
    out = {"market": market.name, "frozen_commit": frozen_commit, "selected": frozen["selected"],
           "verdict_dev": frozen["verdict_dev"]}
    md = [f"# Experiment 5 — Frozen candidates on unseen data ({market.name})\n",
          f"Frozen configuration committed in `{frozen_commit[:10]}`; development verdict: "
          f"**{frozen['verdict_dev']}**.\n",
          "R per trade after costs, exact fractional sizing, risk governor off, unless stated.\n"]

    comp, gates, curves, trades_by = [], {}, {}, {}
    for fam, cand in frozen["candidates"].items():
        tf = cand["tf"]
        th = regime.fit_rules(full.feats[tf], full.mask(tf, full.split("train")))
        allr = []
        for pname, lab, sp in periods:
            sig = from_candidate(lab.feats[tf], tf, cand, th)
            res = lab.run(sig, sp)
            s = metrics.summarize(res)
            bs = day_block_bootstrap(res.trades["r"].to_numpy(), res.trades["exit_tday"].to_numpy(), 10000) \
                if len(res.trades) > 1 else {"lo": np.nan, "hi": np.nan, "p": np.nan}
            comp.append({"candidate": fam, "period": pname, "trades": s["trades"], "exp_r": s.get("exp_r"),
                         "ci_lo": bs["lo"], "ci_hi": bs["hi"], "p_le0": bs["p"], "win_rate": s.get("win_rate"),
                         "pf": s.get("pf"), "total_r": s.get("total_r", 0.0), "sharpe": s.get("sharpe"),
                         "max_dd_pct": s.get("max_dd_pct")})
            t = res.trades.assign(period=pname)
            if pname in ("test", "holdout"):
                trades_by[(fam, pname)] = (res, s, t)
            allr.append(t)
        cat = pd.concat(allr)
        curves[f"{fam}: {cand['key']}"] = cat.set_index("exit_time")["r"].cumsum()
        te = [c for c in comp if c["candidate"] == fam and c["period"] == "test"][0]
        ho = [c for c in comp if c["candidate"] == fam and c["period"] == "holdout"][0]
        gates[fam] = {
            "T1 test mean>0, PF>1, p<0.10": bool(te["trades"] > 1 and te["exp_r"] > 0 and te["pf"] > 1 and te["p_le0"] < 0.10),
            "H1 holdout mean>0": bool(ho["trades"] > 0 and ho["exp_r"] > 0),
            "passed development": bool(cand["passes_dev"]),
        }
        gates[fam]["would be VALIDATED"] = all(gates[fam].values())
    ct = pd.DataFrame(comp)
    out["comparison"] = ct.to_dict("records")
    out["gates"] = gates
    sel = frozen["selected"]
    verdict = ("VALIDATED (provisional) — paper-trade next" if sel and gates[sel]["would be VALIDATED"]
               else "NOT VALIDATED — the system outputs NO TRADE")
    out["verdict"] = verdict
    md += ["## 1. Identical-period comparison\n",
           md_table(ct.set_index(["candidate", "period"]), ".3f"), "",
           "## 2. Pre-registered gates\n", md_table(pd.DataFrame(gates), ".3f"), "",
           f"**Final verdict: {verdict}.**", ""]

    # 3. detail on unseen periods
    for (fam, pname), (res, s, t) in trades_by.items():
        md += [f"## 3. {fam} on {pname} — full metrics\n",
               md_table(pd.DataFrame({k: [s.get(k)] for k in SHOW + ["n_overlap", "n_session", "days"]}).T.rename(columns={0: "value"}), ".3f"),
               "", "Long vs short:\n", md_table(metrics.long_short(t), ".3f"), "",
               "By entry session:\n", md_table(metrics.breakdown(t, "bucket", dict(enumerate(names))), ".3f"), ""]
        lab = full if pname == "test" else hold
        tf = frozen["candidates"][fam]["tf"]
        th = regime.fit_rules(full.feats[tf], full.mask(tf, full.split("train")))
        vt = regime.vol_tercile(lab.feats[tf], th)
        bar_of = pd.Series(np.arange(len(lab.feats[tf])), index=lab.feats[tf]["end_idx"].to_numpy())
        sig_bar = bar_of.reindex(t["entry_i"].to_numpy() - 1).to_numpy()
        t = t.assign(vol_regime=np.where(np.isfinite(sig_bar), vt[np.nan_to_num(sig_bar, nan=0).astype(int)], -1))
        md += ["By volatility regime at entry (thresholds from train):\n",
               md_table(metrics.breakdown(t, "vol_regime", {0: "low", 1: "normal", 2: "high", 3: "extreme", -1: "n/a"}), ".3f"), ""]
        out.setdefault("detail", {})[f"{fam}_{pname}"] = {"summary": s,
                                                         "long_short": metrics.long_short(t).reset_index().to_dict("records")}

    # 4. the account
    acc_rows = []
    for fam, cand in frozen["candidates"].items():
        tf = cand["tf"]
        th = regime.fit_rules(full.feats[tf], full.mask(tf, full.split("train")))
        for pname, lab, sp in periods[2:]:
            sig = from_candidate(lab.feats[tf], tf, cand, th)
            insts = [(MNQ, 0.005), (MNQ, 0.01), (US100_CFD, 0.005)] if market.name == "NQ" else [(market.instrument, 0.005), (market.instrument, 0.01)]
            for inst, rp in insts:
                acct = replace(market.account, risk_pct=rp)
                res = lab.run(sig, sp, inst=inst, acct=acct, fractional=False)
                s = metrics.summarize(res)
                acc_rows.append({"candidate": fam, "period": pname, "instrument": inst.symbol, "risk_pct": 100 * rp,
                                 "signals": res.counts["signals"], "filled": res.counts["filled"],
                                 "rejected_size": res.counts["rejected_size"], "blocked_governor": res.counts["blocked_risk"],
                                 "net_usd": s["net_usd"], "return_pct": s["return_pct"], "max_dd_pct": s["max_dd_pct"],
                                 "exp_usd": s.get("exp_usd"), "avg_contracts": float(res.trades["qty"].mean()) if len(res.trades) else 0.0})
    at = pd.DataFrame(acc_rows)
    out["account"] = at.to_dict("records")
    md += ["## 4. The $5,000 account (whole contracts / 0.1-lot CFD, risk governor ON, compounding)\n",
           "`rejected_size`: signals whose smallest tradeable size would exceed the risk budget — never taken.\n",
           md_table(at.set_index(["candidate", "period", "instrument", "risk_pct"]), ".2f"), ""]

    # 5. Monte Carlo from unseen trades
    mc_rows = []
    for fam in frozen["candidates"]:
        rs = [trades_by[(fam, p)][2] for p in ("test", "holdout") if (fam, p) in trades_by]
        tt = pd.concat(rs) if rs else pd.DataFrame()
        if len(tt) > 5:
            mc = monte_carlo(tt["r"].to_numpy(), (tt["period"] + tt["exit_tday"].astype(str)).to_numpy(),
                             market.account.risk_pct, 5000, 252, seed=1)
            mc_rows.append({"candidate": fam, "source_trades": len(tt), **mc})
    mt = pd.DataFrame(mc_rows)
    out["monte_carlo"] = mt.to_dict("records")
    md += ["## 5. Monte Carlo (5,000 one-year paths; whole days drawn from test+holdout; 0.5% risk per R)\n",
           md_table(mt.set_index("candidate").T, ".3f") if len(mt) else "(no trades)", ""]

    splits = [(pd.Timestamp(s.start), s.name) for s in market.splits] + [(pd.Timestamp(market.holdout.start), "holdout")]
    fig = RESULTS / "figures" / f"equity_{market.name}.png"
    equity_chart(curves, splits, fig, f"{market.name}: frozen candidates, cumulative R after costs (train | validation | test | holdout)")
    md += [f"![equity](figures/{fig.name})", ""]
    save_json(out, f"05_final_{market.name}.json")
    save_md("\n".join(md), f"05_final_{market.name}.md")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="NQ", choices=["NQ", "BTC"])
    a = ap.parse_args()
    main(NQ_MARKET if a.market == "NQ" else BTC_MARKET)
