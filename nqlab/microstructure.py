"""What the available data can and cannot say about order flow.

Available: OHLCV candles (1-minute, 2023-2025) and 1-second OHLCV bars for NQ
over five weeks in Mar-Apr 2026.
NOT available in any source reachable here: bid/ask quotes, order-book depth,
trade-by-trade aggressor flags. So spread, order-book imbalance and true
order-flow imbalance are NOT measured — they are assumed (spread) or absent.

What IS done, labelled as an estimate everywhere it appears:
  * signed volume from the TICK RULE on 1-second bars (a second's volume is
    signed by the direction of its close vs the previous close; unchanged
    seconds inherit the last sign). This is a classical proxy for aggressor
    side; on 1-second bars it misclassifies whenever both sides trade in the
    same second. It is an estimate of order flow, not order flow.
  * volume and range shocks relative to the time-of-day norm (from candles).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_seconds(path) -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["datetime"])
    s.index = s["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return s[["open", "high", "low", "close", "volume"]].astype(float)


def tick_rule_ofi(sec: pd.DataFrame) -> pd.DataFrame:
    """Estimated order-flow imbalance per minute from 1-second bars."""
    c = sec["close"].to_numpy()
    dc = np.diff(c, prepend=c[0])
    sign = np.sign(dc)
    s = pd.Series(sign).replace(0, np.nan).ffill().fillna(0).to_numpy()
    signed = s * sec["volume"].to_numpy()
    m = pd.DataFrame({"signed": signed, "vol": sec["volume"].to_numpy(), "close": c}, index=sec.index)
    g = m.resample("1min")
    out = pd.DataFrame({"ofi_est": g["signed"].sum() / g["vol"].sum().replace(0, np.nan),
                        "vol": g["vol"].sum(), "close": g["close"].last()}).dropna()
    gap = out.index.to_series().diff() != pd.Timedelta(minutes=1)
    ret = out["close"].diff()
    ret[gap] = np.nan
    out["ret"] = ret
    out["ret_next"] = out["ret"].shift(-1)
    out.loc[gap.shift(-1, fill_value=True).to_numpy(), "ret_next"] = np.nan
    return out


def ofi_predictive_test(m: pd.DataFrame, frac_in: float = 0.6) -> dict:
    """Does estimated OFI in minute t predict the return of minute t+1?

    Contemporaneous correlation (OFI vs same-minute return) is reported as a
    sanity check that the estimate carries information at all; the predictive
    slope is fitted on the first part and its sign/size checked on the rest.
    """
    import statsmodels.api as sm
    m = m.dropna(subset=["ofi_est", "ret", "ret_next"])
    cut = int(len(m) * frac_in)
    a, b = m.iloc[:cut], m.iloc[cut:]
    contemp = float(np.corrcoef(m["ofi_est"], m["ret"])[0, 1])
    fit = sm.OLS(a["ret_next"], sm.add_constant(a["ofi_est"])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    corr_out = float(np.corrcoef(b["ofi_est"], b["ret_next"])[0, 1])
    pred = fit.params["const"] + fit.params["ofi_est"] * b["ofi_est"]
    r2_oos = float(1 - ((b["ret_next"] - pred) ** 2).sum() / ((b["ret_next"] - b["ret_next"].mean()) ** 2).sum())
    hit = float((np.sign(b["ofi_est"]) == np.sign(b["ret_next"]))[b["ret_next"] != 0].mean())
    return {"minutes": len(m), "contemp_corr": contemp, "slope_in": float(fit.params["ofi_est"]),
            "t_in": float(fit.tvalues["ofi_est"]), "corr_out": corr_out, "r2_oos": r2_oos,
            "sign_hit_out": hit, "out_minutes": len(b),
            "start": str(m.index[0]), "end": str(m.index[-1])}
