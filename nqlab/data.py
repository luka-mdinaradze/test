"""Loading, validation, session geometry and resampling.

Conventions established by cross-checking the two NQ vendors against each other
(see `validate_cross_source`), not assumed:

* NQ-A stamps a bar with its CLOSE time in ET (first Globex bar is 18:01, the
  last is 17:00). It is converted to OPEN-time stamps here.
* NQ-A is a back-adjusted continuous contract: it equals NQ-B plus a constant
  per contract segment (708.5 / 494 / 256.5 points across the 2025 rolls).
  Point differences are exact; price levels and percentage returns are not.
  Every feature in this project is therefore built from point differences.
* NQ-A has exactly 1,048,575 rows — Excel's row limit. The file was truncated
  by a spreadsheet mid-session, so the final partial trading day is dropped.
* NQ-B is UTC, bar-open stamped, front-month, NOT back-adjusted.

All frames leaving this module are indexed by naive exchange-local timestamps
of the bar OPEN, sorted, de-duplicated, with float OHLC and volume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SessionSpec

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
OHLCV = ["open", "high", "low", "close", "volume"]
EXCEL_ROW_LIMIT = 1_048_575


def _cached(name: str, builder) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = builder()
    df.to_parquet(p)
    return df


def load_nq_a(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "nq_a" / "Dataset_NQ_1min_2022_2025.csv"

    def build():
        raw = pd.read_csv(path)
        ts = pd.to_datetime(raw["timestamp ET"], format="%m/%d/%Y %H:%M") - pd.Timedelta(minutes=1)
        df = raw[OHLCV].astype(float)
        df.index = pd.DatetimeIndex(ts, name="ts")
        if len(raw) == EXCEL_ROW_LIMIT:
            # Spreadsheet truncation: the last trading day is a fragment.
            tday = trading_day(df.index, 18 * 60)
            last = tday.max()
            log.warning("NQ-A has %d rows (Excel limit): dropping truncated trading day %s",
                        len(raw), last)
            df = df[tday != last]
        return _tidy(df)
    return _cached("nq_a_1m", build)


def load_nq_b(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW / "nq_b" / "NQ_1min_20260120_20260415.csv"

    def build():
        raw = pd.read_csv(path, parse_dates=["datetime"])
        idx = raw["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
        df = raw[OHLCV].astype(float)
        df.index = pd.DatetimeIndex(idx, name="ts")
        return _tidy(df)
    return _cached(f"nq_b_{path.stem}", build)


def load_btc(start: str = "2019-01-01") -> pd.DataFrame:
    def build():
        parts = [pd.read_csv(RAW / "btc" / "btcusd_bitstamp_1min_2012-2025.csv.gz"),
                 pd.read_csv(RAW / "btc" / "btcusd_bitstamp_1min_latest.csv")]
        raw = pd.concat(parts).drop_duplicates("timestamp", keep="last")
        df = raw[OHLCV].astype(float)
        df.index = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], unit="s"), name="ts")
        df = df.loc[start:]
        # Bitstamp fills untraded minutes with a flat zero-volume bar. They are
        # real "nothing happened" minutes for exit checks but carry no information.
        return _tidy(df)
    return _cached(f"btc_1m_{start}", build)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# --------------------------------------------------------------------------- sessions

def trading_day(idx: pd.DatetimeIndex, day_start_min: int) -> np.ndarray:
    """Trading-day label (yyyymmdd int) = calendar date on which the day ENDS."""
    shift = (24 * 60 - day_start_min) % (24 * 60)
    d = (idx + pd.Timedelta(minutes=shift)).normalize()
    return (d.year * 10000 + d.month * 100 + d.day).to_numpy()


def prepare(df: pd.DataFrame, sess: SessionSpec) -> pd.DataFrame:
    """Add session geometry and the gap-removed price used by every feature.

    Columns added:
      tday      trading-day label
      mod       clock minute of day of the bar open
      dmin      minutes since the trading-day start
      bucket    index into sess.buckets (reporting only)
      rth       bar inside regular trading hours
      flatten   last bar of the day that starts before the flatten time —
                an open position is closed at this bar's close
      post      bar at/after the flatten time (no position may exist)
      entry_ok  a new position may be OPENED at this bar's open
      po/ph/pl/pc  OHLC with every between-day gap removed (cumulative). Rolls,
                weekend gaps and the daily halt never enter a feature.
    """
    out = df.copy()
    idx = out.index
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    tday = trading_day(idx, sess.day_start_min)
    dmin = (mod - sess.day_start_min) % (24 * 60)
    flat_d = (sess.flatten_min - sess.day_start_min) % (24 * 60) or 24 * 60
    out["tday"] = tday
    out["mod"] = mod
    out["dmin"] = dmin
    out["bucket"] = _bucket(mod, sess.buckets)
    s, e = sess.rth_start_min, sess.rth_end_min
    out["rth"] = (mod >= s) & (mod < e)
    out["post"] = dmin >= flat_d

    pre = ~out["post"].to_numpy()
    day_change = np.r_[True, tday[1:] != tday[:-1]]
    # last pre-flatten bar of each day (handles early-close days automatically)
    flatten = np.zeros(len(out), bool)
    pos = np.flatnonzero(pre)
    if len(pos):
        last_of_day = np.r_[tday[pos][1:] != tday[pos][:-1], True]
        flatten[pos[last_of_day]] = True
    out["flatten"] = flatten
    out["entry_ok"] = (pre & ~flatten & (dmin >= sess.no_entry_first_min)
                       & (dmin < flat_d - sess.no_entry_last_min))

    o = out["open"].to_numpy()
    c = out["close"].to_numpy()
    gap = np.where(day_change, o - np.r_[o[0], c[:-1]], 0.0)
    gap[0] = 0.0
    adj = np.cumsum(gap)
    for a, b in (("open", "po"), ("high", "ph"), ("low", "pl"), ("close", "pc")):
        out[b] = out[a].to_numpy() - adj
    return out


def _bucket(mod: np.ndarray, buckets: tuple) -> np.ndarray:
    b = np.full(len(mod), -1, np.int8)
    for k, (_, s, e) in enumerate(buckets):
        m = (mod >= s) & (mod < e) if s < e else (mod >= s) | (mod < e)
        b[m] = k
    return b


def bucket_names(sess: SessionSpec) -> list[str]:
    return [b[0] for b in sess.buckets]


# --------------------------------------------------------------------------- resampling

def resample(m1: pd.DataFrame, tf: int, sess: SessionSpec) -> pd.DataFrame:
    """Aggregate prepared 1-minute bars to `tf` minutes, never across a trading day.

    Each output bar carries `end_idx`, the positional index of its last 1-minute
    bar. A signal computed on the bar is acted on at the OPEN of 1-minute bar
    end_idx+1 — after the bar has closed — which is what makes signals causal.
    Bars after the flatten time are excluded (nothing can be traded there).
    """
    m1 = m1.reset_index()
    m1["pos"] = np.arange(len(m1))
    m1 = m1[~m1["post"]]
    if tf == 1:
        out = m1.set_index("ts")
        out["end_idx"] = out["pos"].to_numpy()
        out["n1"] = 1
        return out.drop(columns=["pos"])
    off = sess.h1_offset_min if tf == 60 else 0
    key = (m1["mod"].to_numpy() - off) // tf
    g = m1.groupby([m1["tday"].to_numpy(), key], sort=False)
    out = pd.DataFrame({
        "ts": g["ts"].first(),
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "po": g["po"].first(), "ph": g["ph"].max(), "pl": g["pl"].min(), "pc": g["pc"].last(),
        "tday": g["tday"].first(), "mod": g["mod"].first(), "dmin": g["dmin"].first(),
        "bucket": g["bucket"].first(), "rth": g["rth"].first(),
        "end_idx": g["pos"].last(), "n1": g["pos"].size(),
    })
    out = out.sort_values("end_idx").set_index("ts")
    return out


# --------------------------------------------------------------------------- validation

@dataclass
class DataReport:
    name: str
    rows: int
    start: str
    end: str
    trading_days: int
    duplicates: int
    non_monotonic: int
    off_tick_grid: int
    ohlc_violations: int
    nonpositive_volume: int
    median_bars_per_day: float
    short_days: dict
    intraday_gaps_gt5min: int
    large_intraday_jumps: int
    day_boundary_gaps_gt_1pct: int
    notes: str = ""

    def as_row(self) -> dict:
        return self.__dict__


def validate(df: pd.DataFrame, name: str, tick: float | None, sess: SessionSpec) -> DataReport:
    """Integrity checks on raw 1-minute bars. Problems are counted, not hidden."""
    idx = df.index
    tday = trading_day(idx, sess.day_start_min)
    o, h, l, c, v = (df[k].to_numpy() for k in OHLCV)
    off = 0
    if tick:
        for x in (o, h, l, c):
            off += int((np.abs(x / tick - np.round(x / tick)) > 1e-6).sum())
    viol = int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum())
    per_day = pd.Series(1, index=tday).groupby(level=0).sum()
    med = float(per_day.median())
    short = {int(k): int(n) for k, n in per_day[per_day < 0.75 * med].items()}
    same_day = np.r_[False, tday[1:] == tday[:-1]]
    dt = np.r_[0, np.diff(idx.asi8) // 60_000_000_000]
    gaps = int((same_day & (dt > 5)).sum())
    d1 = np.abs(np.diff(c, prepend=c[0]))
    med_move = np.median(d1[d1 > 0]) if (d1 > 0).any() else 0.0
    jumps = int((same_day & (np.abs(o - np.r_[c[0], c[:-1]]) > 30 * med_move)).sum()) if med_move else 0
    bnd = ~same_day
    bnd[0] = False
    big_bnd = int((bnd & (np.abs(o / np.r_[c[0], c[:-1]] - 1) > 0.01)).sum())
    return DataReport(
        name=name, rows=len(df), start=str(idx.min()), end=str(idx.max()),
        trading_days=int(per_day.size), duplicates=int(idx.duplicated().sum()),
        non_monotonic=int((np.diff(idx.asi8) <= 0).sum()), off_tick_grid=off,
        ohlc_violations=viol, nonpositive_volume=int((v <= 0).sum()),
        median_bars_per_day=med, short_days=short, intraday_gaps_gt5min=gaps,
        large_intraday_jumps=jumps, day_boundary_gaps_gt_1pct=big_bnd,
    )


def validate_cross_source(a1m: pd.DataFrame, b1h_path: Path | None = None) -> dict:
    """Compare NQ-A against NQ-B's independent hourly bars on their overlap.

    Returns correlations of hourly ranges and changes, the volume ratio, and the
    (NQ-A minus NQ-B) close offset per contract segment. A back-adjusted series
    shows a constant offset that steps only at rolls.
    """
    b1h_path = b1h_path or RAW / "nq_b" / "NQ_1h_20250321_20260415.csv"
    b = pd.read_csv(b1h_path, parse_dates=["datetime"])
    b.index = b["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    a = a1m[OHLCV].resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                       "close": "last", "volume": "sum"}).dropna()
    j = a.join(b[OHLCV], rsuffix="_b", how="inner")
    off = (j["close"] - j["close_b"]).round(2)
    seg_id = (off.diff().abs() > 2).cumsum()
    segs = off.groupby(seg_id).agg(["median", "count"])
    segs["start"] = off.groupby(seg_id).apply(lambda s: str(s.index[0]))
    segs["end"] = off.groupby(seg_id).apply(lambda s: str(s.index[-1]))
    segs = segs[segs["count"] >= 100]
    return {
        "overlap_hours": int(len(j)),
        "range_corr": float(np.corrcoef(j.high - j.low, j.high_b - j.low_b)[0, 1]),
        "change_corr": float(np.corrcoef(j.close.diff().iloc[1:], j.close_b.diff().iloc[1:])[0, 1]),
        "volume_ratio_median": float((j.volume / j.volume_b).median()),
        "range_abs_diff_median": float(((j.high - j.low) - (j.high_b - j.low_b)).abs().median()),
        "offset_segments": segs.reset_index(drop=True).to_dict("records"),
    }


def split_mask(tday: np.ndarray, start: str, end: str) -> np.ndarray:
    s = int(start.replace("-", ""))
    e = int(end.replace("-", ""))
    return (tday >= s) & (tday <= e)
