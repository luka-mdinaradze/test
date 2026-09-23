#!/usr/bin/env python3
"""Download full BTCUSDT 1h history from Binance's public data archive.

data.binance.vision serves monthly CSV dumps with NO API KEY and no rate limit.
It is the easiest way to get the whole 2017->now window onto disk.

    python3 fetch_data.py                          # BTCUSDT 1h, 2017-08 -> now
    python3 fetch_data.py --symbol ETHUSDT --interval 4h
    python3 fetch_data.py --start 2020-01 --out btc.csv

Then hand the CSV to the harness:

    python3 run_all.py --csv btc.csv
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def parse_month_csv(raw: bytes) -> pd.DataFrame:
    """Parse one monthly dump.

    Two wrinkles Binance introduced over the years and that silently corrupt a
    naive parse: newer files carry a header row, and some carry open_time in
    MICROseconds rather than milliseconds. Both are detected here rather than
    assumed, because a wrong epoch unit produces timestamps in 1970 or 55000
    that look like data and ruin every rolling window downstream.
    """
    head = raw[:64].decode("utf-8", "ignore").lower()
    has_header = "open_time" in head or "open time" in head
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None,
                     names=None if has_header else COLS)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    t = pd.to_numeric(df["open_time"], errors="coerce")
    med = t.dropna().median()
    unit = "us" if med > 1e15 else ("ms" if med > 1e11 else "s")
    df.index = pd.to_datetime(t, unit=unit, utc=True)

    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch(symbol: str, interval: str, start: str, end: str, quiet: bool = False) -> pd.DataFrame:
    frames, missing = [], []
    for ym in month_range(start, end):
        url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{ym}.zip"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                blob = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                missing.append(ym)          # month not published (yet, or listing gap)
                continue
            raise
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            raw = z.read(z.namelist()[0])
        frames.append(parse_month_csv(raw))
        if not quiet:
            print(f"  {ym}  {len(frames[-1]):>5,} bars", flush=True)

    if not frames:
        raise SystemExit("no data downloaded - check the symbol/interval spelling")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if missing and not quiet:
        print(f"  (no file for: {', '.join(missing)})")
    return df


def audit(df: pd.DataFrame, interval: str) -> None:
    """Say plainly what is wrong with the data before anyone backtests on it."""
    step = pd.Timedelta(interval.replace("h", "H").replace("m", "min").replace("d", "D"))
    gaps = df.index.to_series().diff().dropna()
    holes = gaps[gaps > step]
    bad = df[(df["high"] < df["low"]) | (df["high"] < df["open"]) |
             (df["high"] < df["close"]) | (df["low"] > df["open"]) | (df["low"] > df["close"])]
    zeros = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]

    print(f"\nbars          {len(df):,}")
    print(f"range         {df.index[0]} -> {df.index[-1]}")
    print(f"span          {(df.index[-1] - df.index[0]).days / 365.25:.1f} years")
    expected = int((df.index[-1] - df.index[0]) / step) + 1
    print(f"completeness  {len(df) / expected * 100:.2f}% ({expected - len(df):,} bars missing)")
    print(f"gaps          {len(holes)}" + (f"  (largest {holes.max()})" if len(holes) else ""))
    print(f"bad OHLC      {len(bad)}")
    print(f"zero/neg      {len(zeros)}")
    if len(df) < 17_520:
        print("\nWARNING: under 2 years of 1h bars. Walk-forward needs more to mean anything.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()

    end = a.end or (date.today().replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    out = a.out or f"{a.symbol}_{a.interval}.csv"
    print(f"downloading {a.symbol} {a.interval}  {a.start} -> {end}")
    df = fetch(a.symbol, a.interval, a.start, end, a.quiet)
    audit(df, a.interval)
    df.to_csv(out, index_label="open_time")
    size = Path(out).stat().st_size / 1e6
    print(f"\nsaved {out}  ({size:.1f} MB)")
    print(f"\nnext:  python3 run_all.py --csv {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
