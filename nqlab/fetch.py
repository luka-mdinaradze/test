"""Download the research datasets and verify them byte-for-byte.

Each file is pinned to its git blob hash in the public repository it comes from,
so a silently changed upstream file is detected instead of quietly changing the
results.

    python3 -m nqlab.fetch            # download whatever is missing, verify all
    python3 -m nqlab.fetch --verify   # verify only

Provenance (see the report's data section for the validation done on each):
  NQ-A  CME NQ 1-minute, continuous back-adjusted, ET bar-close stamps, Dec-2022
        to Dec-2025. Vendor unstated (NinjaTrader-style export); redistributed
        in MoebiusSC/nq-es-trader-validated. Licence unstated: research use only.
  NQ-B  CME NQ / MNQ 1-minute and 1-second bars from the TopstepX API, UTC
        bar-open stamps, Jan-Apr 2026 (1h from Mar-2025). axb0306/cme-futures-ohlc.
  BTC   Bitstamp BTC/USD 1-minute, 2012 to present. ff137/bitstamp-btcusd-minute-data
        (CC BY-SA 4.0 per its DATA_LICENSE).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

_MOE = "https://raw.githubusercontent.com/MoebiusSC/nq-es-trader-validated/main/data"
_TOP = "https://raw.githubusercontent.com/axb0306/cme-futures-ohlc/main"
_BTC = "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data"

# local name -> (url, git blob sha1). The BTC "latest" file is appended to daily
# upstream, so its hash is recorded but a mismatch is only a warning.
SOURCES = {
    "nq_a/Dataset_NQ_1min_2022_2025.csv": (
        f"{_MOE}/Dataset_NQ_1min_2022_2025.csv", "64215cbad936e5ea63dcd34a25e482f00af1e788"),
    "nq_a/NQ_daily.csv": (f"{_MOE}/NQ_daily.csv", "ad0c12d8413d87f5db8becd76b52f045b468f312"),
    "nq_b/NQ_1min_20260120_20260415.csv": (
        f"{_TOP}/NQ/NQ_1min_20260120_20260415.csv", "9a11a8178695604e92daabce0e9456756a7ce415"),
    "nq_b/MNQ_1min_20260120_20260415.csv": (
        f"{_TOP}/MNQ/MNQ_1min_20260120_20260415.csv", "946c6b76d68548e5b1fe584b6e16de0f19f6d0d4"),
    "nq_b/NQ_1h_20250321_20260415.csv": (
        f"{_TOP}/NQ/NQ_1h_20250321_20260415.csv", "bc84dfc0f54714e94f55ad10f82c86021f338a49"),
    "nq_b/NQ_tick_20260309_20260415.csv": (
        f"{_TOP}/NQ/NQ_tick_20260309_20260415.csv", "fc9730d2844b487be2f1d77d87ef3ddf664f04fd"),
    "btc/btcusd_bitstamp_1min_2012-2025.csv.gz": (
        f"{_BTC}/historical/btcusd_bitstamp_1min_2012-2025.csv.gz",
        "82175ab2da5d4acb24542b8ed254a5b320885f73"),
    "btc/btcusd_bitstamp_1min_latest.csv": (
        f"{_BTC}/updates/btcusd_bitstamp_1min_latest.csv", "b8fadb94a63010def0b9f8b53538a7b35bafd21b"),
}
MUTABLE = {"btc/btcusd_bitstamp_1min_latest.csv"}


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(verify_only: bool = False) -> bool:
    ok = True
    for name, (url, sha) in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            if verify_only:
                print(f"MISSING  {name}")
                ok = False
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"download {name} ...", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        got = git_blob_sha1(dest)
        if got == sha:
            print(f"OK       {name}  {sha[:12]}")
        elif name in MUTABLE:
            print(f"CHANGED  {name}  {got[:12]} (upstream appends daily; recorded {sha[:12]})")
        else:
            print(f"BAD HASH {name}  got {got[:12]} expected {sha[:12]}")
            ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="verify existing files, download nothing")
    args = ap.parse_args()
    sys.exit(0 if fetch(args.verify) else 1)


if __name__ == "__main__":
    main()
