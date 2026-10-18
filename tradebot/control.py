"""Start, stop and inspect bot processes (used by the dashboard and the CLI).

A bot runs as its own background process (`python -m tradebot run ...`), so
closing the dashboard does not stop it. Its PID lives in <state_dir>/bot.pid.

Stopping:
  graceful   writes the kill switch: no new entries, an open position is closed
             at the next bar, then the process exits by itself
  force      sends Ctrl+C (SIGINT) now: state is saved; any stop order already
             resting on the venue stays there
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .config import Config

PROJECT = Path(__file__).resolve().parent.parent


def pid_file(cfg: Config) -> Path:
    return cfg.state_dir / "bot.pid"


def running_pid(cfg: Config) -> int | None:
    """PID of this config's bot if it is alive and really is a tradebot process."""
    p = pid_file(cfg)
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError, PermissionError):
        p.unlink(missing_ok=True)
        return None
    try:
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    except OSError:
        cmd = "tradebot"
    if "tradebot" not in cmd:
        p.unlink(missing_ok=True)
        return None
    return pid


def start(cfg: Config, live_confirmed: bool = False) -> int:
    if running_pid(cfg):
        raise RuntimeError("this bot is already running")
    if cfg.bot.mode == "backtest":
        raise RuntimeError("a backtest config is not a bot — run it from the Backtest tab")
    if cfg.bot.mode == "live" and not live_confirmed:
        raise RuntimeError("live mode needs explicit confirmation")
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "STOP").unlink(missing_ok=True)
    args = [sys.executable, "-m", "tradebot", "run", "--config", str(Path(cfg.path).resolve())]
    if cfg.bot.mode == "live":
        args.append("--i-understand-live")
    out = open(cfg.state_dir / "console.log", "a")
    proc = subprocess.Popen(args, cwd=PROJECT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    pid_file(cfg).write_text(str(proc.pid))
    return proc.pid


def stop_graceful(cfg: Config) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "STOP").write_text("stop requested\n")


def stop_force(cfg: Config) -> bool:
    pid = running_pid(cfg)
    if not pid:
        return False
    os.kill(pid, signal.SIGINT)
    return True


def tail(path: Path, n: int = 60) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 64_000))
        lines = fh.read().decode("utf-8", "replace").splitlines()
    return "\n".join(lines[-n:])
