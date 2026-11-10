"""Health of the scheduled odds collector, read from data/odds/collector.log and Windows Task Scheduler."""
from __future__ import annotations
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def task_status() -> str:
    """Is the Windows scheduled task installed? ('installed' / 'not installed' / 'unknown')"""
    if os.name != "nt":
        return "unknown"
    try:
        r = subprocess.run(["schtasks", "/Query", "/TN", "FootballOddsCollector"],
                           capture_output=True, text=True, timeout=10)
        return "installed" if r.returncode == 0 else "not installed"
    except Exception:  # noqa: BLE001
        return "unknown"


def collector_health(log_path: Path, now: datetime | None = None) -> tuple[str | None, float | None, bool]:
    """(last log line, minutes since it, key currently rejected?).

    Only the MOST RECENT conclusive line counts: an HTTP 401 means rejected, while a line showing the API
    answered ('API ok', or a captured match) means healthy, even if older 401 lines remain above it.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None, None, False
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None, None, False
    now = now or datetime.now(timezone.utc)
    try:
        t = datetime.strptime(lines[-1][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        mins = (now - t).total_seconds() / 60
    except ValueError:
        mins = None
    rejected = False
    for l in reversed(lines):
        if "HTTP 401" in l:
            rejected = True
            break
        if "API ok" in l or " prices from " in l:
            break
    return lines[-1], mins, rejected
