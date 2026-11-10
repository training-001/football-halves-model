"""Throttled automatic results refresh, shared by the app (on start) and the scheduled task."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROCESSED_DIR
from src.ingest.update_current import refresh_all, summarize

STATE = PROCESSED_DIR / "refresh_state.json"
LOG = PROCESSED_DIR / "refresh.log"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def last_attempt(state_path: Path = STATE) -> datetime | None:
    if not Path(state_path).exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(Path(state_path).read_text())["last_attempt"])
    except Exception:  # noqa: BLE001
        return None


def due(min_interval_min: float, state_path: Path = STATE, now: datetime | None = None) -> bool:
    t = last_attempt(state_path)
    return t is None or ((now or _now()) - t).total_seconds() / 60 >= min_interval_min


def auto_refresh(min_interval_min: float = 180, path: Path | None = None, state_path: Path = STATE,
                 log_path: Path = LOG, refresher=refresh_all, now: datetime | None = None, force: bool = False):
    """Run a refresh if the last attempt was long enough ago. Returns the result dict, or None if skipped.

    The attempt time is recorded BEFORE refreshing, so a failing source is not hammered on every call."""
    now = now or _now()
    if not force and not due(min_interval_min, state_path, now):
        return None
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps({"last_attempt": now.isoformat()}))
    res = refresher(path) if path else refresher()
    with open(log_path, "a", encoding="utf-8") as f:
        for line in summarize(res):
            f.write(f"{now:%Y-%m-%d %H:%M:%S}Z  {line}\n")
    return res


def mark_attempt(state_path: Path = STATE, now: datetime | None = None) -> None:
    """Record a manual refresh so the automatic one does not repeat it straight away."""
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps({"last_attempt": (now or _now()).isoformat()}))
