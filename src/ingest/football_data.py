"""Download and parse football-data.co.uk season files.

URL pattern: https://www.football-data.co.uk/mmz4281/<SSYY>/<DIV>.csv   e.g. .../2526/E0.csv

Harmonised output columns
-------------------------
league, season, date, home, away, ft_home, ft_away, ht_home, ht_away, source,
plus any bookmaker odds columns present (renamed, see `_rename_odds`).

Notes
-----
* football-data has NO half-time odds, so the odds are only a baseline for
  full-time 1X2 and Over/Under 2.5. They cannot validate 1H/2H markets.
* Closing odds (…C…) exist from roughly 2019/20 onward; earlier seasons will have NaN.
"""
from __future__ import annotations
import io
import re
import time
from pathlib import Path
import pandas as pd

from src.config import LEAGUES

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
REQUIRED_SRC = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]
_ODDS_RE = re.compile(r"^(B365|PS|P|Avg|Max)(C?)(H|D|A|[<>]2\.5)$")


def season_codes(start: str, end: str) -> list[str]:
    """season_codes('1516','2526') -> ['1516',...,'2526']. Handles the 1999/2000 rollover."""
    def full(yy: str) -> int:           # 90-99 -> 1990s, 00-89 -> 2000s
        y = int(yy[:2])
        return 1900 + y if y >= 90 else 2000 + y
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(full(start), full(end) + 1)]


def _rename_odds(col: str) -> str | None:
    m = _ODDS_RE.match(col)
    if not m:
        return None
    book, closing, sel = m.groups()
    sel = sel.replace(">2.5", "_o25").replace("<2.5", "_u25")
    return f"{book}{closing}{sel}"


def _parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    miss = d.isna()
    if miss.any():
        d[miss] = pd.to_datetime(s[miss], format="%d/%m/%y", errors="coerce")
    return d


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_fd_bytes(raw: bytes, code: str, season: str) -> tuple[pd.DataFrame, dict]:
    """Parse one season file. Returns (dataframe, info) where info counts dropped rows."""
    df = pd.read_csv(io.StringIO(_decode(raw)), skip_blank_lines=True, on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    missing = [c for c in REQUIRED_SRC if c not in df.columns]
    if missing:
        raise ValueError(f"{code} {season}: missing columns {missing}")

    info = {"rows_in": len(df)}
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="any")
    out = pd.DataFrame({
        "league": LEAGUES[code],
        "season": season,
        "date": _parse_dates(df["Date"]),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "ft_home": pd.to_numeric(df["FTHG"], errors="coerce"),
        "ft_away": pd.to_numeric(df["FTAG"], errors="coerce"),
        "ht_home": pd.to_numeric(df["HTHG"], errors="coerce"),
        "ht_away": pd.to_numeric(df["HTAG"], errors="coerce"),
        "source": "football-data",
    })
    for c in df.columns:
        new = _rename_odds(c)
        if new:
            out[new] = pd.to_numeric(df[c], errors="coerce")

    info["bad_dates"] = int(out["date"].isna().sum())
    out = out.dropna(subset=["date"])
    scored = out[["ft_home", "ft_away", "ht_home", "ht_away"]].notna().all(axis=1)
    info["dropped_unscored"] = int((~scored).sum())
    out = out[scored].reset_index(drop=True)
    info["rows_out"] = len(out)
    return out, info


def download_season(season: str, code: str, cache_dir: Path, force: bool = False,
                    timeout: int = 30, retries: int = 3) -> Path | None:
    """Download one file into cache_dir. Returns the path, or None if it does not exist (404)."""
    import requests  # local import so tests run without network libs configured

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{season}_{code}.csv"
    if path.exists() and not force:
        return path
    url = BASE_URL.format(season=season, code=code)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 404:
                print(f"  [404] {url}")
                return None
            r.raise_for_status()
            if b"HomeTeam" not in r.content[:2000]:
                raise ValueError("response does not look like a football-data CSV (HTML error page?)")
            path.write_bytes(r.content)
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt}/{retries} failed for {url}: {exc}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}")


def load_seasons(seasons: list[str], cache_dir: Path, force: bool = False,
                 pause: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download + parse all leagues for the given seasons. Returns (matches, per-file log)."""
    frames, log = [], []
    for season in seasons:
        for code in LEAGUES:
            path = download_season(season, code, cache_dir, force)
            if path is None:
                log.append({"season": season, "code": code, "status": "missing"})
                continue
            df, info = parse_fd_bytes(path.read_bytes(), code, season)
            log.append({"season": season, "code": code, "status": "ok", **info})
            frames.append(df)
            time.sleep(pause)
    if not frames:
        raise RuntimeError("No data downloaded")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(log)
