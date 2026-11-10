"""Upcoming fixtures for the five leagues.

Sources, in order of preference
  1. football-data.co.uk/fixtures.csv  - free, no key, team names already canonical.
     Kickoff times there are UK local time, converted here to UTC.
  2. The Odds API /events              - free call, needs ODDS_API_KEY; names are mapped to canonical
     names on a best-effort basis (unmatched names are reported, never guessed).
  3. The last successful download, kept in data/fixtures_cache.csv.

Everything is stored in UTC; the app converts to the viewer's time zone.
"""
from __future__ import annotations
import difflib
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import LEAGUE_KEYS, LEAGUES, ROOT
from src.cleaning.team_names import canonical

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
CACHE = ROOT / "data" / "fixtures_cache.csv"
COLS = ["league", "kickoff_utc", "time_known", "home", "away", "source"]


# ------------------------------------------------------------------ football-data fixtures.csv
def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_fd_fixtures(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_decode(raw)), on_bad_lines="skip")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for c in ("Div", "Date", "HomeTeam", "AwayTeam"):
        if c not in df.columns:
            raise ValueError(f"fixtures file is missing column {c!r}")
    df = df[df["Div"].isin(LEAGUES) & df["Date"].notna() & df["HomeTeam"].notna() & df["AwayTeam"].notna()].copy()
    d = df["Date"].astype(str).str.strip()
    t = df["Time"].astype(str).str.strip() if "Time" in df.columns else pd.Series("", index=df.index)
    known = t.str.match(r"^\d{1,2}:\d{2}$")
    stamp = d + " " + t.where(known, "12:00")
    ts = pd.to_datetime(stamp, format="%d/%m/%Y %H:%M", errors="coerce")
    miss = ts.isna()
    if miss.any():
        ts[miss] = pd.to_datetime(stamp[miss], format="%d/%m/%y %H:%M", errors="coerce")
    uk = ts.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward")
    out = pd.DataFrame({
        "league": df["Div"].map(LEAGUES),
        "kickoff_utc": uk.dt.tz_convert("UTC"),
        "time_known": known,
        "home": df["HomeTeam"].astype(str).str.strip().map(canonical),
        "away": df["AwayTeam"].astype(str).str.strip().map(canonical),
        "source": "football-data",
    })
    return out.dropna(subset=["kickoff_utc"]).sort_values("kickoff_utc").reset_index(drop=True)


def fetch_fd_fixtures(timeout: int = 30) -> pd.DataFrame:
    import requests
    r = requests.get(FIXTURES_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return parse_fd_fixtures(r.content)


# ------------------------------------------------------------------ The Odds API fallback
def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s)).strip()


_AFFIX = re.compile(r"^(afc|fc|cf|ac|ssc|us|as|ss|rc|rcd)\s+|\s+(fc|cf|afc|sc|ac|calcio)$", re.I)


def _variants(name: str) -> list[str]:
    """The name, then the name with club prefixes/suffixes stripped ('Manchester United FC' -> 'Manchester United')."""
    out, n = [str(name).strip()], str(name).strip()
    for _ in range(3):
        n2 = _AFFIX.sub("", n).strip()
        if n2 == n or not n2:
            break
        out.append(n2)
        n = n2
    return out


def resolve_team(name: str, candidates: set[str]) -> str | None:
    """Map an external team name to one of the canonical `candidates`, or None if unsure."""
    if not candidates:
        return None
    norm = {_norm(x): x for x in candidates}
    variants = _variants(name)
    for v in variants:                                  # exact / alias / normalised equality
        c = canonical(v)
        if c in candidates:
            return c
        if _norm(v) in norm:
            return norm[_norm(v)]
    # all tokens of exactly one candidate appear in the name ("brighton" in "brighton hove albion")
    toks = set(_norm(variants[-1]).split()) | set(_norm(variants[0]).split())
    hits = [x for k, x in norm.items() if k and set(k.split()) <= toks]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(_norm(variants[-1]), list(norm), n=2, cutoff=0.85)
    if len(close) == 1:
        return norm[close[0]]
    return None


def fetch_odds_api_fixtures(client, known_teams: dict[str, set[str]]) -> tuple[pd.DataFrame, list[str]]:
    rows, unresolved = [], []
    for league, sport in LEAGUE_KEYS.items():
        r = client.get(f"/sports/{sport}/events")
        if r.status_code != 200:
            continue
        cands = known_teams.get(league, set())
        for e in r.json():
            h, a = resolve_team(e["home_team"], cands), resolve_team(e["away_team"], cands)
            if h is None or a is None:
                unresolved += [n for n, v in ((e["home_team"], h), (e["away_team"], a)) if v is None]
                continue
            rows.append({"league": league, "kickoff_utc": pd.Timestamp(e["commence_time"]).tz_convert("UTC"),
                         "time_known": True, "home": h, "away": a, "source": "odds-api"})
    df = pd.DataFrame(rows, columns=COLS)
    return df.sort_values("kickoff_utc").reset_index(drop=True), sorted(set(unresolved))


# ------------------------------------------------------------------ orchestration + cache
def _read_cache(path: Path) -> tuple[pd.DataFrame, datetime] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "fetched_at" not in df.columns:
        return None
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    df["time_known"] = df["time_known"].astype(bool)
    fetched = pd.to_datetime(df["fetched_at"].iloc[0], utc=True).to_pydatetime()
    return df[COLS], fetched


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[COLS].copy()
    out["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(path, index=False)


def combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of sources. A pair (league, home, away) occurs once per season, so it identifies a fixture.
    When both sources list it, the row with a known kickoff time wins (football-data first on ties)."""
    df = pd.concat(parts, ignore_index=True)
    df["_p"] = (~df["time_known"].astype(bool)).astype(int)
    df = df.sort_values(["_p", "kickoff_utc"], kind="stable").drop_duplicates(["league", "home", "away"], keep="first")
    return df.drop(columns="_p").sort_values("kickoff_utc").reset_index(drop=True)


def _counts(df: pd.DataFrame) -> str:
    c = df["source"].value_counts()
    return " + ".join(f"{n} from {'football-data.co.uk' if k == 'football-data' else 'The Odds API'}" for k, n in c.items())


def get_fixtures(max_age_min: float = 30, cache_path: Path = CACHE, fetch=fetch_fd_fixtures,
                 odds_client=None, known_teams: dict[str, set[str]] | None = None,
                 force: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (fixtures, human-readable status). Never raises: falls back to cache, then to empty.

    football-data.co.uk only lists the next round, so The Odds API (a free call) is merged in to cover the
    rest of the week when a key is available."""
    now = datetime.now(timezone.utc)
    cached = _read_cache(cache_path)
    if cached and not force and (now - cached[1]).total_seconds() / 60 < max_age_min:
        return cached[0], f"Fixtures: {_counts(cached[0])} (cached {cached[1]:%d %b %H:%M} UTC)"
    parts, errors, unresolved = [], [], []
    try:
        df = fetch()
        if len(df):
            parts.append(df)
        else:
            errors.append("football-data fixtures file contained no matches for the five leagues")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"football-data: {type(exc).__name__}: {str(exc)[:80]}")
    if odds_client is not None and known_teams:
        try:
            df, unresolved = fetch_odds_api_fixtures(odds_client, known_teams)
            if len(df):
                parts.append(df)
            else:
                errors.append("Odds API returned no usable fixtures")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"odds-api: {type(exc).__name__}: {str(exc)[:80]}")
    if parts:
        df = combine(parts)
        _write_cache(df, cache_path)
        msg = f"Fixtures: {_counts(df)} (fetched {now:%d %b %H:%M} UTC)"
        if unresolved:
            msg += f". {len(unresolved)} Odds API team names could not be matched and those matches are missing: {unresolved[:6]}"
        if errors:
            msg += f". Note: {'; '.join(errors)}"
        return df, msg
    if cached:
        return cached[0], f"Live fetch failed ({'; '.join(errors)}). Showing fixtures cached {cached[1]:%d %b %H:%M} UTC."
    return pd.DataFrame(columns=COLS), f"No fixtures available ({'; '.join(errors)})."
