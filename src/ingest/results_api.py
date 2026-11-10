"""Faster results from football-data.org (free plan, needs a free token), merged into matches_all.csv.

football-data.co.uk (the main source) is updated in batches, so the newest matches can be days behind.
This module fills the gap. Rules:
  * Only FINISHED matches with BOTH full-time and half-time scores are added (a match without a
    half-time score is useless to this model and is only counted in the report).
  * Existing rows are never changed. football-data.co.uk rows replace these when it catches up.
  * Team names are mapped to the canonical names; unmatched names are reported, never guessed.
The free plan's scores are documented as delayed, and whether it includes half-time scores is checked
live by this code and reported ('no_halftime').
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.cleaning.validate import validate
from src.ingest.fixtures import resolve_team
from src.ingest.load_raw import add_half_scores

BASE = "https://api.football-data.org/v4"
COMPETITIONS = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A", "BL1": "Bundesliga", "FL1": "Ligue 1"}
DERIVED = ["h2_home", "h2_away", "ht_total", "h2_total", "ft_total", "has_scores"]
SOURCE = "football-data.org"


def season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def parse_matches(league: str, payload: dict, canon_teams: set[str], season: str) -> tuple[list[dict], dict]:
    rows, st = [], {"seen": 0, "no_halftime": 0, "bad_scores": 0, "unresolved": set()}
    for m in payload.get("matches", []):
        if m.get("status") != "FINISHED":
            continue
        st["seen"] += 1
        sc = m.get("score") or {}
        ft, ht = sc.get("fullTime") or {}, sc.get("halfTime") or {}
        vals = [ft.get("home"), ft.get("away"), ht.get("home"), ht.get("away")]
        if vals[0] is None or vals[1] is None:
            st["bad_scores"] += 1
            continue
        if vals[2] is None or vals[3] is None:
            st["no_halftime"] += 1
            continue
        if vals[2] > vals[0] or vals[3] > vals[1]:
            st["bad_scores"] += 1
            continue
        h = resolve_team((m.get("homeTeam") or {}).get("name", ""), canon_teams)
        a = resolve_team((m.get("awayTeam") or {}).get("name", ""), canon_teams)
        if h is None or a is None:
            st["unresolved"].update(n for n, v in (((m.get("homeTeam") or {}).get("name"), h),
                                                   ((m.get("awayTeam") or {}).get("name"), a)) if v is None)
            continue
        rows.append({"league": league, "season": season, "date": pd.Timestamp(m["utcDate"]).tz_convert("UTC").tz_localize(None).normalize(),
                     "home": h, "away": a, "ft_home": vals[0], "ft_away": vals[1], "ht_home": vals[2], "ht_away": vals[3],
                     "source": SOURCE})
    return rows, st


def fetch_finished(token: str, year: int, known_teams: dict[str, set[str]], pause: float = 1.0):
    """One request per competition (the free plan allows 10 per minute). Returns (rows DataFrame, report)."""
    import requests
    season = season_code(year)
    rows, report = [], {"http": {}, "seen": 0, "no_halftime": 0, "bad_scores": 0, "unresolved": set()}
    for code, league in COMPETITIONS.items():
        r = requests.get(f"{BASE}/competitions/{code}/matches", params={"season": year, "status": "FINISHED"},
                         headers={"X-Auth-Token": token}, timeout=30)
        report["http"][code] = r.status_code
        if r.status_code == 200:
            got, st = parse_matches(league, r.json(), known_teams.get(league, set()), season)
            rows += got
            for k in ("seen", "no_halftime", "bad_scores"):
                report[k] += st[k]
            report["unresolved"] |= st["unresolved"]
        time.sleep(pause)
    return pd.DataFrame(rows), report


def merge_api_rows(all_df: pd.DataFrame, api: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add API matches whose (league, season, home, away) is not already present."""
    if api.empty:
        return all_df, 0
    key = ["league", "season", "home", "away"]
    have = all_df[key].astype(str).agg("|".join, axis=1)
    new = api[~api[key].astype(str).agg("|".join, axis=1).isin(set(have))]
    if new.empty:
        return all_df, 0
    out = pd.concat([all_df.drop(columns=DERIVED, errors="ignore"), new], ignore_index=True)
    out = add_half_scores(out).sort_values(["date", "league", "home"]).reset_index(drop=True)
    return out, len(new)


def refresh_from_api(token: str, path: Path = PROCESSED_DIR / "matches_all.csv", year: int = 2026, fetcher=fetch_finished) -> dict:
    all_df = pd.read_csv(path, parse_dates=["date"])
    all_df["season"] = all_df["season"].astype(str)
    known = {}
    for lg, g in all_df.groupby("league"):
        cur = g[g.season == g.season.max()]
        known[lg] = set(cur.home) | set(cur.away)
    api, report = fetcher(token, year, known)
    out, added = merge_api_rows(all_df, api)
    problems = {k: v for k, v in validate(out).items() if v}
    if problems:
        raise ValueError(f"Refusing to write {path.name}: validation {problems}")
    if added:
        tmp = path.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        path.with_suffix(".bak.csv").write_bytes(path.read_bytes())
        os.replace(tmp, path)
    report.update({"added": added, "fetched": len(api), "latest_result": str(out.date.max().date())})
    report["unresolved"] = sorted(report["unresolved"])
    return report
