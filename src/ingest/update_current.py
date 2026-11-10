"""Refresh the current season's results from football-data.co.uk without a full rebuild.

The 2026/27 feed delivered earlier is a static snapshot (it ends 2026-10-29). football-data's own
2026/27 files are updated as matches are played, and they matched that snapshot on all 404 matches
(zero score differences), so they are used for the current season from now on. They also carry
bookmaker odds, which the snapshot lacked.
"""
from __future__ import annotations
import os
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, ROOT
from src.cleaning.team_names import canonical
from src.cleaning.validate import validate
from src.ingest.football_data import load_seasons
from src.ingest.load_raw import add_half_scores
from src.ingest.odds_api import find_secret
from src.ingest.results_api import refresh_from_api

CURRENT = "2627"
CACHE_DIR = ROOT / "data" / "external" / "football_data"
KEY = ["league", "season", "date", "home", "away"]
SCORES = ["ft_home", "ft_away", "ht_home", "ht_away"]
DERIVED = ["h2_home", "h2_away", "ht_total", "h2_total", "ft_total", "has_scores"]


def merge_current(all_df: pd.DataFrame, fd: pd.DataFrame, season: str = CURRENT) -> tuple[pd.DataFrame, dict]:
    """Replace the season's rows by football-data's, keeping any existing rows it lacks."""
    fd = fd.copy()
    fd["home"], fd["away"] = fd["home"].map(canonical), fd["away"].map(canonical)
    cur = all_df[all_df.season == season]
    both = cur[KEY + SCORES].merge(fd[KEY + SCORES], on=KEY, suffixes=("_old", "_new"))
    conflicts = int(sum((both[c + "_old"] != both[c + "_new"]).sum() for c in SCORES))
    only_old = cur.merge(fd[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    rest = all_df[all_df.season != season]
    out = pd.concat([rest, fd, only_old], ignore_index=True).drop(columns=DERIVED, errors="ignore")
    out = add_half_scores(out).sort_values(["date", "league", "home"]).reset_index(drop=True)
    info = {"before": len(cur), "after": int((out.season == season).sum()),
            "new_matches": int((out.season == season).sum()) - len(cur),
            "score_conflicts": conflicts, "kept_from_old_feed": len(only_old)}
    return out, info


def refresh_current(path: Path = PROCESSED_DIR / "matches_all.csv", season: str = CURRENT,
                    cache_dir: Path = CACHE_DIR, loader=load_seasons) -> dict:
    all_df = pd.read_csv(path, parse_dates=["date"])
    all_df["season"] = all_df["season"].astype(str)
    fd, _ = loader([season], cache_dir, force=True, pause=0.2)
    out, info = merge_current(all_df, fd, season)
    problems = {k: v for k, v in validate(out).items() if v}
    if problems:
        raise ValueError(f"Refusing to overwrite {path.name}: validation {problems}")
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    if path.exists():
        path.with_suffix(".bak.csv").write_bytes(path.read_bytes())
    os.replace(tmp, path)
    info["latest_result"] = str(out.date.max().date())
    return info


def refresh_all(path: Path = PROCESSED_DIR / "matches_all.csv", token: str | None = None) -> dict:
    """football-data.co.uk first (authoritative), then football-data.org if a token is configured."""
    out = {"fd": None, "fd_error": None, "api": None, "api_error": None, "api_enabled": False}
    try:
        out["fd"] = refresh_current(path)
    except Exception as exc:  # noqa: BLE001
        out["fd_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    token = token or find_secret("FOOTBALL_DATA_ORG_TOKEN", required=False)[0]
    if token:
        out["api_enabled"] = True
        try:
            out["api"] = refresh_from_api(token, path)
        except Exception as exc:  # noqa: BLE001
            out["api_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return out


def summarize(res: dict) -> list[str]:
    """Plain-language lines describing what a refresh did."""
    lines = []
    if res["fd_error"]:
        lines.append(f"football-data.co.uk: FAILED ({res['fd_error']})")
    elif res["fd"]:
        f = res["fd"]
        msg = f"football-data.co.uk: {f['new_matches']} new matches; its file runs to {f['latest_result']}."
        if f["new_matches"] == 0:
            msg += " (It is updated in batches, often days after the matches.)"
        if f.get("score_conflicts"):
            msg += f" {f['score_conflicts']} score disagreements with earlier rows were resolved in its favour."
        lines.append(msg)
    if res["api_enabled"]:
        if res["api_error"]:
            lines.append(f"football-data.org: FAILED ({res['api_error']})")
        else:
            a = res["api"]
            msg = f"football-data.org: added {a['added']} matches (finished matches seen: {a['seen']})."
            if a["no_halftime"]:
                msg += f" {a['no_halftime']} had no half-time score in the free plan and were skipped."
            if a["unresolved"]:
                msg += f" Unmatched team names: {a['unresolved'][:6]}."
            bad = {k: v for k, v in a["http"].items() if v != 200}
            if bad:
                msg += f" HTTP problems: {bad}."
            lines.append(msg)
    else:
        lines.append("football-data.org: not set up (optional faster source, see README).")
    return lines
