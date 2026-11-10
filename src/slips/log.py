"""Saved slips and their settlement against real results (pre-kickoff legs only, never edited)."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

SLIP_PATH = ROOT / "data" / "records" / "slips.csv"
SLIP_COLS = ["slip_id", "created_utc", "name", "leg_no", "league", "home", "away", "kickoff_utc", "group", "key",
             "selection", "model_prob", "odds", "criteria"]


def load_slips(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else SLIP_PATH
    if not path.exists():
        return pd.DataFrame(columns=SLIP_COLS)
    df = pd.read_csv(path)
    for c in SLIP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SLIP_COLS]


def save_slip(legs: pd.DataFrame, criteria: dict, name: str = "", odds=None, path: Path | None = None,
              now: datetime | None = None) -> str:
    """Append a slip. Only legs that have not kicked off can be saved. Returns the slip id."""
    path = Path(path) if path else SLIP_PATH
    now = now or datetime.now(timezone.utc)
    legs = legs.reset_index(drop=True).copy()
    legs["odds"] = list(odds) if odds is not None else np.nan
    ko = pd.to_datetime(legs.kickoff_utc, utc=True)
    if (ko <= pd.Timestamp(now)).any():
        raise ValueError("A leg has already kicked off; slips can only be saved before kickoff.")
    if legs.empty:
        raise ValueError("The slip has no legs.")
    slip_id = f"S{now:%Y%m%d-%H%M%S}"
    new = pd.DataFrame({
        "slip_id": slip_id, "created_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "name": name, "leg_no": range(1, len(legs) + 1),
        "league": legs.league, "home": legs.home, "away": legs.away,
        "kickoff_utc": ko.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "group": legs.group, "key": legs.key,
        "selection": legs.selection, "model_prob": legs.prob.astype(float), "odds": legs.odds,
        "criteria": json.dumps(criteria, sort_keys=True)})
    merged = pd.concat([load_slips(path), new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return slip_id


def leg_won(key: str, ft_home, ft_away, ht_home, ht_away) -> bool:
    """Did the leg identified by `key` win, given the half-time and full-time scores?"""
    h2h, h2a = ft_home - ht_home, ft_away - ht_away
    parts = key.split("_")
    scope = parts[0]
    if key in ("ft_btts_y", "ft_btts_n"):
        both = ft_home > 0 and ft_away > 0
        return both if key.endswith("_y") else not both
    if scope == "dc":
        return {"1x": ft_home >= ft_away, "x2": ft_home <= ft_away, "12": ft_home != ft_away}[parts[1]]
    goals = {"ft": (ft_home, ft_away), "h1": (ht_home, ht_away), "h2": (h2h, h2a)}[scope]
    tag = parts[1]
    if tag in ("1", "x", "2"):
        return {"1": goals[0] > goals[1], "x": goals[0] == goals[1], "2": goals[0] < goals[1]}[tag]
    total, line = goals[0] + goals[1], float(tag[1:])
    return total > line if tag[0] == "o" else total < line


def settle_slips(slips: pd.DataFrame, matches: pd.DataFrame, max_days: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (leg-level table with status, slip-level summary). Status: won / lost / pending."""
    if slips.empty:
        return pd.DataFrame(), pd.DataFrame()
    legs = slips.copy()
    legs["ko"] = pd.to_datetime(legs.kickoff_utc, utc=True)
    legs["ko_date"] = legs.ko.dt.tz_localize(None).dt.normalize()
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    legs["_i"] = range(len(legs))
    j = legs.merge(m[["league", "home", "away", "date", "ft_home", "ft_away", "ht_home", "ht_away"]],
                   on=["league", "home", "away"], how="left")
    j["gap"] = (j["date"] - j["ko_date"]).abs().dt.days
    ok = j[j.gap <= max_days].sort_values("gap").drop_duplicates("_i")           # the result of THIS fixture
    # a leg whose teams met before (other seasons) must not vanish: with no result in the window it is pending
    waiting = legs[~legs._i.isin(set(ok._i))].copy()
    for c in ("date", "ft_home", "ft_away", "ht_home", "ht_away"):
        waiting[c] = np.nan
    j = pd.concat([ok, waiting], ignore_index=True)
    status = []
    for r in j.itertuples():
        if pd.isna(r.date):
            status.append("pending")
        else:
            status.append("won" if leg_won(r.key, r.ft_home, r.ft_away, r.ht_home, r.ht_away) else "lost")
    j["status"] = status
    leg_tbl = j.sort_values(["created_utc", "leg_no"]).drop(columns=["ko", "ko_date", "_i", "gap"])
    g = leg_tbl.groupby("slip_id", sort=False)
    summ = g.agg(created_utc=("created_utc", "first"), name=("name", "first"), legs=("leg_no", "size"),
                 won=("status", lambda s: (s == "won").sum()), lost=("status", lambda s: (s == "lost").sum()),
                 pending=("status", lambda s: (s == "pending").sum()), model_prob=("model_prob", "prod"),
                 odds=("odds", lambda s: s.prod() if s.notna().all() else np.nan)).reset_index()
    summ["status"] = np.where(summ.lost > 0, "lost", np.where(summ.won == summ.legs, "won", "pending"))
    return leg_tbl.reset_index(drop=True), summ.sort_values("created_utc", ascending=False).reset_index(drop=True)
