"""Load the original per-league CSVs (the fixturedl / openligadb downloader output).

Design notes
------------
* Only files named like  <CTRY>_<tier>_<league>_<season>.csv  are read.
  Files ending in `_training_eligible.csv` are skipped: in the delivered archive
  they are byte-identical to the per-league files (verified by MD5), and the
  combined `all_matches_training_eligible.csv` is exactly their union.
  Re-reading them is what created the duplicates in Book1.ods.
* Deduplication is by fixture key (league, season, date, home, away), never by filename.
* Conflicting duplicates (same key, different scores) raise an error.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, REQUIRED

_PATTERN = re.compile(r"^[A-Z]{3}_\d+_.+_\d{4}\.csv$")
KEY = ["league", "season", "date", "home", "away"]
SCORES = ["ft_home", "ft_away", "ht_home", "ht_away"]


def list_source_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    return sorted(p for p in raw_dir.glob("*.csv")
                  if _PATTERN.match(p.name) and not p.stem.endswith("_training_eligible"))


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = list_source_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No league CSVs found in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["season"] = df["season"].astype(str).str.strip()   # feed stores 2627 as int; football-data uses text
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in SCORES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        conflicts = df[dup].groupby(KEY)[SCORES].nunique(dropna=False).gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"{int(conflicts.sum())} fixtures have conflicting scores")
        df = df.drop_duplicates(KEY, keep="first")

    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)


def add_half_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Second-half goals = full-time minus half-time."""
    out = df.copy()
    out["h2_home"] = out["ft_home"] - out["ht_home"]
    out["h2_away"] = out["ft_away"] - out["ht_away"]
    out["ht_total"] = out["ht_home"] + out["ht_away"]
    out["h2_total"] = out["h2_home"] + out["h2_away"]
    out["ft_total"] = out["ft_home"] + out["ft_away"]
    out["has_scores"] = out[SCORES].notna().all(axis=1)
    return out
