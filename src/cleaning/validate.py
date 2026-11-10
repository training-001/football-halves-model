"""Integrity checks. Returns a dict of check -> count of offending rows (0 = pass)."""
from __future__ import annotations
import pandas as pd


def _res(h, a):
    return pd.Series(["H" if x > y else "A" if x < y else "D" for x, y in zip(h, a)], index=h.index)


def validate(df: pd.DataFrame) -> dict[str, int]:
    s = df[df["has_scores"]]
    sc = ["ft_home", "ft_away", "ht_home", "ht_away"]
    checks = {
        "rows_without_scores": int((~df["has_scores"]).sum()),
        "duplicate_fixtures": int(df.duplicated(["league", "season", "date", "home", "away"]).sum()),
        "negative_scores": int((s[sc] < 0).any(axis=1).sum()),
        "ht_exceeds_ft": int(((s.ht_home > s.ft_home) | (s.ht_away > s.ft_away)).sum()),
        "team_plays_itself": int((df.home == df.away).sum()),
        "non_integer_scores": int((s[sc] % 1 != 0).any(axis=1).sum()),
    }
    if "ft_result" in s:
        checks["ft_result_mismatch"] = int((_res(s.ft_home, s.ft_away) != s.ft_result).sum())
    if "ht_result" in s:
        checks["ht_result_mismatch"] = int((_res(s.ht_home, s.ht_away) != s.ht_result).sum())
    return checks


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    s = df[df["has_scores"]]
    out = (s.groupby("league")
              .agg(matches=("date", "size"), first=("date", "min"), last=("date", "max"),
                   teams=("home", "nunique"),
                   avg_1h_goals=("ht_total", "mean"), avg_2h_goals=("h2_total", "mean"),
                   avg_ft_goals=("ft_total", "mean"))
              )
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out
