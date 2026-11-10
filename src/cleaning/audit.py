"""Dataset-level audits that go beyond row validation."""
from __future__ import annotations
import pandas as pd

KEY = ["league", "season", "date", "home", "away"]
SC = ["ft_home", "ft_away", "ht_home", "ht_away"]


def crosscheck(primary: pd.DataFrame, reference: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare two sources of the same season fixture-by-fixture (e.g. fixturedl vs football-data).

    Fixtures are matched on (league, season, date, home, away) after name canonicalisation.
    Returns (summary, mismatching rows).
    """
    m = primary[KEY + SC].merge(reference[KEY + SC], on=KEY, how="outer",
                                suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"].copy()
    ft_bad = (both.ft_home_a != both.ft_home_b) | (both.ft_away_a != both.ft_away_b)
    ht_bad = (both.ht_home_a != both.ht_home_b) | (both.ht_away_a != both.ht_away_b)
    both["ft_mismatch"], both["ht_mismatch"] = ft_bad, ht_bad
    summary = {
        "matched": len(both),
        "only_in_primary": int((m["_merge"] == "left_only").sum()),
        "only_in_reference": int((m["_merge"] == "right_only").sum()),
        "ft_score_mismatch": int(ft_bad.sum()),
        "ht_score_mismatch": int(ht_bad.sum()),
    }
    return summary, both[ft_bad | ht_bad]


def season_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Per league-season: teams, matches, expected matches (T*(T-1)), and the gap.

    Gaps in a finished season mean missing rows (or a known disruption such as
    Ligue 1 2019/20, which was ended early). Current/partial seasons will show gaps by design.
    """
    long = pd.concat([
        df[["league", "season", "home"]].rename(columns={"home": "team"}),
        df[["league", "season", "away"]].rename(columns={"away": "team"}),
    ])
    out = pd.DataFrame({
        "matches": df.groupby(["league", "season"]).size(),
        "teams": long.groupby(["league", "season"])["team"].nunique(),
    })
    out["expected"] = out["teams"] * (out["teams"] - 1)
    out["missing"] = out["expected"] - out["matches"]
    return out.reset_index()


def team_review(df: pd.DataFrame, current_season: str) -> pd.DataFrame:
    """Teams in the current season never seen in earlier seasons of the same league.

    Genuine promotions show up here, and so does any missing name alias.
    """
    rows = []
    for league, g in df.groupby("league"):
        past = g[g.season != current_season]
        past_teams = set(past.home) | set(past.away)
        cur = g[g.season == current_season]
        for t in sorted((set(cur.home) | set(cur.away)) - past_teams):
            rows.append({"league": league, "team": t})
    return pd.DataFrame(rows, columns=["league", "team"])
