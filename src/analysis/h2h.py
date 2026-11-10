"""Head-to-head meetings and 'when do the goals happen' (first half vs second half), from past league results.

Our data has half-time and full-time scores for every match, so timing is measured per half (not per minute:
goal minutes are not available for most of the history).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def h2h_meetings(df: pd.DataFrame, league: str, home: str, away: str, asof: pd.Timestamp | None = None,
                 n: int = 10) -> pd.DataFrame:
    """Past league meetings of the two teams (either venue), newest first. Result is from `home`'s point of view."""
    m = df[(df.league == league) & (((df.home == home) & (df.away == away)) | ((df.home == away) & (df.away == home)))]
    if asof is not None:
        m = m[m.date < asof]
    m = m.sort_values("date", ascending=False).head(n)
    if m.empty:
        return pd.DataFrame(columns=["Date", "Season", "Home", "Away", "HT", "2H", "FT", "Goals 1H", "Goals 2H", "Total", "Result"])
    mine = np.where(m.home == home, m.ft_home - m.ft_away, m.ft_away - m.ft_home)
    return pd.DataFrame({
        "Date": m.date.dt.strftime("%d %b %Y").values, "Season": m.season.astype(str).values,
        "Home": m.home.values, "Away": m.away.values,
        "HT": (m.ht_home.astype(int).astype(str) + "-" + m.ht_away.astype(int).astype(str)).values,
        "2H": (m.h2_home.astype(int).astype(str) + "-" + m.h2_away.astype(int).astype(str)).values,
        "FT": (m.ft_home.astype(int).astype(str) + "-" + m.ft_away.astype(int).astype(str)).values,
        "Goals 1H": (m.ht_home + m.ht_away).astype(int).values, "Goals 2H": (m.h2_home + m.h2_away).astype(int).values,
        "Total": (m.ft_home + m.ft_away).astype(int).values,
        "Result": np.where(mine > 0, "W", np.where(mine == 0, "D", "L"))})


def h2h_summary(meet: pd.DataFrame) -> dict:
    """Totals over the listed meetings (Result column is from the first-named team's view)."""
    n = len(meet)
    if n == 0:
        return {"n": 0}
    g1, g2, tot = meet["Goals 1H"].sum(), meet["Goals 2H"].sum(), meet["Total"].sum()
    both = meet.FT.map(lambda s: all(int(x) > 0 for x in s.split("-")))
    return {"n": n, "wins": int((meet.Result == "W").sum()), "draws": int((meet.Result == "D").sum()),
            "losses": int((meet.Result == "L").sum()), "avg_goals": tot / n,
            "btts_pct": 100 * both.mean(), "over25_pct": 100 * (meet.Total >= 3).mean(),
            "share_2h": 100 * g2 / tot if tot else float("nan"),
            "goal_in_1h_pct": 100 * (meet["Goals 1H"] > 0).mean(), "goal_in_2h_pct": 100 * (meet["Goals 2H"] > 0).mean()}


def timing_profile(df: pd.DataFrame, league: str, team: str, asof: pd.Timestamp, n: int = 20) -> dict:
    """In the team's last `n` league matches before `asof`: goals scored/conceded in each half."""
    m = df[(df.league == league) & ((df.home == team) | (df.away == team)) & (df.date < asof)]
    m = m.sort_values("date", ascending=False).head(n)
    if m.empty:
        return {"n": 0}
    is_home = (m.home == team).values
    s1 = np.where(is_home, m.ht_home, m.ht_away); c1 = np.where(is_home, m.ht_away, m.ht_home)
    s2 = np.where(is_home, m.h2_home, m.h2_away); c2 = np.where(is_home, m.h2_away, m.h2_home)
    pct = lambda a, b: 100 * a / (a + b) if (a + b) else float("nan")
    return {"n": len(m), "scored_1h": int(s1.sum()), "scored_2h": int(s2.sum()), "conceded_1h": int(c1.sum()),
            "conceded_2h": int(c2.sum()), "scored_2h_share": pct(s2.sum(), s1.sum()),
            "conceded_2h_share": pct(c2.sum(), c1.sum()),
            "scored_in_1h_pct": 100 * (s1 > 0).mean(), "scored_in_2h_pct": 100 * (s2 > 0).mean(),
            "conceded_in_1h_pct": 100 * (c1 > 0).mean(), "conceded_in_2h_pct": 100 * (c2 > 0).mean()}


def league_timing(df: pd.DataFrame, league: str, asof: pd.Timestamp, days: int = 1095) -> dict:
    """Baseline for comparison: the whole league over the training window."""
    m = df[(df.league == league) & (df.date < asof) & (df.date >= asof - pd.Timedelta(days=days))]
    tot = m.ft_total.sum()
    return {"n": len(m), "share_2h": 100 * m.h2_total.sum() / tot if tot else float("nan"),
            "goal_in_1h_pct": 100 * (m.ht_total > 0).mean(), "goal_in_2h_pct": 100 * (m.h2_total > 0).mean()}
