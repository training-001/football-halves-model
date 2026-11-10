"""Short, factual reasons behind a prediction, built from the fitted team strengths."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def strength_ranks(m1, m2, tr: pd.DataFrame) -> pd.DataFrame:
    """Attack and defence rank (1 = best) among teams in the league's latest season of the training window."""
    cur = tr[tr.season == tr.season.max()]
    teams = [t for t in sorted(set(cur.home) | set(cur.away)) if t in m1.teams and t in m2.teams]
    rows = []
    for t in teams:
        i, j = m1.teams[t], m2.teams[t]
        rows.append({"team": t, "att": (m1.a[i] + m2.a[j]) / 2, "dfn": (m1.d[i] + m2.d[j]) / 2})
    s = pd.DataFrame(rows).set_index("team")
    s["att_rank"] = s.att.rank(ascending=False, method="min").astype(int)
    s["def_rank"] = s.dfn.rank(ascending=False, method="min").astype(int)
    return s


def reasons(m1, m2, tr, home, away, lam, leans, cal=None) -> list[str]:
    out = []
    s = strength_ranks(m1, m2, tr)
    n = len(s)
    if home in s.index and away in s.index:
        h, a = s.loc[home], s.loc[away]
        out.append(f"**Strength** (rank of {n}, 1 = best): {home} attack {_ordinal(int(h.att_rank))}, defence {_ordinal(int(h.def_rank))}; "
                   f"{away} attack {_ordinal(int(a.att_rank))}, defence {_ordinal(int(a.def_rank))}.")
    l1h, l1a, l2h, l2a = lam
    g1, g2 = tr.ht_total.mean(), tr.h2_total.mean()
    goals = leans[leans.Market.str.startswith("Full-time goals")].iloc[0]
    out.append(f"**Goals:** model expects {l1h + l1a:.1f} in the 1st half (league average {g1:.1f}) and {l2h + l2a:.1f} "
               f"in the 2nd (average {g2:.1f}), {l1h + l1a + l2h + l2a:.1f} in total; {goals['Model lean'].lower()} at "
               f"{100 * goals['Probability']:.0f}%.")
    ft = leans.iloc[0]
    if cal is not None and "ft_1x2" in cal:
        t = cal["ft_1x2"]
        hit = t[(t.lo <= ft.Probability) & (ft.Probability < t.hi)]
        if len(hit):
            r = hit.iloc[0]
            out.append(f"**Backtest check:** full-time results the model rated {100 * r.lo:.0f}-{100 * min(r.hi, 1):.0f}% for the "
                       f"top outcome won {100 * r.hit_rate:.0f}% of the time ({int(r.n):,} matches).")
    return out
