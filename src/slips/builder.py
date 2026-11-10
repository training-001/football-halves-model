"""Slip (accumulator) builder: turn the model's probabilities into candidate legs, filter them by the
user's criteria, and report what a slip really is statistically.

A slip wins only if EVERY leg wins. Legs from different matches are treated as independent, so the chance
that all win is the product of the leg probabilities. The model has no demonstrated edge over bookmaker prices,
so this is a way to look at the numbers, not a recommendation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.markets.grid_markets import market_probs

GROUPS = ["Full-time result", "Double chance", "Full-time goals", "Both teams to score",
          "1st-half result", "2nd-half result", "1st-half goals", "2nd-half goals"]
LEG_COLS = ["league", "home", "away", "kickoff_utc", "group", "key", "selection", "prob"]


def _over(mu: float, line: float) -> float:
    return float(1 - poisson.cdf(int(line), mu))


def build_legs(fx: pd.DataFrame) -> pd.DataFrame:
    """fx needs: league, home, away, kickoff_utc, l1h, l1a, l2h, l2a (expected goals per half)."""
    if fx.empty:
        return pd.DataFrame(columns=LEG_COLS + ["fair_odds"])
    P = market_probs(fx[["l1h", "l1a", "l2h", "l2a"]].astype(float).values)
    rows = []
    for i, r in enumerate(fx.itertuples(index=False)):
        base = {"league": r.league, "home": r.home, "away": r.away, "kickoff_utc": r.kickoff_utc}

        def add(group, key, sel, p):
            rows.append({**base, "group": group, "key": key, "selection": sel, "prob": float(p)})

        h, a = r.home, r.away
        ft, h1, h2 = P["ft_1x2"][i], P["h1_1x2"][i], P["h2_1x2"][i]
        for pre, group, v in [("ft", "Full-time result", ft), ("h1", "1st-half result", h1), ("h2", "2nd-half result", h2)]:
            tag = {"ft": "", "h1": " (1st half)", "h2": " (2nd half)"}[pre]
            add(group, f"{pre}_1", f"{h} win{tag}", v[0])
            add(group, f"{pre}_x", f"Draw{tag}", v[1])
            add(group, f"{pre}_2", f"{a} win{tag}", v[2])
        add("Double chance", "dc_1x", f"{h} or draw", ft[0] + ft[1])
        add("Double chance", "dc_x2", f"{a} or draw", ft[1] + ft[2])
        add("Double chance", "dc_12", f"{h} or {a} (no draw)", ft[0] + ft[2])
        mu1, mu2 = float(r.l1h + r.l1a), float(r.l2h + r.l2a)
        for pre, group, mu, lines in [("ft", "Full-time goals", mu1 + mu2, [1.5, 2.5, 3.5]),
                                      ("h1", "1st-half goals", mu1, [0.5, 1.5]), ("h2", "2nd-half goals", mu2, [0.5, 1.5])]:
            tag = {"ft": "", "h1": " in 1st half", "h2": " in 2nd half"}[pre]
            for ln in lines:
                po = _over(mu, ln)
                add(group, f"{pre}_o{ln}", f"Over {ln} goals{tag}", po)
                add(group, f"{pre}_u{ln}", f"Under {ln} goals{tag}", 1 - po)
        add("Both teams to score", "ft_btts_y", "Both teams to score: Yes", P["ft_btts"][i])
        add("Both teams to score", "ft_btts_n", "Both teams to score: No", 1 - P["ft_btts"][i])
    out = pd.DataFrame(rows)
    out["fair_odds"] = (1 / out.prob).round(2)
    return out


def select_slip(legs: pd.DataFrame, min_prob: float, max_prob: float, n_legs: int, groups=None, leagues=None,
                rank: str = "probability", one_per_match: bool = True) -> pd.DataFrame:
    """Pick up to n_legs legs with min_prob <= probability <= max_prob.
    rank 'probability' = most likely first; 'odds' = lowest probability (highest fair odds) first within the range."""
    pool = legs[(legs.prob >= min_prob) & (legs.prob <= max_prob)]
    if groups:
        pool = pool[pool.group.isin(groups)]
    if leagues:
        pool = pool[pool.league.isin(leagues)]
    pool = pool.sort_values(["prob", "kickoff_utc"], ascending=[rank == "odds", True])   # odds: low prob (high price) first
    chosen, used = [], set()
    for r in pool.itertuples(index=False):
        match = (r.league, r.home, r.away)
        if one_per_match and match in used:
            continue
        chosen.append(r._asdict())
        used.add(match)
        if len(chosen) >= n_legs:
            break
    out = pd.DataFrame(chosen, columns=list(legs.columns))
    return out.sort_values("kickoff_utc").reset_index(drop=True) if len(out) else out


def win_count_distribution(probs) -> np.ndarray:
    """P(exactly k of the legs win), k = 0..n, for independent legs."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return dist


def slip_stats(legs: pd.DataFrame, odds=None) -> dict:
    p = legs.prob.astype(float).values
    comb = float(np.prod(p)) if len(p) else float("nan")
    out = {"n": len(p), "combined_prob": comb, "fair_odds": 1 / comb if comb > 0 else float("inf"),
           "expected_wins": float(p.sum()), "dist": win_count_distribution(p)}
    if odds is not None and len(odds) == len(p) and all(o is not None and o >= 1.01 for o in odds):
        co = float(np.prod(odds))
        out.update({"combined_odds": co, "ev": comb * co - 1,
                    "implied_prob": 1 / co})
    return out
