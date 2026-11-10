"""Model 'leans': which outcome the model rates most likely, and how strongly.

IMPORTANT: a lean is the MOST LIKELY outcome, not a recommended bet. Whether a price is good
value needs a bookmaker price, and in backtests the model trails closing prices on full-time markets.

Confidence tiers are by stated probability. Backtests (2021/22-2025/26, 8,907 matches) show the model's
stated probabilities track actual hit rates closely, which is what justifies using them as labels.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson

TIERS_3WAY = [(0.60, "Strong"), (0.50, "Moderate"), (0.45, "Slight")]
TIERS_BINARY = [(0.65, "Strong"), (0.58, "Moderate"), (0.53, "Slight")]
NONE = "No clear lean"
RANK = {"Strong": 3, "Moderate": 2, "Slight": 1, NONE: 0}


def tier(p: float, kind: str = "3way") -> str:
    for cut, name in (TIERS_3WAY if kind == "3way" else TIERS_BINARY):
        if p >= cut:
            return name
    return NONE


def over_prob(mu: float, line: float) -> float:
    return float(1 - poisson.cdf(int(line), mu))


def closest_line(mu: float, lines: list[float]) -> tuple[float, float]:
    """The over/under line whose over-probability is nearest 50% (the most informative one)."""
    best = min(lines, key=lambda ln: abs(over_prob(mu, ln) - 0.5))
    return best, over_prob(mu, best)


def lean_table(P: dict, lam: np.ndarray, home: str, away: str) -> pd.DataFrame:
    """One row per headline market. P = market_probs output for a single match (index 0)."""
    l1h, l1a, l2h, l2a = lam
    rows = []

    def add3(name, p):
        i = int(np.argmax(p))
        rows.append((name, [f"{home} win", "Draw", f"{away} win"][i], float(p[i]), tier(p[i], "3way")))

    def addb(name, p_yes, yes, no):
        side, pm = (yes, p_yes) if p_yes >= 0.5 else (no, 1 - p_yes)
        rows.append((name, side, float(pm), tier(pm, "binary")))

    add3("Full-time result", P["ft_1x2"][0])
    add3("1st-half result", P["h1_1x2"][0])
    add3("2nd-half result", P["h2_1x2"][0])
    for name, mu, lines in [("Full-time goals", l1h + l1a + l2h + l2a, [1.5, 2.5, 3.5]),
                            ("1st-half goals", l1h + l1a, [0.5, 1.5, 2.5]),
                            ("2nd-half goals", l2h + l2a, [0.5, 1.5, 2.5])]:
        ln, p = closest_line(mu, lines)
        addb(f"{name} (line {ln})", p, f"Over {ln}", f"Under {ln}")
    addb("Full-time both teams score", float(P["ft_btts"][0]), "Yes", "No")
    out = pd.DataFrame(rows, columns=["Market", "Model lean", "Probability", "Confidence"])
    out["Fair odds"] = (1 / out["Probability"]).round(2)
    return out


def headline(leans: pd.DataFrame) -> str:
    ft = leans.iloc[0]
    if ft["Confidence"] in ("Strong", "Moderate"):          # prefer the full-time result when it has a real lean
        return f"**{ft['Model lean']}** (full-time result), {100 * ft['Probability']:.0f}%, {ft['Confidence'].lower()} lean"
    best = leans.assign(_r=leans.Confidence.map(RANK)).sort_values(["_r", "Probability"], ascending=False).iloc[0]
    if best["_r"] == 0:
        return "No strong lean: the model sees this match as open."
    return f"**{best['Model lean']}** ({best['Market'].lower()}), {100 * best['Probability']:.0f}%, {best['Confidence'].lower()} lean"
