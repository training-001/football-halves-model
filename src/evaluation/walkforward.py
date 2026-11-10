"""Strictly time-ordered walk-forward backtest.

For every block of `step_days` inside a test season, the model is fitted ONLY on matches
of the same league played before the block starts (within `train_days`), with exponential
time decay. Matches in the block are then predicted. Nothing from the block or later is seen.
A naive baseline (league-level home/away rates per half, same weights, no team effects)
is produced from exactly the same training data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.poisson_halves import HalfPoisson, decay_weights


def walk_forward(df: pd.DataFrame, test_seasons: list[str], half_life: float, ridge: float,
                 step_days: int = 28, train_days: int = 1095) -> pd.DataFrame:
    cols = ["m1h", "m1a", "m2h", "m2a", "n1h", "n1a", "n2h", "n2a"]
    parts = []
    for league, g in df.groupby("league"):
        g = g.sort_values("date")
        for season in test_seasons:
            gs = g[g.season == season]
            if gs.empty:
                continue
            cut, end = gs.date.min(), gs.date.max()
            while cut <= end:
                nxt = cut + pd.Timedelta(days=step_days)
                test = gs[(gs.date >= cut) & (gs.date < nxt)]
                if len(test):
                    tr = g[(g.date < cut) & (g.date >= cut - pd.Timedelta(days=train_days))]
                    w = decay_weights((cut - tr.date).dt.days.values, half_life)
                    out = np.zeros((len(test), 8))
                    for k, (hh, aa) in enumerate([("ht_home", "ht_away"), ("h2_home", "h2_away")]):
                        m = HalfPoisson(ridge).fit(tr.home.values, tr.away.values, tr[hh].values, tr[aa].values, w)
                        lh, la = m.predict(test.home.values, test.away.values)
                        out[:, 2 * k], out[:, 2 * k + 1] = lh, la
                        out[:, 4 + 2 * k] = (w * tr[hh].values).sum() / w.sum()
                        out[:, 5 + 2 * k] = (w * tr[aa].values).sum() / w.sum()
                    parts.append(pd.DataFrame(out, columns=cols, index=test.index))
                cut = nxt
    return pd.concat(parts).sort_index()


def poisson_ll(df: pd.DataFrame, lam: pd.DataFrame, prefix: str = "m") -> float:
    """Mean per-team-per-half Poisson log-likelihood (higher is better); used for tuning."""
    from scipy.special import gammaln
    tot = 0.0
    for c, y in [("1h", "ht_home"), ("1a", "ht_away"), ("2h", "h2_home"), ("2a", "h2_away")]:
        l, yy = lam[f"{prefix}{c}"].values, df.loc[lam.index, y].values
        tot += (yy * np.log(l) - l - gammaln(yy + 1)).mean()
    return tot / 4
