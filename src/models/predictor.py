"""Single-fixture prediction using exactly the same fitting recipe as the walk-forward backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.poisson_halves import HalfPoisson, decay_weights


def fit_league_models(df: pd.DataFrame, league: str, asof: pd.Timestamp, half_life: float,
                      ridge: float, train_days: int = 1095):
    """Fit first-half and second-half models on the league's matches played before `asof`."""
    tr = df[(df.league == league) & (df.date < asof) & (df.date >= asof - pd.Timedelta(days=train_days))]
    if len(tr) < 50:
        raise ValueError(f"Only {len(tr)} training matches for {league} before {asof:%Y-%m-%d}")
    w = decay_weights((asof - tr.date).dt.days.values, half_life)
    m1 = HalfPoisson(ridge).fit(tr.home.values, tr.away.values, tr.ht_home.values, tr.ht_away.values, w)
    m2 = HalfPoisson(ridge).fit(tr.home.values, tr.away.values, tr.h2_home.values, tr.h2_away.values, w)
    return m1, m2, tr


def predict_lambdas(m1: HalfPoisson, m2: HalfPoisson, home: str, away: str) -> np.ndarray:
    """Expected goals [1H home, 1H away, 2H home, 2H away]."""
    a, b = m1.predict([home], [away])
    c, d = m2.predict([home], [away])
    return np.array([a[0], b[0], c[0], d[0]])


def is_known(model: HalfPoisson, team: str) -> bool:
    return team in model.teams
