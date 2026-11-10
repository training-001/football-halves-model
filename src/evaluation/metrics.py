"""Log-loss scoring for every market, model vs naive vs bookmaker closing line."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.markets.grid_markets import market_probs

EPS = 1e-12


def _idx(h, a):
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def outcomes(d: pd.DataFrame) -> dict[str, np.ndarray]:
    ht, h2, ft = _idx(d.ht_home, d.ht_away), _idx(d.h2_home, d.h2_away), _idx(d.ft_home, d.ft_away)
    return {
        "h1_1x2": ht, "h2_1x2": h2, "ft_1x2": ft, "htft": 3 * ht + ft,
        "h1_o05": (d.ht_total > 0.5).values, "h1_o15": (d.ht_total > 1.5).values,
        "h2_o05": (d.h2_total > 0.5).values, "h2_o15": (d.h2_total > 1.5).values,
        "ft_o25": (d.ft_total > 2.5).values,
        "h1_btts": ((d.ht_home > 0) & (d.ht_away > 0)).values,
        "h2_btts": ((d.h2_home > 0) & (d.h2_away > 0)).values,
        "ft_btts": ((d.ft_home > 0) & (d.ft_away > 0)).values,
    }


def ll_cat(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))


def ll_bin(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.where(y, np.log(p), np.log(1 - p))


def score_markets(d: pd.DataFrame, lam: pd.DataFrame) -> pd.DataFrame:
    """Per-match log loss for model (m*) and naive (n*) for every market."""
    d = d.loc[lam.index]
    y = outcomes(d)
    pm = market_probs(lam[["m1h", "m1a", "m2h", "m2a"]].values)
    pn = market_probs(lam[["n1h", "n1a", "n2h", "n2a"]].values)
    rows = {}
    for k, yy in y.items():
        f = ll_cat if pm[k].ndim == 2 else ll_bin
        rows[f"{k}__model"], rows[f"{k}__naive"] = f(pm[k], yy), f(pn[k], yy)
    # bookmaker closing line (market average, proportional de-vig) for full-time markets
    if {"AvgCH", "AvgCD", "AvgCA"} <= set(d.columns):
        inv = 1 / d[["AvgCH", "AvgCD", "AvgCA"]].values
        P = inv / inv.sum(1, keepdims=True)
        ok = ~np.isnan(P).any(1)
        v = np.full(len(d), np.nan); v[ok] = ll_cat(P[ok], y["ft_1x2"][ok])
        rows["ft_1x2__market"] = v
    if {"AvgC_o25", "AvgC_u25"} <= set(d.columns):
        io, iu = 1 / d.AvgC_o25.values, 1 / d.AvgC_u25.values
        p = io / (io + iu)
        ok = ~np.isnan(p)
        v = np.full(len(d), np.nan); v[ok] = ll_bin(p[ok], y["ft_o25"][ok])
        rows["ft_o25__market"] = v
    out = pd.DataFrame(rows, index=d.index)
    out["league"], out["season"] = d.league.values, d.season.values
    return out
