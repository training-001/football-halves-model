"""Time-decayed, ridge-regularised Poisson team-strength model for ONE half of play.

    log lambda_home = mu + gamma + attack[home] - defence[away]
    log lambda_away = mu +         attack[away] - defence[home]

Fit separately for the first half and for the second half (so each half gets its own
scoring level, home advantage and team strengths). Teams not seen in training get
attack = defence = 0 (the league-average team; the ridge prior mean).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class HalfPoisson:
    def __init__(self, ridge: float = 5.0):
        self.ridge = ridge
        self.teams: dict[str, int] = {}
        self.mu = self.gamma = 0.0
        self.a = self.d = np.zeros(0)

    def fit(self, home, away, hg, ag, w):
        teams = sorted(set(home) | set(away))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = np.fromiter((self.teams[t] for t in home), int, len(home))
        a = np.fromiter((self.teams[t] for t in away), int, len(away))
        hg, ag, w = (np.asarray(x, float) for x in (hg, ag, w))
        lam_ridge = self.ridge

        def f(p):
            mu, gam, att, de = p[0], p[1], p[2:2 + n], p[2 + n:]
            eh = mu + gam + att[h] - de[a]
            ea = mu + att[a] - de[h]
            lh, la = np.exp(eh), np.exp(ea)
            nll = -(w * (hg * eh - lh)).sum() - (w * (ag * ea - la)).sum()
            nll += 0.5 * lam_ridge * ((att ** 2).sum() + (de ** 2).sum())
            rh, ra = w * (lh - hg), w * (la - ag)
            g = np.zeros_like(p)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + n] = np.bincount(h, rh, n) + np.bincount(a, ra, n) + lam_ridge * att
            g[2 + n:] = -np.bincount(a, rh, n) - np.bincount(h, ra, n) + lam_ridge * de
            return nll, g

        p0 = np.zeros(2 + 2 * n)
        p0[0] = np.log(max((hg * w).sum() + (ag * w).sum(), 1e-9) / (2 * w.sum()))
        res = minimize(f, p0, jac=True, method="L-BFGS-B")
        self.mu, self.gamma = res.x[0], res.x[1]
        self.a, self.d = res.x[2:2 + n], res.x[2 + n:]
        return self

    def predict(self, home, away):
        ah = np.array([self.a[self.teams[t]] if t in self.teams else 0.0 for t in home])
        dh = np.array([self.d[self.teams[t]] if t in self.teams else 0.0 for t in home])
        aa = np.array([self.a[self.teams[t]] if t in self.teams else 0.0 for t in away])
        da = np.array([self.d[self.teams[t]] if t in self.teams else 0.0 for t in away])
        return np.exp(self.mu + self.gamma + ah - da), np.exp(self.mu + aa - dh)


def decay_weights(age_days: np.ndarray, half_life: float) -> np.ndarray:
    return np.exp(-np.log(2) * age_days / half_life)
