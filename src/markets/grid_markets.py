"""Turn four Poisson rates (1H home/away, 2H home/away) into market probabilities.

Assumption (tested separately): goals in the two halves are independent given the rates.
Under that assumption full-time home/away goals are independent Poissons with summed rates.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson

K = 9            # goals 0..8 per half per team in the HT/FT grid
KF = 16          # full-time goal grid
RES = ["H", "D", "A"]


def _res_probs(lh, la, k):
    g = np.arange(k)
    P = poisson.pmf(g[None, :, None], lh[:, None, None]) * poisson.pmf(g[None, None, :], la[:, None, None])
    i, j = np.meshgrid(g, g, indexing="ij")
    out = np.stack([(P * (i > j)).sum((1, 2)), (P * (i == j)).sum((1, 2)), (P * (i < j)).sum((1, 2))], 1)
    return out / out.sum(1, keepdims=True)


def _htft(l1h, l1a, l2h, l2a, chunk=400):
    g = np.arange(K)
    i, j, k, l = np.meshgrid(g, g, g, g, indexing="ij")
    ft = np.sign((i + k) - (j + l))               # 1 home, 0 draw, -1 away
    ft_idx = np.where(ft > 0, 0, np.where(ft == 0, 1, 2))
    ht_idx = np.where(i > j, 0, np.where(i == j, 1, 2))
    masks = [[(ht_idx == r) & (ft_idx == R) for R in range(3)] for r in range(3)]
    n = len(l1h)
    out = np.zeros((n, 9))
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        p = lambda lam: poisson.pmf(g[None, :], lam[sl, None])
        G1 = p(l1h)[:, :, None] * p(l1a)[:, None, :]
        G2 = p(l2h)[:, :, None] * p(l2a)[:, None, :]
        G = G1[:, :, :, None, None] * G2[:, None, None, :, :]   # (n, i, j, k, l)
        for r in range(3):
            for R in range(3):
                out[sl, 3 * r + R] = (G * masks[r][R]).sum((1, 2, 3, 4))
    return out / out.sum(1, keepdims=True)


def market_probs(lam: np.ndarray) -> dict[str, np.ndarray]:
    """lam: (N,4) = [1H home, 1H away, 2H home, 2H away] expected goals."""
    l1h, l1a, l2h, l2a = lam.T
    fh, fa = l1h + l2h, l1a + l2a
    out = {
        "h1_1x2": _res_probs(l1h, l1a, K),
        "h2_1x2": _res_probs(l2h, l2a, K),
        "ft_1x2": _res_probs(fh, fa, KF),
        "h1_o05": 1 - np.exp(-(l1h + l1a)),
        "h1_o15": 1 - poisson.cdf(1, l1h + l1a),
        "h2_o05": 1 - np.exp(-(l2h + l2a)),
        "h2_o15": 1 - poisson.cdf(1, l2h + l2a),
        "ft_o25": 1 - poisson.cdf(2, fh + fa),
        "h1_btts": (1 - np.exp(-l1h)) * (1 - np.exp(-l1a)),
        "h2_btts": (1 - np.exp(-l2h)) * (1 - np.exp(-l2a)),
        "ft_btts": (1 - np.exp(-fh)) * (1 - np.exp(-fa)),
        "htft": _htft(l1h, l1a, l2h, l2a),
    }
    return out


def score_grid(lh: float, la: float, k: int = 8) -> np.ndarray:
    """P(home goals = i, away goals = j) for i, j in 0..k-1 (independent Poissons)."""
    g = np.arange(k)
    return np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
