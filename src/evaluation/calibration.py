"""Reliability of the model's stated probabilities, from the walk-forward backtest predictions."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import REPORT_DIR
from src.evaluation.metrics import outcomes
from src.markets.grid_markets import market_probs

BINS_3 = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01]
BINS_2 = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]


def calibration_tables(df: pd.DataFrame, pred_path=REPORT_DIR / "backtest_predictions_test_2122_2526.csv") -> dict:
    if not pred_path.exists():
        return {}
    pr = pd.read_csv(pred_path, parse_dates=["date"])
    pr["season"] = pr["season"].astype(str)
    m = pr.merge(df, on=["league", "season", "date", "home", "away"], how="inner")
    if m.empty:
        return {}
    P, y = market_probs(m[["m1h", "m1a", "m2h", "m2a"]].values), outcomes(m)
    out = {}
    for k in ("ft_1x2", "h1_1x2", "h2_1x2"):
        p = P[k]
        stat, hit = p.max(1), p.argmax(1) == y[k]
        out[k] = _bin(stat, hit, BINS_3)
    for k in ("ft_o25", "h1_o15", "h2_o15", "ft_btts"):
        p = P[k]
        out[k] = _bin(np.maximum(p, 1 - p), (p >= 0.5) == y[k], BINS_2)
    return out


def _bin(stat, hit, bins) -> pd.DataFrame:
    b = pd.cut(stat, bins, right=False)
    t = (pd.DataFrame({"b": b, "p": stat, "hit": hit}).groupby("b", observed=True)
         .agg(n=("p", "size"), mean_p=("p", "mean"), hit_rate=("hit", "mean")).reset_index())
    t["lo"] = t["b"].map(lambda i: i.left).astype(float)
    t["hi"] = t["b"].map(lambda i: i.right).astype(float)
    return t.drop(columns="b")
