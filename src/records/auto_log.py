"""Headless prediction logging: the scheduled task records predictions for upcoming matches even when
the app is closed. Shares the exact prediction code path with the app (league_predictions)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REPORT_DIR
from src.markets.grid_markets import market_probs
from src.markets.leans import tier
from src.models.predictor import fit_league_models, predict_lambdas
from src.records.prediction_log import log_predictions


def default_params() -> tuple[float, float]:
    p = REPORT_DIR / "best_params.json"
    if p.exists():
        d = json.loads(p.read_text())
        return float(d["half_life"]), float(d["ridge"])
    return 365.0, 10.0


def default_asof(df: pd.DataFrame, now: datetime | None = None) -> pd.Timestamp:
    today = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC").tz_localize(None).normalize()
    return max((df.date.max() + pd.Timedelta(days=1)).normalize(), today)


def league_predictions(m1, m2, g: pd.DataFrame, league: str, asof_ts: pd.Timestamp, half_life: float,
                       ridge: float, data_through: str):
    """Predictions for one league's fixtures `g` (columns home, away, kickoff_utc).
    Returns (expected goals array (N,4), market probabilities dict, rows for the prediction log)."""
    lam = np.array([predict_lambdas(m1, m2, h, a) for h, a in zip(g.home, g.away)])
    P = market_probs(lam)
    ft = P["ft_1x2"]
    log_df = pd.DataFrame({
        "league": league, "home": g.home.values, "away": g.away.values, "kickoff_utc": g.kickoff_utc.values,
        "model_asof": f"{asof_ts:%Y-%m-%d}", "half_life": half_life, "ridge": ridge, "data_through": data_through,
        "l1h": lam[:, 0], "l1a": lam[:, 1], "l2h": lam[:, 2], "l2a": lam[:, 3],
        "ft_H": ft[:, 0], "ft_D": ft[:, 1], "ft_A": ft[:, 2],
        "h1_H": P["h1_1x2"][:, 0], "h1_D": P["h1_1x2"][:, 1], "h1_A": P["h1_1x2"][:, 2],
        "h2_H": P["h2_1x2"][:, 0], "h2_D": P["h2_1x2"][:, 1], "h2_A": P["h2_1x2"][:, 2],
        "ft_o25": P["ft_o25"], "ft_btts": P["ft_btts"], "h1_o05": P["h1_o05"], "h1_o15": P["h1_o15"],
        "h2_o15": P["h2_o15"], "ft_lean": np.array(["H", "D", "A"])[ft.argmax(1)],
        "ft_tier": [tier(v) for v in ft.max(1)]})
    return lam, P, log_df


def log_upcoming(df: pd.DataFrame, fixtures: pd.DataFrame, now: datetime | None = None, days: int = 7,
                 path: Path | None = None) -> dict:
    """Record default-settings predictions for fixtures kicking off within `days`. Returns a small report."""
    now = now or datetime.now(timezone.utc)
    hl, rg = default_params()
    asof = default_asof(df, now)
    ts = pd.Timestamp(now)
    fx = fixtures[(fixtures.kickoff_utc > ts) & (fixtures.kickoff_utc <= ts + pd.Timedelta(days=days))]
    rows, skipped = [], []
    for league, g in fx.groupby("league"):
        try:
            m1, m2, _ = fit_league_models(df, league, asof, hl, rg)
        except ValueError as exc:
            skipped.append(f"{league}: {exc}")
            continue
        rows.append(league_predictions(m1, m2, g, league, asof, hl, rg, f"{df.date.max():%Y-%m-%d}")[2])
    written = log_predictions(pd.concat(rows, ignore_index=True), path=path, now=now) if rows else 0
    return {"fixtures_in_window": len(fx), "rows_written": written, "skipped": skipped}
