"""Prediction records: save what the model said BEFORE kickoff, settle it against results later.

Rules that keep the record honest
  * Only matches that have not kicked off are saved; nothing is saved after the fact.
  * One snapshot per match per UTC day (a later snapshot the same day replaces the earlier one);
    older days' snapshots are never touched. The LATEST pre-kickoff snapshot is the one scored.
  * Each row stores the model settings and the date the training data runs through.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT

LOG_PATH = ROOT / "data" / "records" / "predictions.csv"
FIXTURE_KEY = ["league", "home", "away", "kickoff_utc"]
PROB_COLS = ["ft_H", "ft_D", "ft_A", "h1_H", "h1_D", "h1_A", "h2_H", "h2_D", "h2_A",
             "ft_o25", "ft_btts", "h1_o05", "h1_o15", "h2_o15"]
LOG_COLS = (["snapshot_utc", "snapshot_day", "league", "home", "away", "kickoff_utc", "model_asof", "half_life",
             "ridge", "data_through", "l1h", "l1a", "l2h", "l2a"] + PROB_COLS + ["ft_lean", "ft_tier"])


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(path)
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[LOG_COLS]


def log_predictions(rows: pd.DataFrame, path: Path | None = None, now: datetime | None = None) -> int:
    """Append/replace today's snapshots for matches that have not kicked off. Returns rows written."""
    path = Path(path) if path else LOG_PATH
    now = now or datetime.now(timezone.utc)
    if rows.empty:
        return 0
    new = rows.copy()
    ko = pd.to_datetime(new["kickoff_utc"], utc=True)
    new = new[ko > pd.Timestamp(now)].copy()
    if new.empty:
        return 0
    new["kickoff_utc"] = pd.to_datetime(new["kickoff_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    new["snapshot_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    new["snapshot_day"] = now.strftime("%Y-%m-%d")
    new = new[LOG_COLS]
    old = _read(path)
    key = FIXTURE_KEY + ["snapshot_day"]
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(key, keep="last")
    merged = merged.sort_values(["kickoff_utc", "league", "home", "snapshot_utc"]).reset_index(drop=True)
    # skip the write when nothing material changed (the app reruns on every click)
    prev = old.sort_values(["kickoff_utc", "league", "home", "snapshot_utc"]).reset_index(drop=True)
    cmp_cols = ["l1h", "l1a", "l2h", "l2a"] + PROB_COLS
    if len(prev) == len(merged) and prev[key].astype(str).equals(merged[key].astype(str)) and \
            np.allclose(prev[cmp_cols].astype(float).round(4), merged[cmp_cols].astype(float).round(4), equal_nan=True):
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def load_log(path: Path | None = None) -> pd.DataFrame:
    return _read(Path(path) if path else LOG_PATH)


# ------------------------------------------------------------------ settlement
def _idx(h, a):
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def _ll(p, hit):
    p = np.clip(np.where(hit, p, 1 - p), 1e-12, 1)
    return -np.log(p)


def _join(log: pd.DataFrame, matches: pd.DataFrame, max_days: int = 2) -> pd.DataFrame:
    """Latest pre-kickoff snapshot of each fixture joined to its played result (empty if none)."""
    if log.empty:
        return pd.DataFrame()
    last = (log.sort_values("snapshot_utc").groupby(FIXTURE_KEY, as_index=False).tail(1)).copy()
    last["ko"] = pd.to_datetime(last.kickoff_utc, utc=True)
    last["ko_date"] = last.ko.dt.tz_localize(None).dt.normalize()
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    j = last.merge(m, on=["league", "home", "away"], how="inner", suffixes=("", "_m"))
    j["gap"] = (j["date"] - j["ko_date"]).abs().dt.days
    return j[j.gap <= max_days].sort_values("gap").drop_duplicates(FIXTURE_KEY)


def settle(log: pd.DataFrame, matches: pd.DataFrame, max_days: int = 2) -> pd.DataFrame:
    """Match the latest pre-kickoff snapshot of each fixture to its played result."""
    j = _join(log, matches, max_days)
    if j.empty:
        return pd.DataFrame()
    ft, h1, h2 = _idx(j.ft_home, j.ft_away), _idx(j.ht_home, j.ht_away), _idx(j.h2_home, j.h2_away)
    out = j[["league", "home", "away", "kickoff_utc", "snapshot_utc", "ft_lean", "ft_tier", "ft_home", "ft_away",
             "ht_home", "ht_away"]].copy()
    for name, p3, act in [("ft", j[["ft_H", "ft_D", "ft_A"]].values, ft), ("h1", j[["h1_H", "h1_D", "h1_A"]].values, h1),
                          ("h2", j[["h2_H", "h2_D", "h2_A"]].values, h2)]:
        out[f"{name}_actual"] = np.array(["H", "D", "A"])[act]
        out[f"{name}_lean_hit"] = p3.argmax(1) == act
        out[f"{name}_lean_p"] = p3.max(1)
        out[f"{name}_ll"] = -np.log(np.clip(p3[np.arange(len(act)), act], 1e-12, 1))
    for name, col, actual in [("ft_o25", "ft_o25", (j.ft_home + j.ft_away) > 2.5),
                              ("ft_btts", "ft_btts", (j.ft_home > 0) & (j.ft_away > 0)),
                              ("h1_o05", "h1_o05", (j.ht_home + j.ht_away) > 0.5),
                              ("h1_o15", "h1_o15", (j.ht_home + j.ht_away) > 1.5),
                              ("h2_o15", "h2_o15", (j.h2_home + j.h2_away) > 1.5)]:
        p = j[col].astype(float).values
        out[f"{name}_actual"] = actual.values
        out[f"{name}_lean_hit"] = (p >= 0.5) == actual.values
        out[f"{name}_lean_p"] = np.maximum(p, 1 - p)
        out[f"{name}_ll"] = _ll(p, actual.values)
    # bookmaker closing line (market average, margin removed) where available
    if {"AvgCH", "AvgCD", "AvgCA"} <= set(j.columns):
        inv = 1 / j[["AvgCH", "AvgCD", "AvgCA"]].astype(float).values
        P = inv / inv.sum(1, keepdims=True)
        out["ft_market_ll"] = -np.log(np.clip(P[np.arange(len(ft)), ft], 1e-12, 1))
    if {"AvgC_o25", "AvgC_u25"} <= set(j.columns):
        io, iu = 1 / j.AvgC_o25.astype(float).values, 1 / j.AvgC_u25.astype(float).values
        out["ft_o25_market_ll"] = _ll(io / (io + iu), (j.ft_home + j.ft_away).values > 2.5)
    return out.sort_values("kickoff_utc").reset_index(drop=True)


MARKET_LABELS = {"ft": "Full-time result", "h1": "1st-half result", "h2": "2nd-half result",
                 "ft_o25": "Full-time over/under 2.5", "ft_btts": "Full-time both teams score",
                 "h1_o05": "1st-half over/under 0.5", "h1_o15": "1st-half over/under 1.5",
                 "h2_o15": "2nd-half over/under 1.5"}


def summarise(settled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, label in MARKET_LABELS.items():
        rows.append({"Market": label, "Settled": len(settled),
                     "Lean hit rate": settled[f"{k}_lean_hit"].mean(),
                     "Avg stated prob of lean": settled[f"{k}_lean_p"].mean(),
                     "Avg log loss": settled[f"{k}_ll"].mean()})
    return pd.DataFrame(rows)


def versus_market(settled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_col, mkt_col, label in [("ft_ll", "ft_market_ll", "Full-time 1X2"), ("ft_o25_ll", "ft_o25_market_ll", "Full-time over/under 2.5")]:
        if mkt_col in settled and settled[mkt_col].notna().any():
            s = settled.dropna(subset=[mkt_col])
            d = s[model_col] - s[mkt_col]
            rows.append({"Market": label, "Matches": len(s), "Model log loss": s[model_col].mean(),
                         "Bookmaker log loss": s[mkt_col].mean(), "Model minus bookmaker": d.mean(),
                         "Std. error": d.std() / np.sqrt(len(d)) if len(d) > 1 else np.nan})
    return pd.DataFrame(rows)


def by_confidence(settled: pd.DataFrame) -> pd.DataFrame:
    g = settled.groupby("ft_tier").agg(Matches=("ft_lean_hit", "size"), Hit_rate=("ft_lean_hit", "mean"),
                                       Avg_stated=("ft_lean_p", "mean")).reset_index()
    order = {"Strong": 0, "Moderate": 1, "Slight": 2, "No clear lean": 3}
    return g.sort_values("ft_tier", key=lambda s: s.map(order)).rename(columns={"ft_tier": "Confidence"})


# ------------------------------------------------------------------ diagnostics
def _latest(log: pd.DataFrame) -> pd.DataFrame:
    last = log.sort_values("snapshot_utc").groupby(FIXTURE_KEY, as_index=False).tail(1).copy()
    last["ko"] = pd.to_datetime(last.kickoff_utc, utc=True)
    return last


def coverage(log: pd.DataFrame) -> pd.DataFrame:
    """Recorded matches per kickoff date. A date with no row means nothing was saved before those kickoffs."""
    if log.empty:
        return pd.DataFrame(columns=["Kickoff date", "Matches recorded"])
    last = _latest(log)
    c = last.groupby(last.ko.dt.strftime("%Y-%m-%d")).size().rename("Matches recorded").reset_index()
    return c.rename(columns={"ko": "Kickoff date"}).sort_values("Kickoff date", ascending=False).reset_index(drop=True)


def pending_report(log: pd.DataFrame, matches: pd.DataFrame, now=None, max_days: int = 2) -> pd.DataFrame:
    """Recorded matches that have kicked off but are not scored, with the reason."""
    cols = ["Kickoff (UTC)", "League", "Match", "Reason"]
    if log.empty:
        return pd.DataFrame(columns=cols)
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    last = _latest(log)
    past = last[last.ko <= now]
    settled = settle(log, matches)
    done = set(zip(settled.league, settled.home, settled.away, settled.kickoff_utc)) if not settled.empty else set()
    m = matches.copy()
    m["date"] = pd.to_datetime(m["date"])
    latest = m["date"].max()
    rows = []
    for r in past.itertuples():
        if (r.league, r.home, r.away, r.kickoff_utc) in done:
            continue
        ko_date = r.ko.tz_localize(None).normalize()
        pair = m[(m.league == r.league) & (m.home == r.home) & (m.away == r.away) & (m.date >= ko_date - pd.Timedelta(days=30))]
        reason = (f"No result in your data yet (latest result in data: {latest:%d %b}). "
                  "football-data.co.uk updates in batches; use Refresh results." if pair.empty else
                  f"A result exists but its date is more than {max_days} days from the recorded kickoff.")
        rows.append({"Kickoff (UTC)": r.kickoff_utc, "League": r.league, "Match": f"{r.home} v {r.away}", "Reason": reason})
    return pd.DataFrame(rows, columns=cols).sort_values("Kickoff (UTC)", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ result record (results next to probabilities)
RESULT_PATH = ROOT / "data" / "records" / "results_record.csv"
_NAMES = np.array(["Home", "Draw", "Away"])
BINARY_MARKETS = [("ft_o25", "Over 2.5 goals", lambda j: (j.ft_home + j.ft_away) > 2.5),
                  ("ft_btts", "Both teams score", lambda j: (j.ft_home > 0) & (j.ft_away > 0)),
                  ("h1_o05", "1H over 0.5", lambda j: (j.ht_home + j.ht_away) > 0.5),
                  ("h1_o15", "1H over 1.5", lambda j: (j.ht_home + j.ht_away) > 1.5),
                  ("h2_o15", "2H over 1.5", lambda j: (j.h2_home + j.h2_away) > 1.5)]


def result_record(log: pd.DataFrame, matches: pd.DataFrame, max_days: int = 2) -> pd.DataFrame:
    """One row per scored match: actual scores beside the probabilities the model gave BEFORE kickoff.

    Probabilities are in percent. '*_p_actual' is the probability the model gave to what actually happened."""
    j = _join(log, matches, max_days)
    if j.empty:
        return pd.DataFrame()
    score = lambda a, b: a.astype(int).astype(str) + "-" + b.astype(int).astype(str)
    r = {"kickoff_utc": j.kickoff_utc.values, "league": j.league.values, "home": j.home.values, "away": j.away.values,
         "ht_score": score(j.ht_home, j.ht_away).values, "h2_score": score(j.h2_home, j.h2_away).values,
         "ft_score": score(j.ft_home, j.ft_away).values}
    acts = {"ft": _idx(j.ft_home, j.ft_away), "h1": _idx(j.ht_home, j.ht_away), "h2": _idx(j.h2_home, j.h2_away)}
    for k, act in acts.items():
        p = j[[f"{k}_H", f"{k}_D", f"{k}_A"]].astype(float).values
        for i, nm in enumerate(["home", "draw", "away"]):
            r[f"{k}_p_{nm}"] = 100 * p[:, i]
        r[f"{k}_actual"] = _NAMES[act]
        r[f"{k}_lean"] = _NAMES[p.argmax(1)]
        r[f"{k}_p_actual"] = 100 * p[np.arange(len(act)), act]
        r[f"{k}_hit"] = p.argmax(1) == act
    r["xg_1h"], r["goals_1h"] = (j.l1h + j.l1a).astype(float).values, (j.ht_home + j.ht_away).values
    r["xg_2h"], r["goals_2h"] = (j.l2h + j.l2a).astype(float).values, (j.h2_home + j.h2_away).values
    r["xg_ft"], r["goals_ft"] = r["xg_1h"] + r["xg_2h"], j.ft_home.values + j.ft_away.values
    for key, _label, fn in BINARY_MARKETS:
        p, happened = j[key].astype(float).values, fn(j).values
        r[f"{key}_p"], r[f"{key}_happened"] = 100 * p, happened
        r[f"{key}_p_actual"] = 100 * np.where(happened, p, 1 - p)
    r["snapshot_utc"], r["data_through"] = j.snapshot_utc.values, j.data_through.values
    return pd.DataFrame(r).sort_values("kickoff_utc", ascending=False).reset_index(drop=True)


def save_result_record(log: pd.DataFrame, matches: pd.DataFrame, path: Path | None = None) -> int:
    """Write the result record to CSV (atomically, and only if it changed). Returns rows in the record."""
    path = Path(path) if path else RESULT_PATH
    rec = result_record(log, matches)
    if rec.empty:
        return 0
    text = rec.round(3).to_csv(index=False)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return len(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return len(rec)
