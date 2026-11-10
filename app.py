"""Football 1H / 2H prediction system - graphical interface.

Run from the project folder:
    python -m streamlit run app.py
A browser tab opens at http://localhost:8501
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from src.config import PROCESSED_DIR, REPORT_DIR, ROOT
from src.evaluation.calibration import calibration_tables
from src.ingest.collector_status import collector_health, task_status
from src.ingest.fixtures import get_fixtures
from src.ingest.odds_api import OddsClient, load_key
from src.ingest.auto_refresh import auto_refresh, last_attempt, mark_attempt
from src.ingest.odds_api import find_secret
from src.ingest.update_current import refresh_all, summarize
from src.markets.grid_markets import market_probs, score_grid
from src.markets.leans import NONE, headline, lean_table, tier
from src.models.explain import reasons
from src.slips.builder import GROUPS, build_legs, select_slip, slip_stats
from src.slips.log import load_slips, save_slip, settle_slips
from src.models.predictor import fit_league_models, is_known, predict_lambdas
from src.analysis.h2h import h2h_meetings, h2h_summary, league_timing, timing_profile
from src.records.auto_log import league_predictions
from src.records.prediction_log import (BINARY_MARKETS, FIXTURE_KEY, coverage, pending_report, result_record, by_confidence, load_log, log_predictions, settle, summarise,
                                         versus_market)

st.set_page_config(page_title="Football halves model", layout="wide")
TZ_OPTIONS = ["Africa/Dar_es_Salaam", "UTC", "Europe/London", "Europe/Paris", "America/New_York"]
TIER_CSS = {"Strong": "background-color:#0b6b2f;color:#ffffff;font-weight:700",
            "Moderate": "background-color:#7cc99a;color:#06260f;font-weight:700",
            "Slight": "background-color:#d9f2e0;color:#0b3d1d;font-weight:700"}
LEGEND = ("Green = the outcome the model rates most likely (darker = stronger: Strong 60%+ for 3-way results, "
          "65%+ for two-way markets). **Most likely is not the same as good value.**")


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_matches(mtime: float) -> pd.DataFrame | None:
    """mtime is part of the cache key, so a background refresh of the file is picked up automatically."""
    p = PROCESSED_DIR / "matches_all.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["season"] = df["season"].astype(str)
    return df


@st.cache_data
def get_calibration(_df: pd.DataFrame) -> dict:
    return calibration_tables(_df)


def default_params() -> tuple[float, float]:
    p = REPORT_DIR / "best_params.json"
    if p.exists():
        d = json.loads(p.read_text())
        return float(d["half_life"]), float(d["ridge"])
    return 365.0, 10.0


def stamp_of(df: pd.DataFrame) -> str:
    return f"{len(df)}|{df.date.max():%Y-%m-%d}"


@st.cache_resource
def _models(_df: pd.DataFrame, league: str, asof: str, half_life: float, ridge: float, stamp: str):
    return fit_league_models(_df, league, pd.Timestamp(asof), half_life, ridge)


def get_models(df: pd.DataFrame, league: str, asof: str, half_life: float, ridge: float):
    return _models(df, league, asof, half_life, ridge, stamp_of(df))


def current_teams(df: pd.DataFrame) -> dict[str, set[str]]:
    out = {}
    for lg, g in df.groupby("league"):
        cur = g[g.season == g.season.max()]
        out[lg] = set(cur.home) | set(cur.away)
    return out


@st.cache_data(ttl=1800, show_spinner="Loading upcoming fixtures...")
def load_fixtures(nonce: int, force: bool, _df: pd.DataFrame):
    try:
        client = OddsClient(load_key())
    except SystemExit:
        client = None
    return get_fixtures(force=force, odds_client=client, known_teams=current_teams(_df))


def fair(x: float) -> float:
    return round(1 / x, 2) if x > 1e-6 else float("nan")


def style_rows(df: pd.DataFrame, tiers: list[str], fmt: dict | None = None):
    """Colour whole rows by confidence tier (bold + green)."""
    df = df.reset_index(drop=True)

    def f(row):
        return [TIER_CSS.get(tiers[row.name], "")] * len(row)
    sty = df.style.apply(f, axis=1)
    return sty.format(fmt) if fmt else sty


FMT = {"Probability": "{:.1%}", "Model fair odds": "{:.2f}", "Fair odds": "{:.2f}"}


def one_x_two(p: np.ndarray, home: str, away: str):
    df = pd.DataFrame({"Outcome": [home, "Draw", away], "Probability": p, "Model fair odds": [fair(v) for v in p]})
    i = int(np.argmax(p))
    return style_rows(df, [tier(p[k], "3way") if k == i else NONE for k in range(3)], FMT)


def over_rows(label: str, mu: float, lines: list[float]):
    rows = []
    for ln in lines:
        po = float(1 - poisson.cdf(int(ln), mu))
        rows += [(f"{label} over {ln}", po), (f"{label} under {ln}", 1 - po)]
    return rows


def two_way_table(rows):
    df = pd.DataFrame({"Market": [r[0] for r in rows], "Probability": [r[1] for r in rows],
                       "Fair odds": [fair(r[1]) for r in rows]})
    return style_rows(df, [tier(r[1], "binary") for r in rows], FMT)


def top_scores(grid: np.ndarray, n: int = 8):
    flat = [(f"{i}-{j}", grid[i, j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    flat.sort(key=lambda t: -t[1])
    df = pd.DataFrame({"Score": [s for s, _ in flat[:n]], "Probability": [p for _, p in flat[:n]],
                       "Fair odds": [fair(p) for _, p in flat[:n]]})
    return df.style.apply(lambda r: ["font-weight:700" if r.name == 0 else ""] * len(r), axis=1).format(FMT)


def local_labels(fx: pd.DataFrame, tz: str) -> pd.DataFrame:
    fx = fx.copy()
    try:
        loc = fx.kickoff_utc.dt.tz_convert(tz)
    except Exception:  # noqa: BLE001  (time-zone database missing on Windows: pip install tzdata)
        st.warning(f"Time zone {tz} unavailable (run: python -m pip install tzdata). Showing UTC.")
        loc = fx.kickoff_utc.dt.tz_convert("UTC")
    fx["when"] = np.where(fx.time_known, loc.dt.strftime("%a %d %b %H:%M"), loc.dt.strftime("%a %d %b") + " (time TBC)")
    return fx


TRIPLES = [("FT 1", "FT X", "FT 2"), ("1H 1", "1H X", "1H 2"), ("2H 1", "2H X", "2H 2")]


def slate_style(row):
    css = pd.Series("", index=row.index)
    for trio in TRIPLES:
        vals = row[list(trio)].astype(float).values / 100
        i = int(vals.argmax())
        css[trio[i]] = TIER_CSS.get(tier(vals[i]), "")
    return css


def build_slate(df, fx, asof_ts, half_life, ridge):
    """Returns (display table, rows for the prediction log)."""
    parts, logs = [], []
    data_through = f"{df.date.max():%Y-%m-%d}"
    for league, g in fx.groupby("league"):
        try:
            m1, m2, _ = get_models(df, league, str(asof_ts.date()), float(half_life), float(ridge))
        except ValueError:
            continue
        lam, P, log_df = league_predictions(m1, m2, g, league, asof_ts, half_life, ridge, data_through)
        ft = P["ft_1x2"]
        notes = [", ".join(f"{t} (no history)" for t in (h, a) if not is_known(m1, t)) for h, a in zip(g.home, g.away)]
        parts.append(pd.DataFrame({
            "_ko": g.kickoff_utc.values, "Kickoff": g.when.values, "League": league,
            "Home": g.home.values, "Away": g.away.values,
            "FT lean": np.array(["Home", "Draw", "Away"])[ft.argmax(1)], "Lean strength": log_df.ft_tier.values,
            "xG 1H": (lam[:, 0] + lam[:, 1]).round(2), "xG 2H": (lam[:, 2] + lam[:, 3]).round(2),
            "FT 1": 100 * ft[:, 0], "FT X": 100 * ft[:, 1], "FT 2": 100 * ft[:, 2],
            "1H 1": 100 * P["h1_1x2"][:, 0], "1H X": 100 * P["h1_1x2"][:, 1], "1H 2": 100 * P["h1_1x2"][:, 2],
            "2H 1": 100 * P["h2_1x2"][:, 0], "2H X": 100 * P["h2_1x2"][:, 1], "2H 2": 100 * P["h2_1x2"][:, 2],
            "FT O2.5": 100 * P["ft_o25"], "FT BTTS": 100 * P["ft_btts"],
            "1H O0.5": 100 * P["h1_o05"], "1H O1.5": 100 * P["h1_o15"], "2H O1.5": 100 * P["h2_o15"],
            "Note": notes}))
        logs.append(log_df)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    out = pd.concat(parts).sort_values("_ko").drop(columns="_ko").reset_index(drop=True)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(1)
    return out, pd.concat(logs, ignore_index=True)


def render_match(df, league, home, away, asof_ts, half_life, ridge):
    try:
        m1, m2, tr = get_models(df, league, str(asof_ts.date()), float(half_life), float(ridge))
    except ValueError as exc:
        st.error(str(exc))
        return
    for t in (home, away):
        if not is_known(m1, t):
            st.warning(f"{t} has no matches in the training window, so it is treated as an average team "
                       "(this usually overrates newly promoted clubs).")
    lam = predict_lambdas(m1, m2, home, away)
    P = market_probs(lam[None, :])
    l1h, l1a, l2h, l2a = lam
    tot1, tot2 = l1h + l1a, l2h + l2a

    st.subheader(f"{home} v {away}")

    # ---- short summary
    leans = lean_table(P, lam, home, away)
    head = headline(leans)
    (st.info if head.startswith("No strong") else st.success)(f"Model view: {head}")
    st.dataframe(style_rows(leans[["Market", "Model lean", "Probability", "Confidence", "Fair odds"]],
                            leans.Confidence.tolist(), {"Probability": "{:.0%}", "Fair odds": "{:.2f}"}),
                 hide_index=True, width="stretch")
    for line in reasons(m1, m2, tr, home, away, lam, leans, get_calibration(df)):
        st.markdown("- " + line)
    st.caption(LEGEND + " In backtests the model trailed bookmaker closing prices on full-time markets, so compare "
               "with a real price (box at the bottom) before treating any lean as a bet.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Expected goals, 1st half", f"{l1h:.2f} - {l1a:.2f}", f"total {tot1:.2f}", delta_color="off")
    k2.metric("Expected goals, 2nd half", f"{l2h:.2f} - {l2a:.2f}", f"total {tot2:.2f}", delta_color="off")
    k3.metric("Expected goals, full time", f"{l1h + l2h:.2f} - {l1a + l2a:.2f}", f"total {tot1 + tot2:.2f}", delta_color="off")

    a, b, c = st.columns(3)
    a.markdown("**First-half result**")
    a.dataframe(one_x_two(P["h1_1x2"][0], home, away), hide_index=True, width="stretch")
    b.markdown("**Second-half result** (second half in isolation)")
    b.dataframe(one_x_two(P["h2_1x2"][0], home, away), hide_index=True, width="stretch")
    c.markdown("**Full-time result**")
    c.dataframe(one_x_two(P["ft_1x2"][0], home, away), hide_index=True, width="stretch")

    st.markdown("**Goals and both-teams-to-score**")
    btts = lambda k, lab: [(f"{lab} BTTS yes", float(P[k][0])), (f"{lab} BTTS no", 1 - float(P[k][0]))]
    rows1 = over_rows("1H", tot1, [0.5, 1.5, 2.5]) + btts("h1_btts", "1H")
    rows2 = over_rows("2H", tot2, [0.5, 1.5, 2.5]) + btts("h2_btts", "2H")
    rows3 = over_rows("FT", tot1 + tot2, [1.5, 2.5, 3.5]) + btts("ft_btts", "FT")
    d1, d2, d3 = st.columns(3)
    d1.dataframe(two_way_table(rows1), hide_index=True, width="stretch")
    d2.dataframe(two_way_table(rows2), hide_index=True, width="stretch")
    d3.dataframe(two_way_table(rows3), hide_index=True, width="stretch")

    st.markdown("**Half-time / full-time**")
    labels = [home, "Draw", away]
    grid = P["htft"][0].reshape(3, 3)
    htft = pd.DataFrame(grid, index=[f"HT: {x}" for x in labels], columns=[f"FT: {x}" for x in labels])
    best = np.unravel_index(grid.argmax(), grid.shape)
    st.dataframe(htft.style.format("{:.1%}").apply(
        lambda d: pd.DataFrame([[("font-weight:700;background-color:#7cc99a;color:#06260f" if (i, j) == best else "")
                                 for j in range(3)] for i in range(3)], index=d.index, columns=d.columns), axis=None),
        width="stretch")

    st.markdown("**Most likely exact scores**")
    e1, e2, e3 = st.columns(3)
    e1.caption("First half")
    e1.dataframe(top_scores(score_grid(l1h, l1a)), hide_index=True, width="stretch")
    e2.caption("Second half")
    e2.dataframe(top_scores(score_grid(l2h, l2a)), hide_index=True, width="stretch")
    e3.caption("Full time")
    e3.dataframe(top_scores(score_grid(l1h + l2h, l1a + l2a, k=10)), hide_index=True, width="stretch")

    # ---- head-to-head and when the goals happen
    st.markdown("**Head-to-head and when the goals happen**")
    meet = h2h_meetings(df, league, home, away, asof_ts, n=10)
    hs = h2h_summary(meet)
    if hs["n"] == 0:
        st.caption("No earlier league meetings of these two teams in the data (since 2015/16), for example a newly promoted club.")
    else:
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("League meetings (latest)", hs["n"])
        q2.metric(f"{home} W-D-L", f"{hs['wins']}-{hs['draws']}-{hs['losses']}")
        q3.metric("Average goals", f"{hs['avg_goals']:.1f}")
        q4.metric("Both teams scored", f"{hs['btts_pct']:.0f}%")
        q5.metric("Goals after half-time", f"{hs['share_2h']:.0f}%")
        res_css = {"W": "background-color:#7cc99a;color:#06260f;font-weight:700", "D": "background-color:#d9d9d9;color:#222222;font-weight:700",
                   "L": "background-color:#e57373;color:#3d0000;font-weight:700"}
        st.dataframe(meet.style.map(lambda v: res_css.get(v, ""), subset=["Result"]), hide_index=True, width="stretch")
        st.caption(f"Result is from {home}'s point of view, whichever team was at home. HT = half-time score, 2H = score in the second half alone.")
    n_t = 20
    rows = []
    for team in (home, away):
        pf = timing_profile(df, league, team, asof_ts, n=n_t)
        if pf["n"]:
            rows.append({"Team": f"{team} (last {pf['n']} league matches)", "Scored 1H": pf["scored_1h"], "Scored 2H": pf["scored_2h"],
                         "% of scored in 2H": pf["scored_2h_share"], "Conceded 1H": pf["conceded_1h"], "Conceded 2H": pf["conceded_2h"],
                         "% of conceded in 2H": pf["conceded_2h_share"], "Matches scoring in 1H %": pf["scored_in_1h_pct"],
                         "Matches scoring in 2H %": pf["scored_in_2h_pct"]})
    lt = league_timing(df, league, asof_ts)
    rows.append({"Team": f"League average (last 3 seasons)", "% of scored in 2H": lt["share_2h"], "% of conceded in 2H": lt["share_2h"]})
    tim = pd.DataFrame(rows)
    pcts = [c for c in tim.columns if c.endswith("%") or "in 2H" in c]
    st.dataframe(tim.style.format({c: "{:.0f}" for c in pcts}, na_rep="-"), hide_index=True, width="stretch")
    st.caption("Goal timing is measured per half (first half vs second half) from half-time and full-time scores; goal minutes "
               "are not available for most of the history. Small samples: treat as context. In tests, head-to-head history "
               "added only a small amount of predictive power beyond team strength (about 0.003 in log loss), so it is shown "
               "for context and is NOT used in the probabilities above.")

    # ---- price check
    with st.expander("Check a bookmaker price"):
        outs = {}
        for nm, key in [("Full-time", "ft_1x2"), ("1st-half", "h1_1x2"), ("2nd-half", "h2_1x2")]:
            for lab, p in zip([f"{home} win", "Draw", f"{away} win"], P[key][0]):
                outs[f"{nm} result: {lab}"] = float(p)
        for lab, p in rows1 + rows2 + rows3:
            outs[lab] = float(p)
        outs["First 5 minutes 1X2: Draw (experimental, rough estimate from 258 matches)"] = float(np.exp(-0.035 * (tot1 + tot2)))
        sel = st.selectbox("Outcome", list(outs), key="oc_market")
        odds = st.number_input("Decimal odds on offer", min_value=0.0, max_value=1000.0, value=0.0, step=0.01, key="oc_odds")
        p = outs[sel]
        if odds < 1.01:
            st.caption(f"Model probability {p:.1%}, fair odds {fair(p):.2f}. Enter the price you see to compare.")
        else:
            ev = p * odds - 1
            m1c, m2c, m3c = st.columns(3)
            m1c.metric("Break-even probability", f"{1 / odds:.1%}")
            m2c.metric("Model probability", f"{p:.1%}", f"{100 * (p - 1 / odds):+.1f} pts", delta_color="off")
            m3c.metric("Expected value per 1 staked", f"{ev:+.1%}")
            if ev > 0:
                st.warning("Positive on paper only. The model has trailed closing prices in backtests, so treat this as "
                           "something to investigate, not a signal. Small gaps are well inside the model's error.")
            else:
                st.info("The price is not better than the model's fair price.")

    st.caption(f"Fitted on {len(tr):,} {league} matches before {asof_ts:%d %b %Y}. Assumes goals in the two halves "
               "are independent given team strengths. In 2026/27 so far, second-half goals have run about 12% above "
               "the model's forecast.")

OUT_CSS = "background-color:#bde0fe;color:#06223d;font-weight:700"     # the outcome that actually happened


def gave_css(v: float) -> str:
    """Colour for 'probability the model gave to what happened': green = expected, red = surprise."""
    if v >= 50:
        return "background-color:#0b6b2f;color:#ffffff;font-weight:700"
    if v >= 33:
        return "background-color:#7cc99a;color:#06260f;font-weight:700"
    if v >= 20:
        return "background-color:#f6c177;color:#3d2400;font-weight:700"
    return "background-color:#e57373;color:#3d0000;font-weight:700"


def render_result_record(rr: pd.DataFrame) -> None:
    """Actual results beside the probabilities the model gave before kickoff."""
    st.markdown("### Results vs model")
    st.caption("Each row shows what the model gave every outcome BEFORE kickoff (%) and what happened. "
               "Blue = the outcome that happened. 'Gave %' is the probability the model had put on it: "
               "green = the model expected it, red = a surprise to the model.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Matches scored", len(rr))
    k2.metric("Full-time lean correct", f"{100 * rr.ft_hit.mean():.0f}%")
    k3.metric("Avg probability given to the full-time result", f"{rr.ft_p_actual.mean():.0f}%")
    st.markdown("**Goals per match: model vs reality (all scored matches)**")
    st.dataframe(pd.DataFrame({"": ["1st half", "2nd half", "Full time"],
                               "Model expected": [rr.xg_1h.mean(), rr.xg_2h.mean(), rr.xg_ft.mean()],
                               "Actual": [rr.goals_1h.mean(), rr.goals_2h.mean(), rr.goals_ft.mean()]})
                 .style.format({"Model expected": "{:.2f}", "Actual": "{:.2f}"}), hide_index=True, width="stretch")
    leagues = sorted(rr.league.unique())
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    chosen = f1.multiselect("Leagues", leagues, default=leagues, key="rec_leagues")
    misses = f2.checkbox("Only lean misses", key="rec_misses")
    halves = f3.checkbox("Show halves", key="rec_halves")
    goals = f4.checkbox("Show goals markets", key="rec_goals")
    n_show = st.slider("Rows to show (most recent)", 5, 200, 30, 5, key="rec_n")
    v = rr[rr.league.isin(chosen)]
    if misses:
        v = v[~v.ft_hit]
    v = v.head(n_show).reset_index(drop=True)
    if v.empty:
        st.info("No matches match these filters.")
        return
    cols = {"Kickoff (UTC)": pd.to_datetime(v.kickoff_utc, utc=True).dt.strftime("%d %b %H:%M"), "League": v.league,
            "Match": v.home + " v " + v.away, "HT": v.ht_score, "2H": v.h2_score, "FT": v.ft_score}
    groups = [("FT", "ft")] + ([("1H", "h1"), ("2H", "h2")] if halves else [])
    for label, k in groups:
        cols[f"{label} Home %"], cols[f"{label} Draw %"], cols[f"{label} Away %"] = v[f"{k}_p_home"], v[f"{k}_p_draw"], v[f"{k}_p_away"]
        cols[f"{label} result"], cols[f"{label} gave %"] = v[f"{k}_actual"], v[f"{k}_p_actual"]
        if label == "FT":
            cols["Lean"] = np.where(v.ft_hit, "✓", "✗")
    if goals:
        cols["Goals 1H exp/act"] = [f"{a:.1f} / {b}" for a, b in zip(v.xg_1h, v.goals_1h)]
        cols["Goals 2H exp/act"] = [f"{a:.1f} / {b}" for a, b in zip(v.xg_2h, v.goals_2h)]
        cols["Goals FT exp/act"] = [f"{a:.1f} / {b}" for a, b in zip(v.xg_ft, v.goals_ft)]
        cols["Over 2.5 %"], cols["Over 2.5?"] = v.ft_o25_p, np.where(v.ft_o25_happened, "yes", "no")
        cols["BTTS %"], cols["BTTS?"] = v.ft_btts_p, np.where(v.ft_btts_happened, "yes", "no")
    table = pd.DataFrame(cols)
    prefixes = [g[0] for g in groups]

    def row_css(row):
        css = pd.Series("", index=row.index)
        for pre in prefixes:
            css[f"{pre} {row[f'{pre} result']} %"] = OUT_CSS
            css[f"{pre} gave %"] = gave_css(row[f"{pre} gave %"])
        return css
    pct_cols = [c for c in table.columns if c.endswith("%") and not c.endswith("result")]
    st.dataframe(table.style.apply(row_css, axis=1).format({c: "{:.0f}" for c in pct_cols}),
                 hide_index=True, width="stretch", height=min(700, 40 + 35 * len(table)))

    st.markdown("**Match detail**")
    labels = [f"{a} | {b} v {c} ({d})" for a, b, c, d in zip(table["Kickoff (UTC)"], v.home, v.away, v.ft_score)]
    pick = st.selectbox("Match", labels, key="rec_pick")
    r = v.iloc[labels.index(pick)]
    rows = []
    for key, name in [("ft", "Full-time result"), ("h1", "1st-half result"), ("h2", "2nd-half result")]:
        for outcome, col in [("Home", "home"), ("Draw", "draw"), ("Away", "away")]:
            rows.append((name, outcome, r[f"{key}_p_{col}"], outcome == r[f"{key}_actual"]))
    for key, label, _fn in BINARY_MARKETS:
        yes, no = ("Over", "Under") if "over" in label.lower() else ("Yes", "No")
        rows += [(label, yes, r[f"{key}_p"], bool(r[f"{key}_happened"])), (label, no, 100 - r[f"{key}_p"], not bool(r[f"{key}_happened"]))]
    det = pd.DataFrame(rows, columns=["Market", "Outcome", "Model %", "Happened"])
    det["Happened"] = np.where(det["Happened"], "✓ happened", "")
    st.dataframe(det.style.apply(lambda row: [OUT_CSS if row["Happened"] else ""] * len(row), axis=1).format({"Model %": "{:.1f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"Expected vs actual goals: 1st half {r.xg_1h:.1f} vs {r.goals_1h}, 2nd half {r.xg_2h:.1f} vs {r.goals_2h}, "
               f"full time {r.xg_ft:.1f} vs {r.goals_ft}. Prediction saved {r.snapshot_utc} (data through {r.data_through}).")
    st.download_button("Download the results record (CSV)", data=rr.round(3).to_csv(index=False),
                       file_name="results_record.csv", mime="text/csv", key="dl_record")


# ---------------------------------------------------------------- app
if "auto_checked" not in st.session_state:          # once per browser session, and at most every 3 hours overall
    st.session_state.auto_checked = True
    if not os.environ.get("FOOTBALL_APP_NO_AUTO_REFRESH"):
        try:
            with st.spinner("Checking for new results..."):
                res = auto_refresh(180)
            if res is not None:
                st.session_state.flash = summarize(res)
        except Exception as exc:  # noqa: BLE001
            st.session_state.flash = [f"Automatic results check failed: {type(exc).__name__}: {str(exc)[:120]}"]

_p = PROCESSED_DIR / "matches_all.csv"
df = load_matches(_p.stat().st_mtime if _p.exists() else 0.0)
st.title("Football first-half / second-half model")
IS_CLOUD = (find_secret("FOOTBALL_APP_CLOUD", required=False)[0] or "0") not in ("0", "")
if IS_CLOUD:
    st.warning("Cloud copy. Predictions, slips and the results record saved here are TEMPORARY: they are lost when the app "
               "restarts or sleeps, and the odds collector cannot run here. Your own computer keeps the permanent track record.")
if df is None:
    st.error(f"Could not find {PROCESSED_DIR / 'matches_all.csv'}. Run `python run_build_history.py` first.")
    st.stop()

today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
hl0, rg0 = default_params()
if "nonce" not in st.session_state:
    st.session_state.nonce, st.session_state.force = 0, False
last = df.date.max()
default_asof = max((last + pd.Timedelta(days=1)).date(), today.date())

with st.sidebar:
    st.header("Fixtures")
    tz = st.selectbox("Show kickoff times in", TZ_OPTIONS, key="tz")
    st.session_state.setdefault("days", 7)           # set here, not via a widget default: the "show next matches" button changes it
    days = st.slider("Show matches in the next (days)", 1, 30, key="days")
    c_a, c_b = st.columns(2)
    if c_a.button("Reload fixtures", key="reload_fx"):
        st.session_state.nonce += 1
        st.session_state.force = True
        st.rerun()
    if c_b.button("Refresh results", key="refresh_res", help="Download the latest 2026/27 results now"):
        try:
            with st.spinner("Downloading latest results..."):
                res = refresh_all()
                mark_attempt()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.flash = summarize(res)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not refresh results: {exc}")
    if "flash" in st.session_state:
        lines = st.session_state.pop("flash")
        (st.warning if any("FAILED" in l or "failed" in l for l in lines) else st.success)("\n".join(f"- {l}" for l in lines))
    st.caption(f"Latest result in data: {last:%d %b %Y}")
    la = last_attempt()
    if la is not None:
        st.caption(f"Last automatic results check: {la:%d %b %H:%M} UTC (every 3 hours at most)")
    if (today - last).days > 3:
        st.warning(f"Results end {last:%d %b}. football-data.co.uk updates in batches, often days after matches. "
                   "Click 'Refresh results', or add a football-data.org token for faster results (README).")
    if find_secret("FOOTBALL_DATA_ORG_TOKEN", required=False)[0] is None:
        st.caption("Optional faster results: add FOOTBALL_DATA_ORG_TOKEN to .env.")

    st.header("Model settings")
    half_life = st.slider("Time-decay half-life (days)", 90, 1500, int(hl0), 15,
                          help="Smaller = recent form counts more. Tuned on 2017-2021 only.")
    ridge = st.slider("Shrinkage (ridge)", 1.0, 60.0, float(rg0), 1.0,
                      help="Higher = team strengths pulled harder toward the league average.")
    asof = st.date_input("Predict as of", value=default_asof, help="Only matches before this date are used.")
    st.caption(f"Data: {len(df):,} matches, {df.date.min():%d %b %Y} to {last:%d %b %Y}")
    st.info("Probabilities are model estimates, not betting advice. In backtests the model "
            "did NOT beat bookmaker closing prices on full-time markets.", icon="⚠️")

asof_ts = pd.Timestamp(asof)
fx_all, fx_status = load_fixtures(st.session_state.nonce, st.session_state.pop("force", False), df)
now = pd.Timestamp.now(tz="UTC")
fx = fx_all[(fx_all.kickoff_utc >= now - pd.Timedelta(hours=2)) & (fx_all.kickoff_utc <= now + pd.Timedelta(days=days))]
fx = local_labels(fx.sort_values("kickoff_utc"), tz).reset_index(drop=True) if len(fx) else fx.assign(when=[])
upcoming_all = fx_all[fx_all.kickoff_utc >= now - pd.Timedelta(hours=2)]       # everything still to be played
slate, log_rows = build_slate(df, fx, asof_ts, half_life, ridge) if len(fx) else (pd.DataFrame(), pd.DataFrame())

tab_slate, tab_pred, tab_slip, tab_rec, tab_bt, tab_data, tab_odds = st.tabs(
    ["Upcoming slate", "Match predictor", "Slip builder", "Track record", "Backtest results", "Data", "Odds collection"])


def empty_window_help(prefix: str) -> None:
    """Explain WHY nothing is shown (e.g. an international break) and offer a one-click fix."""
    if len(upcoming_all):
        nxt = upcoming_all.kickoff_utc.min()
        gap = (nxt - now).total_seconds() / 86400
        try:
            nxt_txt = nxt.tz_convert(tz).strftime("%a %d %b %H:%M")
        except Exception:  # noqa: BLE001
            nxt_txt = nxt.strftime("%a %d %b %H:%M UTC")
        need = int(np.ceil(gap)) + 2
        st.info(f"No matches in the next {days} days, but {len(upcoming_all)} later fixtures are loaded. The next kickoff is "
                f"**{nxt_txt}** (in {gap:.0f} days). A gap like this is usually an international break; the leagues pause.")
        if need <= 30:
            st.button(f"Show matches up to {need} days ahead", key=f"show_next_{prefix}",
                      on_click=lambda n=need: st.session_state.update(days=n))
        else:
            st.caption("The next fixture is more than 30 days away. Pick teams by hand in the 'Match predictor' tab.")
    else:
        past = len(fx_all)
        st.info("No upcoming fixtures are loaded" + (f" ({past} fixtures were loaded but all have already kicked off)" if past else "") +
                ". Try 'Reload fixtures' in the sidebar. You can still pick teams by hand in the 'Match predictor' tab.")

# ---------------------------------------------------------------- slate
with tab_slate:
    st.caption(fx_status)
    if fx.empty:
        empty_window_help("slate")
    else:
        leagues = sorted(slate.League.unique())
        chosen = st.multiselect("Leagues", leagues, default=leagues, key="slate_leagues")
        view = slate[slate.League.isin(chosen)].reset_index(drop=True)
        st.markdown(f"**{len(view)} matches** - probabilities in %. 1/X/2 = home/draw/away. "
                    "xG = model expected goals in that half. " + LEGEND)
        num_cols = [c for c in view.select_dtypes("number").columns]
        st.dataframe(view.style.apply(slate_style, axis=1).format({c: "{:.1f}" for c in num_cols if c.startswith(("FT", "1H", "2H"))}),
                     hide_index=True, width="stretch", height=min(650, 40 + 35 * len(view)))
        is_default = (abs(half_life - hl0) < 1e-9 and abs(ridge - rg0) < 1e-9 and asof_ts.date() == default_asof)
        if is_default:
            log_predictions(log_rows)
            n_logged = len(load_log().drop_duplicates(FIXTURE_KEY))
            st.caption(f"Saved to the track record before kickoff ({n_logged} matches recorded so far). "
                       "See the 'Track record' tab.")
        else:
            st.caption("Not saved to the track record: model settings differ from the defaults.")

# ---------------------------------------------------------------- predictor
with tab_pred:
    mode = st.radio("Choose match", ["Upcoming fixtures", "Pick teams manually"], horizontal=True,
                    index=0 if len(fx) else 1, key="mode")
    league = home = away = None
    if mode == "Upcoming fixtures":
        if fx.empty:
            st.info("No upcoming fixtures loaded - switch to 'Pick teams manually'.")
        else:
            labels = [f"{r.when}  |  {r.league}  |  {r.home} v {r.away}" for r in fx.itertuples()]
            pick = st.selectbox("Match", labels, key="match_pick")
            row = fx.iloc[labels.index(pick)]
            league, home, away = row["league"], row["home"], row["away"]
    else:
        c1, c2, c3 = st.columns(3)
        league = c1.selectbox("League", sorted(df.league.unique()), key="league")
        played = df[(df.league == league) & (df.date < asof_ts)]
        latest = played[played.season == played.season.max()]
        teams = sorted(set(latest.home) | set(latest.away))
        if len(teams) < 16:          # very early in a season: fall back to the last 12 months
            recent = played[played.date >= asof_ts - pd.Timedelta(days=365)]
            teams = sorted(set(recent.home) | set(recent.away))
        home = c2.selectbox("Home team", teams, index=0, key="home")
        away = c3.selectbox("Away team", [t for t in teams if t != home], index=0, key="away")
    if league:
        render_match(df, league, home, away, asof_ts, half_life, ridge)

# ---------------------------------------------------------------- slip builder
with tab_slip:
    st.markdown("Build a slip from the model's probabilities. **A slip only wins if every leg wins**, so the chances "
                "multiply: ten 70% legs win together about 3% of the time. Legs from different matches are treated as "
                "independent. This is a way to look at the numbers, not a recommendation: in backtests the model trailed "
                "bookmaker closing prices on full-time markets.")
    if log_rows.empty:
        empty_window_help("slip")
    else:
        legs_all = build_legs(log_rows)
        c1, c2 = st.columns(2)
        sel_leagues = c1.multiselect("Leagues", sorted(legs_all.league.unique()), default=sorted(legs_all.league.unique()), key="sb_leagues")
        sel_groups = c2.multiselect("Markets", GROUPS, default=["Full-time result", "Double chance", "Full-time goals", "Both teams to score"], key="sb_groups")
        d1, d2 = st.columns(2)
        lo, hi = d1.slider("Model probability of each leg (%)", 30, 99, (65, 90), key="sb_range")
        n_legs = d2.slider("Number of legs", 1, 10, 3, key="sb_n")
        rank_label = st.radio("Pick legs by", ["Highest probability first", "Highest fair odds first (lowest probability in the range)"],
                              horizontal=True, key="sb_rank")
        one = st.checkbox("At most one leg per match", value=True, key="sb_one")
        slip = select_slip(legs_all, lo / 100, hi / 100, n_legs, groups=sel_groups, leagues=sel_leagues,
                           rank="probability" if rank_label.startswith("Highest probability") else "odds", one_per_match=one)
        if slip.empty:
            st.info("No legs meet these criteria. Lower the minimum probability, add markets or leagues, or widen the date window in the sidebar.")
        else:
            if len(slip) < n_legs:
                st.warning(f"Only {len(slip)} leg(s) meet the criteria, fewer than the {n_legs} requested.")
            try:
                when = pd.to_datetime(slip.kickoff_utc, utc=True).dt.tz_convert(tz).dt.strftime("%a %d %b %H:%M")
            except Exception:  # noqa: BLE001
                when = pd.to_datetime(slip.kickoff_utc, utc=True).dt.strftime("%a %d %b %H:%M UTC")
            show = pd.DataFrame({"Kickoff": when.values, "League": slip.league.values, "Match": (slip.home + " v " + slip.away).values,
                                 "Market": slip.group.values, "Selection": slip.selection.values,
                                 "Model %": 100 * slip.prob.values, "Fair odds": slip.fair_odds.values})
            tiers = [tier(p, "binary") for p in slip.prob]
            st.dataframe(style_rows(show, tiers, {"Model %": "{:.1f}", "Fair odds": "{:.2f}"}), hide_index=True, width="stretch")
            st.caption("Green shading shows the model's confidence in each leg (darker = stronger).")
            stt = slip_stats(slip)
            m1c, m2c, m3c = st.columns(3)
            m1c.metric("Chance every leg wins", f"{100 * stt['combined_prob']:.1f}%")
            m2c.metric("Combined fair odds", f"{stt['fair_odds']:.2f}")
            m3c.metric("Expected legs won", f"{stt['expected_wins']:.1f} of {stt['n']}")
            dist = pd.DataFrame({"Legs that win": [f"{k} of {stt['n']}" for k in range(stt["n"], -1, -1)],
                                 "Chance": [100 * stt["dist"][k] for k in range(stt["n"], -1, -1)]})
            with st.expander("Chance of each outcome"):
                st.dataframe(dist.style.format({"Chance": "{:.1f}%"}), hide_index=True, width="stretch")
                st.caption("If each leg also carries a typical bookmaker margin of around 5%, the combined price is "
                           "roughly 5% worse per leg compounded, so longer slips cost more in margin.")
            odds_in = []
            with st.expander("Enter the bookmaker odds you see (optional)"):
                for r in slip.itertuples():
                    odds_in.append(st.number_input(f"{r.selection}  ({r.home} v {r.away})", min_value=0.0, max_value=1000.0, value=0.0,
                                                   step=0.01, key=f"sb_odds_{r.key}_{r.home}_{r.away}"))
                if all(o >= 1.01 for o in odds_in):
                    s2 = slip_stats(slip, odds_in)
                    o1, o2, o3 = st.columns(3)
                    o1.metric("Combined odds", f"{s2['combined_odds']:.2f}")
                    o2.metric("Break-even chance", f"{100 * s2['implied_prob']:.1f}%")
                    o3.metric("Expected value per 1 staked", f"{s2['ev']:+.1%}")
                    if s2["ev"] > 0:
                        st.warning("Positive on paper only. Small gaps are well inside the model's error, and the model has not beaten closing prices.")
                else:
                    st.caption("Enter a price for every leg to see the combined odds and expected value.")
            sv1, sv2 = st.columns([3, 1])
            slip_name = sv1.text_input("Name for this slip (optional)", key="sb_name")
            if sv2.button("Save this slip", key="sb_save"):
                try:
                    use_odds = odds_in if odds_in and all(o >= 1.01 for o in odds_in) else None
                    sid = save_slip(slip, {"min_prob": lo, "max_prob": hi, "n_legs": n_legs, "groups": sel_groups, "leagues": sel_leagues,
                                           "rank": rank_label, "one_per_match": one}, name=slip_name, odds=use_odds)
                    st.success(f"Saved as {sid}. It is scored automatically once the matches are played and results are in.")
                except ValueError as exc:
                    st.error(str(exc))
    saved = load_slips()
    if not saved.empty:
        st.markdown("---")
        st.markdown("### Saved slips")
        leg_tbl, summ = settle_slips(saved, df)
        if not summ.empty:
            done = summ[summ.status != "pending"]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Saved slips", len(summ)); q2.metric("Won", int((summ.status == "won").sum()))
            q3.metric("Lost", int((summ.status == "lost").sum())); q4.metric("Waiting", int((summ.status == "pending").sum()))
            if len(done):
                st.caption(f"Of {len(done)} settled slips, the model's own combined probabilities add up to "
                           f"{done.model_prob.sum():.1f} expected wins; actual wins: {int((done.status == 'won').sum())}.")
            st.dataframe(summ.assign(model_prob=100 * summ.model_prob)[["created_utc", "name", "legs", "won", "lost", "pending", "status", "model_prob", "odds"]]
                         .rename(columns={"created_utc": "Saved (UTC)", "name": "Name", "legs": "Legs", "won": "Won", "lost": "Lost",
                                          "pending": "Waiting", "status": "Slip", "model_prob": "Model chance %", "odds": "Odds"})
                         .style.format({"Model chance %": "{:.1f}", "Odds": "{:.2f}"}), hide_index=True, width="stretch")
            with st.expander("Every leg of every saved slip"):
                st.dataframe(leg_tbl[["slip_id", "leg_no", "kickoff_utc", "home", "away", "selection", "model_prob", "odds", "status"]],
                             hide_index=True, width="stretch")
        st.download_button("Download saved slips (CSV)", data=saved.to_csv(index=False), file_name="slips.csv", mime="text/csv", key="dl_slips")

# ---------------------------------------------------------------- track record
with tab_rec:
    st.markdown("Every prediction shown on the slate is saved **before kickoff** and never edited afterwards "
                "(one snapshot per match per day). Once a match is played and results are refreshed, it is scored here. "
                "This is the honest test: no hindsight, unlike a backtest.")
    log = load_log()
    if log.empty:
        st.info("Nothing recorded yet. Predictions are saved automatically when the Upcoming slate shows matches "
                "with the default model settings.")
    else:
        settled = settle(log, df)
        n_fix = len(log.drop_duplicates(FIXTURE_KEY))
        r1, r2, r3 = st.columns(3)
        r1.metric("Matches recorded", n_fix)
        r2.metric("Played and scored", len(settled))
        r3.metric("Waiting for result", n_fix - len(settled))
        with st.expander("Which matches are recorded, and which are waiting?", expanded=len(settled) == 0):
            st.markdown("**Recorded matches by kickoff date.** A missing date means nothing was saved before those "
                        "kickoffs: the app was closed and the scheduled task was not running.")
            st.dataframe(coverage(log), hide_index=True, width="stretch")
            pend = pending_report(log, df)
            if pend.empty:
                st.caption("Every recorded match that has kicked off has been scored.")
            else:
                st.markdown("**Kicked off but not scored yet**")
                st.dataframe(pend, hide_index=True, width="stretch")
        if settled.empty:
            st.info("None of the recorded matches has been scored yet. After they are played, click 'Refresh results' in the sidebar.")
        else:
            render_result_record(result_record(log, df))
            st.markdown("---")
            st.markdown("**How the model's leans did** (lean = the more likely side; log loss: lower is better)")
            st.dataframe(summarise(settled).style.format({"Lean hit rate": "{:.0%}", "Avg stated prob of lean": "{:.0%}",
                                                          "Avg log loss": "{:.3f}"}), hide_index=True, width="stretch")
            st.markdown("**Full-time result by confidence**")
            st.dataframe(by_confidence(settled).style.format({"Hit_rate": "{:.0%}", "Avg_stated": "{:.0%}"}),
                         hide_index=True, width="stretch")
            vm = versus_market(settled)
            if not vm.empty:
                st.markdown("**Model vs bookmaker closing price** (market average, margin removed; negative = model better)")
                st.dataframe(vm.style.format({"Model log loss": "{:.4f}", "Bookmaker log loss": "{:.4f}",
                                              "Model minus bookmaker": "{:+.4f}", "Std. error": "{:.4f}"}),
                             hide_index=True, width="stretch")
                st.caption("Needs a few hundred matches before the gap means much; the backtest gap was +0.026 on full-time 1X2.")
        st.download_button("Download the full prediction log (CSV)", data=log.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
    cal = get_calibration(df)
    if cal:
        with st.expander("How reliable are the confidence labels? (backtest, 2021/22-2025/26)"):
            st.caption("Stated = average probability the model gave; Actual = how often that outcome happened.")
            names = {"ft_1x2": "Full-time result (top outcome)", "h1_1x2": "1st-half result (top outcome)",
                     "h2_1x2": "2nd-half result (top outcome)", "ft_o25": "Full-time over/under 2.5 (likelier side)",
                     "h1_o15": "1st-half over/under 1.5 (likelier side)", "h2_o15": "2nd-half over/under 1.5 (likelier side)",
                     "ft_btts": "Full-time both teams score (likelier side)"}
            pick = st.selectbox("Market", list(names), format_func=names.get, key="cal_market")
            t = cal[pick].assign(Range=lambda x: (100 * x.lo).round(0).astype(int).astype(str) + "-" +
                                 (100 * x.hi.clip(upper=1)).round(0).astype(int).astype(str) + "%")
            st.dataframe(t[["Range", "n", "mean_p", "hit_rate"]].rename(columns={"n": "Matches", "mean_p": "Stated", "hit_rate": "Actual"})
                         .style.format({"Stated": "{:.1%}", "Actual": "{:.1%}"}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- backtest
with tab_bt:
    p = REPORT_DIR / "backtest_summary.csv"
    if not p.exists():
        st.info("Run `python run_backtest.py` to create the backtest results.")
    else:
        bt = pd.read_csv(p)
        which = st.radio("Evaluation set", sorted(bt["set"].unique()), horizontal=True)
        sub = bt[bt["set"] == which].copy()
        st.markdown("Log loss, **lower is better**. *Naive* = league-average rates with no team information. "
                    "*Bookmaker close* = market-average closing odds with the margin removed (full-time markets only).")
        cols = [c for c in ["market", "n", "model", "naive", "model_minus_naive", "se", "bookmaker_close",
                            "model_minus_market", "se_vs_market"] if c in sub.columns]
        st.dataframe(sub[cols].round(4), hide_index=True, width="stretch")
        chart = sub.set_index("market")[["model_minus_naive"]].rename(
            columns={"model_minus_naive": "model minus naive (negative = better)"})
        st.bar_chart(chart)
        if "model_minus_market" in sub.columns and sub["model_minus_market"].notna().any():
            st.warning("Against the bookmaker closing line the model is WORSE on full-time 1X2 and over/under 2.5 "
                       "(positive differences, several standard errors from zero). No edge has been demonstrated.")
        bs = REPORT_DIR / "backtest_by_season.csv"
        if bs.exists():
            with st.expander("Per-season log loss"):
                st.dataframe(pd.read_csv(bs).round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- data
with tab_data:
    st.markdown("**Matches per league and season**")
    piv = df.pivot_table(index="season", columns="league", values="date", aggfunc="count", fill_value=0)
    st.dataframe(piv, width="stretch")
    g = df.groupby("season").agg(first_half=("ht_total", "mean"), second_half=("h2_total", "mean"))
    st.markdown("**Average goals per match by half**")
    st.line_chart(g)
    hw = df.assign(home_adv=df.ft_home - df.ft_away).groupby("season")["home_adv"].mean().rename("home goals minus away goals")
    st.markdown("**Home advantage by season**")
    st.line_chart(hw)
    cp = REPORT_DIR / "season_completeness.csv"
    if cp.exists():
        comp = pd.read_csv(cp)
        with st.expander("Seasons with missing matches"):
            st.dataframe(comp[(comp.missing != 0) & (comp.season.astype(str) != "2627")], hide_index=True)

# ---------------------------------------------------------------- odds collection
with tab_odds:
    snap = ROOT / "data" / "odds" / "first_half_snapshots.csv"
    logf = ROOT / "data" / "odds" / "collector.log"
    st.markdown("**Automatic collection**")
    status = task_status()
    last_line, mins, key_rejected = collector_health(logf)
    if status == "installed":
        st.success("Scheduled task installed: the collector runs every 10 minutes while you are logged in and the computer is awake.")
    elif status == "not installed":
        st.warning("Automatic collection is not set up yet. Double-click `install_collector_task.bat` in the project folder (one time).")
    else:
        st.info("Cannot check the Windows scheduled task from here. If you installed it, the log below shows whether it runs.")
    if key_rejected:
        st.error("The Odds API rejected your key (HTTP 401). In a terminal in the project folder run "
                 "`python run_odds_collect.py --test-key` to see why, fix the key in `.env`, then run `install_collector_task.bat` again.")
    elif mins is not None:
        txt = f"Last collector run: {mins:.0f} minutes ago." if mins < 120 else f"Last collector run: {mins / 60:.1f} hours ago."
        (st.warning if (status == "installed" and mins > 40) else st.caption)(
            txt + (" It has not run recently: is the computer asleep or logged out?" if status == "installed" and mins > 40 else ""))
    if not snap.exists():
        st.info("The odds collector runs on your own computer, not in this cloud copy." if IS_CLOUD else
                "No snapshots yet. Start the collector (README: Step 4b) and this tab will fill in.")
    else:
        s = pd.read_csv(snap)
        m1c, m2c, m3c = st.columns(3)
        m1c.metric("Matches captured", s.event_id.nunique())
        m2c.metric("Bookmakers seen", s.bookmaker.nunique())
        m3c.metric("Last snapshot (UTC)", str(s.snapshot_utc.max()))
        st.markdown("**Captured per league**")
        st.dataframe(s.groupby("league").event_id.nunique().rename("matches").reset_index(),
                     hide_index=True, width="stretch")
        st.markdown("**Bookmaker coverage by market**")
        st.dataframe(s.groupby(["market", "bookmaker"]).event_id.nunique().rename("matches").reset_index(),
                     hide_index=True, width="stretch")
    if logf.exists():
        with st.expander("Collector log (last 20 lines)"):
            st.code("\n".join(logf.read_text(encoding="utf-8").splitlines()[-20:]))
