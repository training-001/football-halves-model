"""Team-name canonicalisation.

Canonical names are the short names used by football-data.co.uk, because that is
where the long history comes from. The 2026/27 feeds (fixturedl / openligadb) use
longer official names, so they are mapped onto the canonical ones here.

Rules
-----
* Only explicit aliases are applied. There is NO fuzzy matching: a wrong silent
  merge of two clubs is worse than a loud failure.
* Names with no alias pass through unchanged (correct for football-data names).
* After building the dataset, `run_build_history.py` writes reports/team_review.csv
  listing current-season teams never seen in earlier seasons of the same league.
  Genuine promoted clubs appear there, and so does any alias you still need to add.
"""
from __future__ import annotations
import re
import unicodedata


def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


# variant -> canonical (football-data.co.uk spelling)
_ALIASES = {
    # --- Premier League ---
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    # --- La Liga ---
    "Athletic Club": "Ath Bilbao",
    "Atlético de Madrid": "Ath Madrid",
    "CA Osasuna": "Osasuna",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "R. Racing Club": "Santander",
    "RC Deportivo": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Sociedad": "Sociedad",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Atletico Madrid": "Ath Madrid",
    "Club Atletico de Madrid": "Ath Madrid",
    "Athletic Bilbao": "Ath Bilbao",
    "Espanyol": "Espanol",
    # --- Serie A ---
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    # --- Bundesliga ---
    "FC Cologne": "FC Koln",
    "1. FC Köln": "FC Koln",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "FC Bayern München": "Bayern Munich",
    "FC Schalke 04": "Schalke 04",
    "Hamburger SV": "Hamburg",
    "SC Freiburg": "Freiburg",
    "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg",
    "SV Werder Bremen": "Werder Bremen",
    "TSG Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    # --- Ligue 1 ---
    "AJ Auxerre": "Auxerre",
    "AS Monaco": "Monaco",
    "Angers SCO": "Angers",
    "Estac Troyes": "Troyes",
    "FC Lorient": "Lorient",
    "Havre Athletic Club": "Le Havre",
    "LOSC Lille": "Lille",
    "Le Mans FC": "Le Mans",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "Paris Saint-Germain": "Paris SG",
    "Paris Saint Germain": "Paris SG",
    "RC Lens": "Lens",
    "RC Strasbourg Alsace": "Strasbourg",
    "Stade Brestois 29": "Brest",
    "Stade Rennais FC": "Rennes",
    "Stade Rennais FC 1901": "Rennes",
    "Toulouse FC": "Toulouse",
}

ALIASES = {_key(k): v for k, v in _ALIASES.items()}


def canonical(name: str) -> str:
    """Return the canonical team name (unchanged if no alias exists)."""
    name = re.sub(r"\s+", " ", str(name)).strip()
    return ALIASES.get(_key(name), name)
