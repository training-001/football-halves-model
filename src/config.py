"""Project-wide configuration. Add a league by adding one entry to LEAGUES."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

# code -> display name used in the source files `league` column
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Columns required for 1H/2H modelling
REQUIRED = ["league", "season", "date", "home", "away",
            "ft_home", "ft_away", "ht_home", "ht_away"]


# The Odds API sport keys for the same five leagues
LEAGUE_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}
