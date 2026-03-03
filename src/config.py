from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path("C:/AAM_BMKT673")
DATA_DIR = PROJECT_ROOT / "data"
ATTENDANCE_DIR = DATA_DIR / "attendance"

# ---------------------------------------------------------------------
# League + team scope (Week 1–3 deliverable)
# ---------------------------------------------------------------------
LEAGUE = "NFL"

DIVISIONS = {
    "AFC_West": ["Denver Broncos", "Kansas City Chiefs", "Los Angeles Chargers", "Las Vegas Raiders"],
    "NFC_South": ["Atlanta Falcons", "Carolina Panthers", "New Orleans Saints", "Tampa Bay Buccaneers"],
}

SCOPE_TEAMS = DIVISIONS["AFC_West"] + DIVISIONS["NFC_South"]

# Sports Reference uses full names; nflverse uses abbreviations.
TEAM_NAME_TO_ABBR = {
    "Denver Broncos": "DEN",
    "Kansas City Chiefs": "KC",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV",
    "Atlanta Falcons": "ATL",
    "Carolina Panthers": "CAR",
    "New Orleans Saints": "NO",
    "Tampa Bay Buccaneers": "TB",
}

SCOPE_TEAM_ABBRS = set(TEAM_NAME_TO_ABBR.values())

# ---------------------------------------------------------------------
# Project time window
# ---------------------------------------------------------------------
TRAIN_YEARS = list(range(2015, 2024))  # 2015–2023
HOLDOUT_YEAR = 2024
ALL_YEARS = TRAIN_YEARS + [HOLDOUT_YEAR]

# ---------------------------------------------------------------------
# Stadium coordinates for weather (approximate; acceptable for daily weather)
# ---------------------------------------------------------------------
STADIUM_COORDS = {
    "DEN": (39.7439, -105.0201),  # Empower Field at Mile High
    "KC": (39.0489, -94.4840),    # GEHA Field at Arrowhead Stadium
    "LAC": (33.9535, -118.3392),  # SoFi Stadium
    "LV": (36.0909, -115.1830),   # Allegiant Stadium
    "ATL": (33.7554, -84.4013),   # Mercedes-Benz Stadium
    "CAR": (35.2258, -80.8528),   # Bank of America Stadium
    "NO": (29.9509, -90.0811),    # Caesars Superdome
    "TB": (27.9759, -82.5033),    # Raymond James Stadium
}
