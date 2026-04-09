"""
File: clean_schedules.py

Purpose:
Clean and standardize raw NFL schedule data at the game level for downstream
warehouse loading, weather matching, feature engineering, and modeling.

Inputs:
* data/raw/schedules/nfl_schedules_raw.csv

Outputs:
* data/processed/schedules/clean_schedules.csv
* data/processed/schedules/team_alias_reference.csv
* data/processed/schedules/venue_reference.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_SCHEDULES_PATH = Path("data/raw/schedules/nfl_schedules_raw.csv")
OUTPUT_DIR = Path("data/processed/schedules")
CLEAN_SCHEDULES_PATH = OUTPUT_DIR / "clean_schedules.csv"
TEAM_ALIAS_REFERENCE_PATH = OUTPUT_DIR / "team_alias_reference.csv"
VENUE_REFERENCE_PATH = OUTPUT_DIR / "venue_reference.csv"


TEAM_ABBREVIATION_MAP = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "CLV": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "GNB": "GB",
    "HOU": "HOU",
    "HTX": "HOU",
    "IND": "IND",
    "CLT": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KAN": "KC",
    "KC": "KC",
    "LA": "LAR",
    "LAC": "LAC",
    "LAR": "LAR",
    "LV": "LV",
    "LVR": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NWE": "NE",
    "NO": "NO",
    "NOR": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "OAK": "LV",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "LAC",
    "SFO": "SF",
    "SF": "SF",
    "SEA": "SEA",
    "STL": "LAR",
    "TB": "TB",
    "TAM": "TB",
    "TEN": "TEN",
    "OTI": "TEN",
    "WAS": "WSH",
    "WFT": "WSH",
    "WSH": "WSH",
}

TEAM_NAME_TO_CURRENT_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "jacksonville jags": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "oakland raiders": "LV",
    "los angeles chargers": "LAC",
    "san diego chargers": "LAC",
    "los angeles rams": "LAR",
    "st. louis rams": "LAR",
    "st louis rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WSH",
    "washington football team": "WSH",
    "washington redskins": "WSH",
}

CURRENT_TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WSH": "Washington Commanders",
}

VENUE_NORMALIZATION_MAP = {
    "arrowhead stadium": "Arrowhead Stadium",
    "state farm stadium": "State Farm Stadium",
    "university of phoenix stadium": "State Farm Stadium",
    "mercedes-benz stadium": "Mercedes-Benz Stadium",
    "mercedes benz stadium": "Mercedes-Benz Stadium",
    "m&t bank stadium": "M&T Bank Stadium",
    "new era field": "Highmark Stadium",
    "highmark stadium": "Highmark Stadium",
    "ralph wilson stadium": "Highmark Stadium",
    "bank of america stadium": "Bank of America Stadium",
    "soldier field": "Soldier Field",
    "paycor stadium": "Paycor Stadium",
    "paul brown stadium": "Paycor Stadium",
    "huntington bank field": "Huntington Bank Field",
    "firstenergy stadium": "Huntington Bank Field",
    "at&t stadium": "AT&T Stadium",
    "empower field at mile high": "Empower Field at Mile High",
    "sports authority field at mile high": "Empower Field at Mile High",
    "invesco field at mile high": "Empower Field at Mile High",
    "ford field": "Ford Field",
    "lambeau field": "Lambeau Field",
    "nrg stadium": "NRG Stadium",
    "lucas oil stadium": "Lucas Oil Stadium",
    "everbank stadium": "EverBank Stadium",
    "everbank field": "EverBank Stadium",
    "tiaa bank field": "EverBank Stadium",
    "alltel stadium": "EverBank Stadium",
    "geha field at arrowhead stadium": "Arrowhead Stadium",
    "sofi stadium": "SoFi Stadium",
    "sofi": "SoFi Stadium",
    "allegiant stadium": "Allegiant Stadium",
    "oakland-alameda county coliseum": "Oakland Coliseum",
    "o.co coliseum": "Oakland Coliseum",
    "ringcentral coliseum": "Oakland Coliseum",
    "hard rock stadium": "Hard Rock Stadium",
    "sun life stadium": "Hard Rock Stadium",
    "u.s. bank stadium": "U.S. Bank Stadium",
    "us bank stadium": "U.S. Bank Stadium",
    "gillette stadium": "Gillette Stadium",
    "caesars superdome": "Caesars Superdome",
    "mercedes-benz superdome": "Caesars Superdome",
    "metlife stadium": "MetLife Stadium",
    "lincoln financial field": "Lincoln Financial Field",
    "acrisure stadium": "Acrisure Stadium",
    "heinz field": "Acrisure Stadium",
    "levi's stadium": "Levi's Stadium",
    "levis stadium": "Levi's Stadium",
    "lumen field": "Lumen Field",
    "centurylink field": "Lumen Field",
    "raymond james stadium": "Raymond James Stadium",
    "nissan stadium": "Nissan Stadium",
    "fedexfield": "Northwest Stadium",
    "fedex field": "Northwest Stadium",
    "northwest stadium": "Northwest Stadium",
    "qualcomm stadium": "Qualcomm Stadium",
    "dignity health sports park": "Dignity Health Sports Park",
    "los angeles memorial coliseum": "Los Angeles Memorial Coliseum",
    "edward jones dome": "Edward Jones Dome",
    "the dome at america's center": "The Dome at America's Center",
    "wembley stadium": "Wembley Stadium",
    "tottenham hotspur stadium": "Tottenham Hotspur Stadium",
    "allianz arena": "Allianz Arena",
    "deutsche bank park": "Deutsche Bank Park",
    "estadio azteca": "Estadio Azteca",
    "twickenham stadium": "Twickenham Stadium",
    "estadio banorte": "Estadio Banorte",
    "azteca stadium": "Estadio Azteca",
    "tottenham stadium": "Tottenham Hotspur Stadium",
    "tom benson hall of fame stadium": "Tom Benson Hall of Fame Stadium",
}

TEAM_HOME_VENUE_MAP = {
    "ARI": "State Farm Stadium",
    "ATL": "Mercedes-Benz Stadium",
    "BAL": "M&T Bank Stadium",
    "BUF": "Highmark Stadium",
    "CAR": "Bank of America Stadium",
    "CHI": "Soldier Field",
    "CIN": "Paycor Stadium",
    "CLE": "Huntington Bank Field",
    "DAL": "AT&T Stadium",
    "DEN": "Empower Field at Mile High",
    "DET": "Ford Field",
    "GB": "Lambeau Field",
    "HOU": "NRG Stadium",
    "IND": "Lucas Oil Stadium",
    "JAX": "EverBank Stadium",
    "KC": "Arrowhead Stadium",
    "LAC": "SoFi Stadium",
    "LAR": "SoFi Stadium",
    "LV": "Allegiant Stadium",
    "MIA": "Hard Rock Stadium",
    "MIN": "U.S. Bank Stadium",
    "NE": "Gillette Stadium",
    "NO": "Caesars Superdome",
    "NYG": "MetLife Stadium",
    "NYJ": "MetLife Stadium",
    "PHI": "Lincoln Financial Field",
    "PIT": "Acrisure Stadium",
    "SEA": "Lumen Field",
    "SF": "Levi's Stadium",
    "TB": "Raymond James Stadium",
    "TEN": "Nissan Stadium",
    "WSH": "Northwest Stadium",
}

INTERNATIONAL_COUNTRY_MAP = {
    "Wembley Stadium": "United Kingdom",
    "Tottenham Hotspur Stadium": "United Kingdom",
    "Allianz Arena": "Germany",
    "Deutsche Bank Park": "Germany",
    "Estadio Azteca": "Mexico",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = " ".join(text.split())
    return text


def clean_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = text.replace(".", "")
    return text


def resolve_team_abbreviation(row: pd.Series, prefix: str) -> str | None:
    abbr_candidates = [
        row.get(f"{prefix}_team"),
        row.get(f"{prefix}_abbr"),
        row.get(f"{prefix}_team_abbr"),
        row.get(f"{prefix}_team_code"),
        row.get(f"{prefix}"),
    ]

    name_candidates = [
        row.get(f"{prefix}_team_name"),
        row.get(f"{prefix}_name"),
        row.get(f"{prefix}_team_full"),
    ]

    for candidate in abbr_candidates:
        key = normalize_text(candidate).upper()
        if key in TEAM_ABBREVIATION_MAP:
            mapped_value = TEAM_ABBREVIATION_MAP[key]
            if mapped_value is not None:
                return mapped_value

    for candidate in name_candidates:
        key = clean_key(candidate)
        if key in TEAM_NAME_TO_CURRENT_ABBR:
            return TEAM_NAME_TO_CURRENT_ABBR[key]

    return None


def resolve_team_name(team_abbr: str | None) -> str | None:
    if team_abbr is None:
        return None
    return CURRENT_TEAM_NAME_MAP.get(team_abbr)


def normalize_venue_name(value: object) -> str | None:
    raw_value = normalize_text(value)
    if not raw_value:
        return None

    cleaned_value = VENUE_NORMALIZATION_MAP.get(clean_key(raw_value))
    if cleaned_value:
        return cleaned_value

    return raw_value


def infer_venue(row: pd.Series) -> str | None:
    venue_candidates = [
        row.get("stadium"),
        row.get("game_stadium"),
        row.get("venue"),
        row.get("location"),
        row.get("game_location"),
        row.get("site"),
    ]

    for candidate in venue_candidates:
        normalized = normalize_venue_name(candidate)
        if normalized:
            return normalized

    home_team = row.get("home_team_abbr")
    if home_team in TEAM_HOME_VENUE_MAP:
        return TEAM_HOME_VENUE_MAP[home_team]

    return None


def infer_game_date(df: pd.DataFrame) -> pd.Series:
    date_candidates = ["game_date", "gameday", "date"]

    for column in date_candidates:
        if column in df.columns:
            return pd.to_datetime(df[column], errors="coerce")

    if {"season", "month", "day"}.issubset(df.columns):
        date_string = (
            df["season"].astype("Int64").astype(str)
            + "-"
            + df["month"].astype("Int64").astype(str).str.zfill(2)
            + "-"
            + df["day"].astype("Int64").astype(str).str.zfill(2)
        )
        return pd.to_datetime(date_string, errors="coerce")

    return pd.Series(pd.NaT, index=df.index)


def infer_kickoff_time(df: pd.DataFrame) -> pd.Series:
    time_candidates = ["start_time", "gametime", "kickoff_time", "time"]

    for column in time_candidates:
        if column in df.columns:
            return df[column].astype("string").str.strip()

    return pd.Series(pd.NA, index=df.index, dtype="string")


def infer_game_type(df: pd.DataFrame) -> pd.Series:
    if "game_type" in df.columns:
        series = df["game_type"].astype("string").str.upper().str.strip()
    elif "season_type" in df.columns:
        series = df["season_type"].astype("string").str.upper().str.strip()
    else:
        series = pd.Series("REG", index=df.index, dtype="string")

    series = series.replace(
        {
            "REGULAR": "REG",
            "REGULAR SEASON": "REG",
            "POST": "POST",
            "PLAYOFF": "POST",
            "PLAYOFFS": "POST",
            "WC": "POST",
            "DIV": "POST",
            "CONF": "POST",
            "SB": "POST",
            "PRE": "PRE",
            "PRESEASON": "PRE",
        }
    )

    return series


def infer_week(df: pd.DataFrame) -> pd.Series:
    if "week" in df.columns:
        return pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    return pd.Series(pd.NA, index=df.index, dtype="Int64")


def detect_neutral_site(row: pd.Series) -> bool:
    explicit_candidates = [
        row.get("neutral_site"),
        row.get("is_neutral_site"),
        row.get("neutral"),
    ]

    for candidate in explicit_candidates:
        if pd.isna(candidate):
            continue
        value = str(candidate).strip().lower()
        if value in {"1", "true", "t", "yes", "y"}:
            return True
        if value in {"0", "false", "f", "no", "n"}:
            return False

    venue = row.get("venue_clean")
    home_team = row.get("home_team_abbr")
    if venue and home_team in TEAM_HOME_VENUE_MAP:
        return venue != TEAM_HOME_VENUE_MAP[home_team]

    return False


def detect_international_game(row: pd.Series) -> bool:
    venue = row.get("venue_clean")
    if venue in INTERNATIONAL_COUNTRY_MAP:
        return True

    country_candidates = [
        row.get("country"),
        row.get("game_country"),
        row.get("site_country"),
    ]
    for candidate in country_candidates:
        country = clean_key(candidate)
        if country and country not in {"usa", "united states", "united states of america"}:
            return True

    return False


def infer_country(row: pd.Series) -> str:
    venue = row.get("venue_clean")
    if venue in INTERNATIONAL_COUNTRY_MAP:
        return INTERNATIONAL_COUNTRY_MAP[venue]

    country_candidates = [
        row.get("country"),
        row.get("game_country"),
        row.get("site_country"),
    ]
    for candidate in country_candidates:
        normalized = normalize_text(candidate)
        if normalized:
            return normalized

    return "United States"


def infer_city(row: pd.Series) -> str | None:
    city_candidates = [
        row.get("city"),
        row.get("game_city"),
        row.get("site_city"),
    ]
    for candidate in city_candidates:
        normalized = normalize_text(candidate)
        if normalized:
            return normalized

    venue = row.get("venue_clean")
    if venue == "Wembley Stadium":
        return "London"
    if venue == "Tottenham Hotspur Stadium":
        return "London"
    if venue == "Allianz Arena":
        return "Munich"
    if venue == "Deutsche Bank Park":
        return "Frankfurt"
    if venue == "Estadio Azteca":
        return "Mexico City"
    if venue == "Tom Benson Hall of Fame Stadium":
        return "Canton"

    return None


def build_game_id(row: pd.Series) -> str:
    season = row.get("season")
    week = row.get("week")
    game_type = row.get("game_type")
    away = row.get("away_team_abbr")
    home = row.get("home_team_abbr")
    date_value = row.get("game_date")

    season_str = str(int(season)) if pd.notna(season) else "unknown"
    week_str = f"{int(week):02d}" if pd.notna(week) else "00"
    game_type_str = normalize_text(game_type).upper() or "UNK"
    away_str = away or "UNK"
    home_str = home or "UNK"
    date_str = date_value.strftime("%Y%m%d") if pd.notna(date_value) else "nodate"

    return f"{season_str}_{game_type_str}_{week_str}_{away_str}_{home_str}_{date_str}"


def load_raw_schedules() -> pd.DataFrame:
    return pd.read_csv(RAW_SCHEDULES_PATH)


def build_clean_schedules(raw_df: pd.DataFrame) -> pd.DataFrame:
    schedules_df = raw_df.copy()

    schedules_df["game_date"] = infer_game_date(schedules_df)
    schedules_df["kickoff_time_raw"] = infer_kickoff_time(schedules_df)
    schedules_df["game_type"] = infer_game_type(schedules_df)
    schedules_df["week"] = infer_week(schedules_df)

    if "season" in schedules_df.columns:
        schedules_df["season"] = pd.to_numeric(
            schedules_df["season"], errors="coerce"
        ).astype("Int64")
    else:
        schedules_df["season"] = schedules_df["game_date"].dt.year.astype("Int64")

    schedules_df["home_team_raw"] = (
    schedules_df.get("home_team", pd.Series(index=schedules_df.index, dtype="object"))
    .fillna(schedules_df.get("home_team_name", pd.Series(index=schedules_df.index, dtype="object")))
    .fillna(schedules_df.get("home", pd.Series(index=schedules_df.index, dtype="object")))
    .astype("string")
)

    schedules_df["away_team_raw"] = (
        schedules_df.get("away_team", pd.Series(index=schedules_df.index, dtype="object"))
        .fillna(schedules_df.get("away_team_name", pd.Series(index=schedules_df.index, dtype="object")))
        .fillna(schedules_df.get("away", pd.Series(index=schedules_df.index, dtype="object")))
        .astype("string")
    )

    schedules_df["home_team_abbr"] = schedules_df.apply(
        lambda row: resolve_team_abbreviation(row, "home"),
        axis=1,
    )
    schedules_df["away_team_abbr"] = schedules_df.apply(
        lambda row: resolve_team_abbreviation(row, "away"),
        axis=1,
    )

    schedules_df["home_team_name"] = schedules_df["home_team_abbr"].map(resolve_team_name)
    schedules_df["away_team_name"] = schedules_df["away_team_abbr"].map(resolve_team_name)

    schedules_df["venue_raw"] = (
        schedules_df.get("stadium", pd.Series(index=schedules_df.index))
        .fillna(schedules_df.get("venue", pd.Series(index=schedules_df.index)))
        .fillna(schedules_df.get("location", pd.Series(index=schedules_df.index)))
        .astype("string")
    )

    schedules_df["venue_clean"] = schedules_df.apply(infer_venue, axis=1)
    schedules_df["is_neutral_site"] = schedules_df.apply(detect_neutral_site, axis=1)
    schedules_df["is_international"] = schedules_df.apply(detect_international_game, axis=1)
    schedules_df["game_country"] = schedules_df.apply(infer_country, axis=1)
    schedules_df["game_city"] = schedules_df.apply(infer_city, axis=1)

    schedules_df["home_stadium_venue"] = schedules_df["home_team_abbr"].map(TEAM_HOME_VENUE_MAP)

    schedules_df["weather_match_venue"] = np.where(
        schedules_df["is_neutral_site"],
        schedules_df["venue_clean"],
        schedules_df["home_stadium_venue"],
    )

    schedules_df["weather_match_type"] = np.where(
        schedules_df["is_neutral_site"],
        "actual_game_venue",
        "home_team_primary_stadium",
    )

    schedules_df["game_id"] = schedules_df.apply(build_game_id, axis=1)

    clean_columns = [
        "game_id",
        "season",
        "game_type",
        "week",
        "game_date",
        "kickoff_time_raw",
        "away_team_abbr",
        "away_team_name",
        "away_team_raw",
        "home_team_abbr",
        "home_team_name",
        "home_team_raw",
        "venue_clean",
        "venue_raw",
        "home_stadium_venue",
        "weather_match_venue",
        "weather_match_type",
        "game_city",
        "game_country",
        "is_neutral_site",
        "is_international",
    ]

    clean_df = schedules_df[clean_columns].copy()
    clean_df = clean_df.sort_values(
        by=["season", "game_date", "week", "away_team_abbr", "home_team_abbr"]
    ).reset_index(drop=True)

    return clean_df


def build_team_alias_reference(clean_df: pd.DataFrame) -> pd.DataFrame:
    home_alias_df = clean_df[
        ["home_team_raw", "home_team_abbr", "home_team_name"]
    ].rename(
        columns={
            "home_team_raw": "raw_team_value",
            "home_team_abbr": "team_abbr_standard",
            "home_team_name": "team_name_standard",
        }
    )

    away_alias_df = clean_df[
        ["away_team_raw", "away_team_abbr", "away_team_name"]
    ].rename(
        columns={
            "away_team_raw": "raw_team_value",
            "away_team_abbr": "team_abbr_standard",
            "away_team_name": "team_name_standard",
        }
    )

    alias_df = pd.concat([home_alias_df, away_alias_df], ignore_index=True)
    alias_df["raw_team_value"] = alias_df["raw_team_value"].astype("string").str.strip()
    alias_df = alias_df.dropna(subset=["raw_team_value", "team_abbr_standard"])
    alias_df = alias_df.drop_duplicates().sort_values(
        by=["team_abbr_standard", "raw_team_value"]
    ).reset_index(drop=True)

    return alias_df


def build_venue_reference(clean_df: pd.DataFrame) -> pd.DataFrame:
    venue_df = clean_df[
        [
            "venue_raw",
            "venue_clean",
            "home_team_abbr",
            "home_stadium_venue",
            "weather_match_venue",
            "is_neutral_site",
            "is_international",
            "game_city",
            "game_country",
        ]
    ].copy()

    venue_df["venue_raw"] = venue_df["venue_raw"].astype("string").str.strip()
    venue_df = venue_df.drop_duplicates().sort_values(
        by=["venue_clean", "venue_raw", "home_team_abbr"]
    ).reset_index(drop=True)

    return venue_df


def validate_clean_schedules(clean_df: pd.DataFrame) -> None:
    required_columns = [
        "game_id",
        "season",
        "game_date",
        "away_team_abbr",
        "home_team_abbr",
        "venue_clean",
        "weather_match_venue",
    ]

    missing_counts = clean_df[required_columns].isna().sum()
    failing_columns = missing_counts[missing_counts > 0]

    if not failing_columns.empty:
        missing_text = ", ".join(
            f"{column}={count}" for column, count in failing_columns.items()
        )
        raise ValueError(f"Missing required clean schedule values: {missing_text}")

    if clean_df["game_id"].duplicated().any():
        duplicate_count = int(clean_df["game_id"].duplicated().sum())
        raise ValueError(f"Duplicate game_id values found: {duplicate_count}")


def write_outputs(
    clean_df: pd.DataFrame,
    alias_df: pd.DataFrame,
    venue_df: pd.DataFrame,
) -> None:
    clean_df.to_csv(CLEAN_SCHEDULES_PATH, index=False)
    alias_df.to_csv(TEAM_ALIAS_REFERENCE_PATH, index=False)
    venue_df.to_csv(VENUE_REFERENCE_PATH, index=False)


def main() -> None:
    ensure_output_dir()
    raw_df = load_raw_schedules()
    clean_df = build_clean_schedules(raw_df)
    validate_clean_schedules(clean_df)
    alias_df = build_team_alias_reference(clean_df)
    venue_df = build_venue_reference(clean_df)
    write_outputs(clean_df, alias_df, venue_df)


if __name__ == "__main__":
    main()