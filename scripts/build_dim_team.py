"""
File: build_dim_team.py

Purpose:
Build the dim_team warehouse table from clean_schedules.csv using standardized
team abbreviations and team names. The output contains one row per unique NFL
franchise appearing in the schedule data from 2015 through 2025.

Inputs:
- data/processed/schedules/clean_schedules.csv

Outputs:
- data/processed/warehouse/dim_team.csv
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "schedules" / "clean_schedules.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "warehouse" / "dim_team.csv"

REQUIRED_COLUMNS = [
    "season",
    "home_team_abbr",
    "home_team_name",
    "away_team_abbr",
    "away_team_name",
]

TEAM_NAME_OVERRIDES = {
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
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
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
    "WAS": "Washington Commanders",
}


def load_clean_schedules(file_path: Path) -> pd.DataFrame:
    """Load the clean schedules file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    schedules_df = pd.read_csv(file_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in schedules_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in clean_schedules.csv: {missing_text}")

    return schedules_df


def standardize_team_records(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and standardize unique team records from home and away teams."""
    schedules_df["season"] = pd.to_numeric(schedules_df["season"], errors="coerce")
    schedules_df = schedules_df[schedules_df["season"].between(2015, 2025, inclusive="both")].copy()

    home_teams_df = schedules_df[["home_team_abbr", "home_team_name"]].rename(
        columns={
            "home_team_abbr": "team_abbr",
            "home_team_name": "team_name",
        }
    )

    away_teams_df = schedules_df[["away_team_abbr", "away_team_name"]].rename(
        columns={
            "away_team_abbr": "team_abbr",
            "away_team_name": "team_name",
        }
    )

    teams_df = pd.concat([home_teams_df, away_teams_df], ignore_index=True)

    teams_df["team_abbr"] = teams_df["team_abbr"].astype(str).str.strip().str.upper()
    teams_df["team_name"] = teams_df["team_name"].astype(str).str.strip()

    teams_df = teams_df[
        teams_df["team_abbr"].notna()
        & teams_df["team_name"].notna()
        & (teams_df["team_abbr"] != "")
        & (teams_df["team_name"] != "")
    ].copy()

    teams_df["team_name"] = teams_df["team_abbr"].map(TEAM_NAME_OVERRIDES).fillna(teams_df["team_name"])

    teams_df = (
        teams_df.sort_values(["team_abbr", "team_name"])
        .drop_duplicates(subset=["team_abbr"], keep="first")
        .reset_index(drop=True)
    )

    return teams_df


def add_warehouse_fields(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Add warehouse-ready fields to the team dimension."""
    teams_df["team_id"] = range(1, len(teams_df) + 1)
    teams_df["market"] = teams_df["team_name"].str.split().str[:-1].str.join(" ")
    teams_df["league"] = "NFL"

    dim_team_df = teams_df[["team_id", "team_abbr", "team_name", "market", "league"]].copy()
    dim_team_df = dim_team_df.sort_values("team_id").reset_index(drop=True)

    return dim_team_df


def validate_dim_team(dim_team_df: pd.DataFrame) -> None:
    """Validate the final team dimension output."""
    if dim_team_df.empty:
        raise ValueError("dim_team output is empty.")

    if dim_team_df["team_abbr"].duplicated().any():
        duplicates = dim_team_df.loc[dim_team_df["team_abbr"].duplicated(), "team_abbr"].tolist()
        raise ValueError(f"Duplicate team abbreviations found in dim_team: {duplicates}")

    if dim_team_df["team_id"].duplicated().any():
        raise ValueError("Duplicate team_id values found in dim_team.")

    required_output_columns = ["team_id", "team_abbr", "team_name", "market", "league"]
    missing_output_columns = [col for col in required_output_columns if col not in dim_team_df.columns]
    if missing_output_columns:
        missing_text = ", ".join(missing_output_columns)
        raise ValueError(f"Missing required output columns: {missing_text}")


def write_dim_team(dim_team_df: pd.DataFrame, file_path: Path) -> None:
    """Write the team dimension to CSV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dim_team_df.to_csv(file_path, index=False)


def main() -> None:
    """Run the dim_team build process."""
    schedules_df = load_clean_schedules(INPUT_PATH)
    teams_df = standardize_team_records(schedules_df)
    dim_team_df = add_warehouse_fields(teams_df)
    validate_dim_team(dim_team_df)
    write_dim_team(dim_team_df, OUTPUT_PATH)

    print(f"dim_team file written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()