"""
File: build_fact_game.py

Purpose:
Build the fact_game table at one row per game using clean_schedules.csv as the
game backbone and joining dimension keys from dim_team, dim_venue, dim_date,
and matched attendance data.

Inputs:
- data/processed/schedules/clean_schedules.csv
- data/processed/game_attendance_matched.csv
- data/processed/warehouse/dim_team.csv
- data/processed/warehouse/dim_venue.csv
- data/processed/warehouse/dim_date.csv

Outputs:
- data/processed/warehouse/fact_game.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_SCHEDULES_PATH = PROJECT_ROOT / "data" / "processed" / "schedules" / "clean_schedules.csv"
MATCHED_ATTENDANCE_PATH = PROJECT_ROOT / "data" / "processed" / "game_attendance_matched.csv"
DIM_TEAM_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_team.csv"
DIM_VENUE_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_venue.csv"
DIM_DATE_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_date.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_game.csv"

EXPECTED_OUTPUT_COLUMNS = [
    "game_id",
    "date_id",
    "home_team_id",
    "away_team_id",
    "venue_id",
    "attendance",
    "start_time",
    "season",
    "week",
    "game_type",
]

REQUIRED_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "game_date",
    "kickoff_time_raw",
    "home_team_abbr",
    "away_team_abbr",
    "venue_clean",
]

REQUIRED_ATTENDANCE_COLUMNS = [
    "game_id",
    "attendance",
]

REQUIRED_DIM_TEAM_COLUMNS = [
    "team_id",
    "team_abbr",
]

REQUIRED_DIM_DATE_COLUMNS = [
    "date_id",
    "date",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    logging.info("Loading file: %s", file_path)
    return pd.read_csv(file_path)


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{df_name} is missing required columns: {missing_text}")


def clean_text_value(value: object) -> str | None:
    if pd.isna(value):
        return None

    cleaned_value = str(value).strip()
    if not cleaned_value or cleaned_value.lower() in {"nan", "none", "null"}:
        return None

    return cleaned_value


def parse_game_dates(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.copy()
    schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"], errors="coerce")

    missing_dates = int(schedule_df["game_date"].isna().sum())
    if missing_dates > 0:
        raise ValueError(f"Unable to parse {missing_dates} game_date values in clean_schedules.csv")

    schedule_df["game_date"] = schedule_df["game_date"].dt.normalize()
    return schedule_df


def parse_dim_dates(dim_date_df: pd.DataFrame) -> pd.DataFrame:
    dim_date_df = dim_date_df.copy()
    dim_date_df["date"] = pd.to_datetime(dim_date_df["date"], errors="coerce")

    missing_dates = int(dim_date_df["date"].isna().sum())
    if missing_dates > 0:
        raise ValueError(f"Unable to parse {missing_dates} date values in dim_date.csv")

    dim_date_df["date"] = dim_date_df["date"].dt.normalize()
    return dim_date_df


def standardize_schedule_fields(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.copy()

    schedule_df["game_id"] = schedule_df["game_id"].astype(str).str.strip()
    schedule_df["home_team_abbr"] = schedule_df["home_team_abbr"].astype(str).str.strip().str.upper()
    schedule_df["away_team_abbr"] = schedule_df["away_team_abbr"].astype(str).str.strip().str.upper()
    schedule_df["venue_clean"] = schedule_df["venue_clean"].apply(clean_text_value)
    schedule_df["game_type"] = schedule_df["game_type"].astype(str).str.strip().str.upper()
    schedule_df["season"] = pd.to_numeric(schedule_df["season"], errors="coerce").astype("Int64")
    schedule_df["week"] = pd.to_numeric(schedule_df["week"], errors="coerce").astype("Int64")
    schedule_df["start_time"] = schedule_df["kickoff_time_raw"].apply(clean_text_value)

    return schedule_df


def standardize_team_dimension(dim_team_df: pd.DataFrame) -> pd.DataFrame:
    dim_team_df = dim_team_df.copy()
    dim_team_df["team_abbr"] = dim_team_df["team_abbr"].astype(str).str.strip().str.upper()
    dim_team_df["team_id"] = pd.to_numeric(dim_team_df["team_id"], errors="coerce").astype("Int64")
    dim_team_df = dim_team_df.dropna(subset=["team_abbr", "team_id"])
    dim_team_df = dim_team_df.drop_duplicates(subset=["team_abbr"], keep="first")
    return dim_team_df


def standardize_venue_dimension(dim_venue_df: pd.DataFrame) -> pd.DataFrame:
    dim_venue_df = dim_venue_df.copy()

    if "venue_clean" in dim_venue_df.columns:
        venue_name_column = "venue_clean"
    elif "venue_name" in dim_venue_df.columns:
        venue_name_column = "venue_name"
    else:
        raise ValueError("dim_venue must contain either 'venue_clean' or 'venue_name'")

    dim_venue_df["venue_clean"] = dim_venue_df[venue_name_column].apply(clean_text_value)
    dim_venue_df["venue_id"] = pd.to_numeric(dim_venue_df["venue_id"], errors="coerce").astype("Int64")
    dim_venue_df = dim_venue_df.dropna(subset=["venue_clean", "venue_id"])
    dim_venue_df = dim_venue_df.drop_duplicates(subset=["venue_clean"], keep="first")

    return dim_venue_df[["venue_id", "venue_clean"]].copy()


def standardize_attendance(attendance_df: pd.DataFrame) -> pd.DataFrame:
    attendance_df = attendance_df.copy()
    attendance_df["game_id"] = attendance_df["game_id"].astype(str).str.strip()
    attendance_df["attendance"] = pd.to_numeric(attendance_df["attendance"], errors="coerce").astype("Int64")
    attendance_df = attendance_df.drop_duplicates(subset=["game_id"], keep="first")
    return attendance_df


def build_team_lookup(dim_team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    home_team_lookup = dim_team_df.rename(
        columns={
            "team_abbr": "home_team_abbr",
            "team_id": "home_team_id",
        }
    )[["home_team_abbr", "home_team_id"]]

    away_team_lookup = dim_team_df.rename(
        columns={
            "team_abbr": "away_team_abbr",
            "team_id": "away_team_id",
        }
    )[["away_team_abbr", "away_team_id"]]

    return home_team_lookup, away_team_lookup


def merge_dimension_keys(
    schedule_df: pd.DataFrame,
    dim_team_df: pd.DataFrame,
    dim_venue_df: pd.DataFrame,
    dim_date_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
) -> pd.DataFrame:
    home_team_lookup, away_team_lookup = build_team_lookup(dim_team_df)

    fact_df = schedule_df.merge(
        dim_date_df[["date_id", "date"]],
        left_on="game_date",
        right_on="date",
        how="left",
    )

    fact_df = fact_df.merge(
        home_team_lookup,
        on="home_team_abbr",
        how="left",
    )

    fact_df = fact_df.merge(
        away_team_lookup,
        on="away_team_abbr",
        how="left",
    )

    fact_df = fact_df.merge(
        dim_venue_df,
        on="venue_clean",
        how="left",
    )

    fact_df = fact_df.merge(
        attendance_df[["game_id", "attendance"]],
        on="game_id",
        how="left",
    )

    return fact_df


def validate_fact_game(fact_df: pd.DataFrame) -> None:
    duplicate_games = int(fact_df["game_id"].duplicated().sum())
    if duplicate_games > 0:
        raise ValueError(f"Duplicate game_id values found in fact_game output: {duplicate_games}")

    required_columns = [
        "game_id",
        "date_id",
        "home_team_id",
        "away_team_id",
        "venue_id",
        "season",
        "week",
        "game_type",
    ]

    missing_summary: list[str] = []
    for column in required_columns:
        missing_count = int(fact_df[column].isna().sum())
        if missing_count > 0:
            missing_summary.append(f"{column}={missing_count}")

    if missing_summary:
        raise ValueError(f"Missing required fact_game values: {', '.join(missing_summary)}")

    invalid_attendance_count = int((fact_df["attendance"].dropna() <= 0).sum())
    if invalid_attendance_count > 0:
        raise ValueError(f"Found {invalid_attendance_count} non-positive attendance values")

    missing_output_columns = [column for column in EXPECTED_OUTPUT_COLUMNS if column not in fact_df.columns]
    if missing_output_columns:
        raise ValueError(f"Missing expected output columns: {', '.join(missing_output_columns)}")


def log_attendance_coverage(fact_df: pd.DataFrame) -> None:
    total_games = len(fact_df)
    matched_attendance = int(fact_df["attendance"].notna().sum())
    missing_attendance = int(fact_df["attendance"].isna().sum())

    logging.info("Total games in fact_game: %s", total_games)
    logging.info("Games with attendance: %s", matched_attendance)
    logging.info("Games missing attendance: %s", missing_attendance)


def finalize_fact_game(fact_df: pd.DataFrame) -> pd.DataFrame:
    fact_df = fact_df.copy()

    fact_df["date_id"] = pd.to_numeric(fact_df["date_id"], errors="coerce").astype("Int64")
    fact_df["home_team_id"] = pd.to_numeric(fact_df["home_team_id"], errors="coerce").astype("Int64")
    fact_df["away_team_id"] = pd.to_numeric(fact_df["away_team_id"], errors="coerce").astype("Int64")
    fact_df["venue_id"] = pd.to_numeric(fact_df["venue_id"], errors="coerce").astype("Int64")
    fact_df["attendance"] = pd.to_numeric(fact_df["attendance"], errors="coerce").astype("Int64")
    fact_df["season"] = pd.to_numeric(fact_df["season"], errors="coerce").astype("Int64")
    fact_df["week"] = pd.to_numeric(fact_df["week"], errors="coerce").astype("Int64")

    fact_df = fact_df[EXPECTED_OUTPUT_COLUMNS].copy()
    fact_df = fact_df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    return fact_df


def write_output(fact_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact_df.to_csv(output_path, index=False)
    logging.info("fact_game file written: %s", output_path)


def main() -> None:
    configure_logging()

    schedules_df = load_csv(CLEAN_SCHEDULES_PATH)
    attendance_df = load_csv(MATCHED_ATTENDANCE_PATH)
    dim_team_df = load_csv(DIM_TEAM_PATH)
    dim_venue_df = load_csv(DIM_VENUE_PATH)
    dim_date_df = load_csv(DIM_DATE_PATH)

    validate_required_columns(schedules_df, REQUIRED_SCHEDULE_COLUMNS, "clean_schedules")
    validate_required_columns(attendance_df, REQUIRED_ATTENDANCE_COLUMNS, "game_attendance_matched")
    validate_required_columns(dim_team_df, REQUIRED_DIM_TEAM_COLUMNS, "dim_team")
    validate_required_columns(dim_date_df, REQUIRED_DIM_DATE_COLUMNS, "dim_date")

    schedules_df = parse_game_dates(schedules_df)
    dim_date_df = parse_dim_dates(dim_date_df)

    schedules_df = standardize_schedule_fields(schedules_df)
    attendance_df = standardize_attendance(attendance_df)
    dim_team_df = standardize_team_dimension(dim_team_df)
    dim_venue_df = standardize_venue_dimension(dim_venue_df)

    fact_game_df = merge_dimension_keys(
        schedule_df=schedules_df,
        dim_team_df=dim_team_df,
        dim_venue_df=dim_venue_df,
        dim_date_df=dim_date_df,
        attendance_df=attendance_df,
    )

    fact_game_df = finalize_fact_game(fact_game_df)
    validate_fact_game(fact_game_df)
    log_attendance_coverage(fact_game_df)
    write_output(fact_game_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()