"""
File: extract_nfl_schedules.py

Purpose: Extract NFL game schedule and attendance data for all teams and all games
from 2015 through 2025 using nflreadpy, backfill any missing columns from the
nflverse games CSV when needed, standardize key fields, and write the raw
game-level dataset to CSV for downstream cleaning and warehouse loading.

Inputs:

* nflreadpy schedules dataset
* nflverse games CSV backup source

Outputs:

* data/raw/nfl_schedules_raw.csv
* data/raw/nfl_schedules_extract_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd


START_SEASON = 2015
END_SEASON = 2025

BACKUP_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_OUTPUT_PATH = RAW_DIR / "nfl_schedules_raw.csv"
SUMMARY_OUTPUT_PATH = RAW_DIR / "nfl_schedules_extract_summary.csv"

BASE_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "weekday",
    "gametime",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "location",
    "result",
    "total",
    "overtime",
    "old_game_id",
    "gsis",
    "nfl_detail_id",
    "pfr",
    "pff",
    "espn",
    "ftn",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium_id",
    "stadium",
]

OPTIONAL_BACKFILL_COLUMNS = [
    "attendance",
    "game_stadium",
]

REQUIRED_OUTPUT_COLUMNS = BASE_COLUMNS + OPTIONAL_BACKFILL_COLUMNS


def ensure_output_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_schedules_data() -> pd.DataFrame:
    schedules_df = nfl.load_schedules(seasons=list(range(START_SEASON, END_SEASON + 1)))
    return schedules_df.to_pandas()


def load_backup_games_data() -> pd.DataFrame:
    return pd.read_csv(BACKUP_GAMES_URL, low_memory=False)


def validate_base_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required base columns in source data: {missing_str}")


def clean_text_column(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def add_missing_columns(df: pd.DataFrame, columns_to_add: list[str]) -> pd.DataFrame:
    output_df = df.copy()

    for column in columns_to_add:
        if column not in output_df.columns:
            output_df[column] = pd.NA

    return output_df


def backfill_optional_columns(
    schedules_df: pd.DataFrame,
    backup_df: pd.DataFrame,
) -> pd.DataFrame:
    output_df = schedules_df.copy()

    needs_backfill = any(column not in output_df.columns for column in OPTIONAL_BACKFILL_COLUMNS)
    if not needs_backfill:
        return output_df

    if "game_id" not in backup_df.columns:
        raise ValueError("Backup source is missing game_id and cannot be used for backfill.")

    available_backup_columns = [
        column for column in ["game_id"] + OPTIONAL_BACKFILL_COLUMNS
        if column in backup_df.columns
    ]

    if len(available_backup_columns) <= 1:
        output_df = add_missing_columns(output_df, OPTIONAL_BACKFILL_COLUMNS)
        return output_df

    backup_subset_df = backup_df.loc[:, available_backup_columns].copy()

    for column in OPTIONAL_BACKFILL_COLUMNS:
        if column not in backup_subset_df.columns:
            backup_subset_df[column] = pd.NA

    output_df = output_df.merge(
        backup_subset_df,
        how="left",
        on="game_id",
        suffixes=("", "_backup"),
    )

    for column in OPTIONAL_BACKFILL_COLUMNS:
        backup_column = f"{column}_backup"

        if column not in output_df.columns:
            output_df[column] = output_df[backup_column]
        else:
            output_df[column] = output_df[column].combine_first(output_df[backup_column])

        if backup_column in output_df.columns:
            output_df = output_df.drop(columns=backup_column)

    return output_df


def finalize_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df = add_missing_columns(output_df, REQUIRED_OUTPUT_COLUMNS)

    if output_df["game_stadium"].isna().all() and "stadium" in output_df.columns:
        output_df["game_stadium"] = output_df["stadium"]

    return output_df


def build_raw_schedule_dataset(
    schedules_source_df: pd.DataFrame,
    backup_source_df: pd.DataFrame,
) -> pd.DataFrame:
    validate_base_columns(schedules_source_df, BASE_COLUMNS)

    schedules_df = schedules_source_df.copy()

    schedules_df["season"] = pd.to_numeric(schedules_df["season"], errors="coerce")
    schedules_df = schedules_df[
        schedules_df["season"].between(START_SEASON, END_SEASON, inclusive="both")
    ].copy()

    schedules_df = backfill_optional_columns(
        schedules_df=schedules_df,
        backup_df=backup_source_df,
    )

    schedules_df = finalize_missing_columns(schedules_df)
    schedules_df = schedules_df.loc[:, REQUIRED_OUTPUT_COLUMNS].copy()

    schedules_df["season"] = schedules_df["season"].astype("Int64")
    schedules_df["week"] = pd.to_numeric(schedules_df["week"], errors="coerce").astype("Int64")
    schedules_df["away_score"] = pd.to_numeric(
        schedules_df["away_score"], errors="coerce"
    ).astype("Int64")
    schedules_df["home_score"] = pd.to_numeric(
        schedules_df["home_score"], errors="coerce"
    ).astype("Int64")
    schedules_df["attendance"] = pd.to_numeric(
        schedules_df["attendance"], errors="coerce"
    ).astype("Int64")

    numeric_columns = [
        "result",
        "total",
        "away_rest",
        "home_rest",
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_line",
        "under_odds",
        "over_odds",
        "temp",
        "wind",
    ]

    for column in numeric_columns:
        schedules_df[column] = pd.to_numeric(schedules_df[column], errors="coerce")

    schedules_df["gameday"] = pd.to_datetime(
        schedules_df["gameday"], errors="coerce"
    ).dt.date

    text_columns = [
        "game_id",
        "game_type",
        "weekday",
        "gametime",
        "away_team",
        "home_team",
        "location",
        "old_game_id",
        "gsis",
        "nfl_detail_id",
        "pfr",
        "pff",
        "espn",
        "ftn",
        "roof",
        "surface",
        "stadium_id",
        "game_stadium",
        "stadium",
    ]

    for column in text_columns:
        schedules_df[column] = clean_text_column(schedules_df[column])

    schedules_df["overtime"] = schedules_df["overtime"].fillna(False).astype(bool)
    schedules_df["div_game"] = schedules_df["div_game"].fillna(False).astype(bool)

    duplicate_game_ids = schedules_df["game_id"].duplicated(keep=False)
    if duplicate_game_ids.any():
        duplicate_count = int(duplicate_game_ids.sum())
        raise ValueError(f"Duplicate game_id values found in extracted data: {duplicate_count}")

    schedules_df = schedules_df.sort_values(
        by=["season", "gameday", "week", "away_team", "home_team"]
    ).reset_index(drop=True)

    return schedules_df


def build_summary_dataset(schedules_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        schedules_df.groupby(["season", "game_type"], dropna=False)
        .agg(
            games=("game_id", "nunique"),
            games_with_attendance=("attendance", lambda x: x.notna().sum()),
            games_missing_attendance=("attendance", lambda x: x.isna().sum()),
        )
        .reset_index()
        .sort_values(by=["season", "game_type"])
        .reset_index(drop=True)
    )

    return summary_df


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


def main() -> None:
    ensure_output_directory(RAW_DIR)

    schedules_source_df = load_schedules_data()
    backup_source_df = load_backup_games_data()

    schedules_df = build_raw_schedule_dataset(
        schedules_source_df=schedules_source_df,
        backup_source_df=backup_source_df,
    )
    summary_df = build_summary_dataset(schedules_df)

    write_csv(schedules_df, RAW_OUTPUT_PATH)
    write_csv(summary_df, SUMMARY_OUTPUT_PATH)

    print(f"Saved raw schedules: {RAW_OUTPUT_PATH}")
    print(f"Saved extract summary: {SUMMARY_OUTPUT_PATH}")
    print(f"Rows written: {len(schedules_df):,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"extract_nfl_schedules.py failed: {exc}", file=sys.stderr)
        raise