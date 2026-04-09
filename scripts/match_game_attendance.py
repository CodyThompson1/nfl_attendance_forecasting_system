"""
File: match_game_attendance.py

Purpose:
Match cleaned team-week attendance data to the cleaned game schedule backbone so
each game has one home-team attendance value and diagnostic match fields.
Exclude 2020 from the matched game output because 2020 is not part of the
model-ready dataset.

Inputs:
* data/processed/schedules/clean_schedules.csv
* data/processed/attendance/team_week_attendance_clean.csv

Outputs:
* data/processed/game_attendance_matched.csv
* data/processed/game_attendance_unmatched_diagnostics.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

SCHEDULES_FILE = PROCESSED_DIR / "schedules" / "clean_schedules.csv"
ATTENDANCE_FILE = PROCESSED_DIR / "attendance" / "team_week_attendance_clean.csv"

OUTPUT_FILE = PROCESSED_DIR / "game_attendance_matched.csv"
DIAGNOSTICS_FILE = PROCESSED_DIR / "game_attendance_unmatched_diagnostics.csv"

EXCLUDED_SEASONS = {2020}

REQUIRED_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team_abbr",
]

REQUIRED_ATTENDANCE_COLUMNS = [
    "season",
    "week",
    "team_abbr_std",
    "attendance",
]


def validate_columns(df: pd.DataFrame, required_columns: list[str], file_path: Path) -> None:
    """Raise an error if any required columns are missing."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {file_path.name}: {missing_text}")


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def standardize_schedule_keys(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize schedule match key columns."""
    clean_df = schedule_df.copy()

    clean_df["season"] = pd.to_numeric(clean_df["season"], errors="coerce").astype("Int64")
    clean_df["week"] = pd.to_numeric(clean_df["week"], errors="coerce").astype("Int64")
    clean_df["home_team_abbr"] = (
        clean_df["home_team_abbr"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return clean_df


def standardize_attendance_keys(attendance_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize attendance match key columns and attendance values."""
    clean_df = attendance_df.copy()

    clean_df["season"] = pd.to_numeric(clean_df["season"], errors="coerce").astype("Int64")
    clean_df["week"] = pd.to_numeric(clean_df["week"], errors="coerce").astype("Int64")
    clean_df["team_abbr_std"] = (
        clean_df["team_abbr_std"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    clean_df["attendance"] = pd.to_numeric(clean_df["attendance"], errors="coerce").astype("Int64")

    return clean_df


def exclude_non_model_seasons(df: pd.DataFrame, season_column: str) -> pd.DataFrame:
    """Remove excluded seasons from a dataframe."""
    return df.loc[~df[season_column].isin(EXCLUDED_SEASONS)].copy()


def prepare_attendance_source(attendance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a one-row-per-season-week-team attendance source for matching.

    If duplicates exist for the same season + week + team_abbr_std:
    * keep rows with non-null attendance first
    * then keep the largest attendance value
    * then keep the first remaining row
    """
    working_df = attendance_df.copy()

    working_df["attendance_not_null"] = working_df["attendance"].notna().astype(int)
    working_df = working_df.sort_values(
        by=["season", "week", "team_abbr_std", "attendance_not_null", "attendance"],
        ascending=[True, True, True, False, False],
        na_position="last",
    )

    deduped_df = working_df.drop_duplicates(
        subset=["season", "week", "team_abbr_std"],
        keep="first",
    ).copy()

    duplicate_counts = (
        attendance_df.groupby(["season", "week", "team_abbr_std"], dropna=False)
        .size()
        .reset_index(name="attendance_source_row_count")
    )

    deduped_df = deduped_df.merge(
        duplicate_counts,
        on=["season", "week", "team_abbr_std"],
        how="left",
    )

    deduped_df = deduped_df.drop(columns=["attendance_not_null"])

    return deduped_df


def determine_unmatched_reason(row: pd.Series) -> str:
    """Assign a diagnostic reason for unmatched attendance."""
    if pd.notna(row["attendance"]):
        return ""

    if pd.isna(row["season"]) or pd.isna(row["week"]) or pd.isna(row["home_team_abbr"]):
        return "missing_schedule_key"

    if not row["season_exists_in_attendance"]:
        return "season_not_found_in_attendance"

    if not row["season_week_exists_in_attendance"]:
        return "season_week_not_found_in_attendance"

    if not row["season_week_team_exists_in_attendance"]:
        return "home_team_not_found_for_season_week"

    return "attendance_missing_after_match"


def build_match_output(
    schedules_df: pd.DataFrame,
    attendance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match home-team attendance to the game schedule backbone."""
    attendance_match_df = prepare_attendance_source(attendance_df)

    matched_df = schedules_df.merge(
        attendance_match_df[
            ["season", "week", "team_abbr_std", "attendance", "attendance_source_row_count"]
        ],
        left_on=["season", "week", "home_team_abbr"],
        right_on=["season", "week", "team_abbr_std"],
        how="left",
    )

    season_keys = set(
        attendance_df.loc[attendance_df["season"].notna(), "season"].tolist()
    )
    season_week_keys = set(
        zip(
            attendance_df.loc[
                attendance_df["season"].notna() & attendance_df["week"].notna(), "season"
            ],
            attendance_df.loc[
                attendance_df["season"].notna() & attendance_df["week"].notna(), "week"
            ],
        )
    )
    season_week_team_keys = set(
        zip(
            attendance_df.loc[
                attendance_df["season"].notna()
                & attendance_df["week"].notna()
                & attendance_df["team_abbr_std"].notna(),
                "season",
            ],
            attendance_df.loc[
                attendance_df["season"].notna()
                & attendance_df["week"].notna()
                & attendance_df["team_abbr_std"].notna(),
                "week",
            ],
            attendance_df.loc[
                attendance_df["season"].notna()
                & attendance_df["week"].notna()
                & attendance_df["team_abbr_std"].notna(),
                "team_abbr_std",
            ],
        )
    )

    matched_df["season_exists_in_attendance"] = matched_df["season"].apply(
        lambda value: value in season_keys if pd.notna(value) else False
    )
    matched_df["season_week_exists_in_attendance"] = matched_df.apply(
        lambda row: (row["season"], row["week"]) in season_week_keys
        if pd.notna(row["season"]) and pd.notna(row["week"])
        else False,
        axis=1,
    )
    matched_df["season_week_team_exists_in_attendance"] = matched_df.apply(
        lambda row: (row["season"], row["week"], row["home_team_abbr"]) in season_week_team_keys
        if pd.notna(row["season"]) and pd.notna(row["week"]) and pd.notna(row["home_team_abbr"])
        else False,
        axis=1,
    )

    matched_df["matched_flag"] = matched_df["attendance"].notna()
    matched_df["unmatched_reason"] = matched_df.apply(determine_unmatched_reason, axis=1)

    matched_df["attendance"] = matched_df["attendance"].astype("Int64")
    matched_df["attendance_source_row_count"] = matched_df["attendance_source_row_count"].astype("Int64")

    matched_df = matched_df.drop(columns=["team_abbr_std"])

    return matched_df


def write_outputs(matched_df: pd.DataFrame) -> None:
    """Write matched output and unmatched diagnostics to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    matched_df.to_csv(OUTPUT_FILE, index=False)

    unmatched_df = matched_df.loc[~matched_df["matched_flag"]].copy()
    unmatched_df.to_csv(DIAGNOSTICS_FILE, index=False)


def print_summary(matched_df: pd.DataFrame, original_schedule_count: int) -> None:
    """Print a concise run summary."""
    total_games = len(matched_df)
    matched_games = int(matched_df["matched_flag"].sum())
    unmatched_games = total_games - matched_games
    excluded_games = original_schedule_count - total_games

    print(f"Matched game attendance file written: {OUTPUT_FILE}")
    print(f"Unmatched diagnostics file written: {DIAGNOSTICS_FILE}")
    print(f"Original schedule games: {original_schedule_count}")
    print(f"Excluded games from omitted seasons: {excluded_games}")
    print(f"Total games in matched output: {total_games}")
    print(f"Matched games: {matched_games}")
    print(f"Unmatched games: {unmatched_games}")

    if unmatched_games > 0:
        reason_counts = (
            matched_df.loc[~matched_df["matched_flag"], "unmatched_reason"]
            .value_counts(dropna=False)
            .sort_index()
        )
        print("Unmatched reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")


def main() -> None:
    """Run the attendance matching pipeline step."""
    schedules_df = load_csv(SCHEDULES_FILE)
    attendance_df = load_csv(ATTENDANCE_FILE)

    validate_columns(schedules_df, REQUIRED_SCHEDULE_COLUMNS, SCHEDULES_FILE)
    validate_columns(attendance_df, REQUIRED_ATTENDANCE_COLUMNS, ATTENDANCE_FILE)

    schedules_df = standardize_schedule_keys(schedules_df)
    attendance_df = standardize_attendance_keys(attendance_df)

    original_schedule_count = len(schedules_df)

    schedules_df = exclude_non_model_seasons(schedules_df, "season")
    attendance_df = exclude_non_model_seasons(attendance_df, "season")

    matched_df = build_match_output(schedules_df, attendance_df)
    write_outputs(matched_df)
    print_summary(matched_df, original_schedule_count)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"match_game_attendance.py failed: {error}", file=sys.stderr)
        raise