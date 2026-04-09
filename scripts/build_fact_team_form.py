"""
File: build_fact_team_form.py

Purpose:
Build the fact_team_form table with two rows per game, one for the home team
and one for the away team. The script calculates rolling win percentage using
only games before the current game, games played before the current game,
prior season win percentage, and rest days.

Inputs:
- data/processed/schedules/clean_schedules.csv
- data/raw/schedules/nfl_schedules_raw.csv
- data/processed/warehouse/dim_team.csv

Outputs:
- data/processed/warehouse/fact_team_form.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_SCHEDULES_PATH = PROJECT_ROOT / "data" / "processed" / "schedules" / "clean_schedules.csv"
RAW_SCHEDULES_PATH = PROJECT_ROOT / "data" / "raw" / "schedules" / "nfl_schedules_raw.csv"
DIM_TEAM_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_team.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_team_form.csv"

EXPECTED_OUTPUT_COLUMNS = [
    "game_id",
    "team_id",
    "rolling_win_pct",
    "games_played_before_game",
    "prior_season_win_pct",
    "rest_days",
]

REQUIRED_CLEAN_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_date",
    "home_team_abbr",
    "away_team_abbr",
]

REQUIRED_RAW_SCHEDULE_COLUMNS = [
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

REQUIRED_DIM_TEAM_COLUMNS = [
    "team_id",
    "team_abbr",
]

TEAM_ABBR_CANONICAL_MAP = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "STL": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "LV": "LV",
    "OAK": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WSH",
    "WSH": "WSH",
}


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


def normalize_team_abbr(value: object) -> str | None:
    if pd.isna(value):
        return None

    cleaned_value = str(value).strip().upper()
    if not cleaned_value:
        return None

    return TEAM_ABBR_CANONICAL_MAP.get(cleaned_value, cleaned_value)


def parse_clean_schedule_dates(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.copy()
    schedule_df["game_date"] = pd.to_datetime(schedule_df["game_date"], errors="coerce")

    missing_dates = int(schedule_df["game_date"].isna().sum())
    if missing_dates > 0:
        raise ValueError(f"Unable to parse {missing_dates} game_date values in clean_schedules.csv")

    schedule_df["game_date"] = schedule_df["game_date"].dt.normalize()
    return schedule_df


def parse_raw_schedule_dates(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["gameday"] = pd.to_datetime(raw_df["gameday"], errors="coerce")

    missing_dates = int(raw_df["gameday"].isna().sum())
    if missing_dates > 0:
        raise ValueError(f"Unable to parse {missing_dates} gameday values in nfl_schedules_raw.csv")

    raw_df["gameday"] = raw_df["gameday"].dt.normalize()
    return raw_df


def standardize_clean_schedule_fields(schedule_df: pd.DataFrame) -> pd.DataFrame:
    schedule_df = schedule_df.copy()
    schedule_df["game_id"] = schedule_df["game_id"].astype(str).str.strip()
    schedule_df["season"] = pd.to_numeric(schedule_df["season"], errors="coerce").astype("Int64")
    schedule_df["week"] = pd.to_numeric(schedule_df["week"], errors="coerce").astype("Int64")
    schedule_df["home_team_abbr"] = schedule_df["home_team_abbr"].apply(normalize_team_abbr)
    schedule_df["away_team_abbr"] = schedule_df["away_team_abbr"].apply(normalize_team_abbr)
    return schedule_df


def standardize_raw_schedule_fields(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["season"] = pd.to_numeric(raw_df["season"], errors="coerce").astype("Int64")
    raw_df["week"] = pd.to_numeric(raw_df["week"], errors="coerce").astype("Int64")
    raw_df["home_team"] = raw_df["home_team"].apply(normalize_team_abbr)
    raw_df["away_team"] = raw_df["away_team"].apply(normalize_team_abbr)
    raw_df["home_score"] = pd.to_numeric(raw_df["home_score"], errors="coerce")
    raw_df["away_score"] = pd.to_numeric(raw_df["away_score"], errors="coerce")
    return raw_df


def standardize_team_dimension(dim_team_df: pd.DataFrame) -> pd.DataFrame:
    dim_team_df = dim_team_df.copy()
    dim_team_df["team_abbr"] = dim_team_df["team_abbr"].apply(normalize_team_abbr)
    dim_team_df["team_id"] = pd.to_numeric(dim_team_df["team_id"], errors="coerce").astype("Int64")
    dim_team_df = dim_team_df.dropna(subset=["team_abbr", "team_id"])
    dim_team_df = dim_team_df.drop_duplicates(subset=["team_abbr"], keep="first")
    return dim_team_df


def build_score_backbone(clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_score_df = raw_df[
        [
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    ].copy()

    raw_score_df = raw_score_df.rename(
        columns={
            "gameday": "game_date",
            "home_team": "home_team_abbr",
            "away_team": "away_team_abbr",
        }
    )

    raw_score_df = raw_score_df.drop_duplicates(
        subset=["season", "week", "game_date", "home_team_abbr", "away_team_abbr"],
        keep="first",
    )

    merged_df = clean_df.merge(
        raw_score_df,
        on=["season", "week", "game_date", "home_team_abbr", "away_team_abbr"],
        how="left",
    )

    missing_mask = merged_df["home_score"].isna() | merged_df["away_score"].isna()
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        unmatched_examples = merged_df.loc[
            missing_mask,
            ["game_id", "season", "week", "game_date", "away_team_abbr", "home_team_abbr"],
        ].head(20)
        raise ValueError(
            "Unable to match score data from nfl_schedules_raw.csv to some clean schedule games. "
            f"Example unmatched rows: {unmatched_examples.to_dict(orient='records')}"
        )

    return merged_df


def build_team_game_rows(schedule_df: pd.DataFrame) -> pd.DataFrame:
    home_df = schedule_df[
        ["game_id", "season", "game_date", "home_team_abbr", "home_score", "away_score"]
    ].copy()
    home_df = home_df.rename(
        columns={
            "home_team_abbr": "team_abbr",
            "home_score": "team_score",
            "away_score": "opponent_score",
        }
    )
    home_df["team_side"] = "HOME"

    away_df = schedule_df[
        ["game_id", "season", "game_date", "away_team_abbr", "away_score", "home_score"]
    ].copy()
    away_df = away_df.rename(
        columns={
            "away_team_abbr": "team_abbr",
            "away_score": "team_score",
            "home_score": "opponent_score",
        }
    )
    away_df["team_side"] = "AWAY"

    team_games_df = pd.concat([home_df, away_df], ignore_index=True)

    team_games_df["game_result_value"] = np.where(
        team_games_df["team_score"] > team_games_df["opponent_score"],
        1.0,
        np.where(team_games_df["team_score"] < team_games_df["opponent_score"], 0.0, 0.5),
    )

    return team_games_df


def add_team_ids(team_games_df: pd.DataFrame, dim_team_df: pd.DataFrame) -> pd.DataFrame:
    fact_df = team_games_df.merge(
        dim_team_df[["team_id", "team_abbr"]],
        on="team_abbr",
        how="left",
    )

    missing_team_ids = int(fact_df["team_id"].isna().sum())
    if missing_team_ids > 0:
        missing_teams = sorted(fact_df.loc[fact_df["team_id"].isna(), "team_abbr"].dropna().unique())
        missing_text = ", ".join(missing_teams[:20])
        raise ValueError(
            f"Unable to map {missing_team_ids} team rows to dim_team. Missing team_abbr values: {missing_text}"
        )

    return fact_df


def calculate_prior_season_win_pct(team_games_df: pd.DataFrame) -> pd.DataFrame:
    season_results_df = (
        team_games_df.groupby(["team_abbr", "season"], as_index=False)
        .agg(
            wins_equivalent=("game_result_value", "sum"),
            games_played=("game_result_value", "count"),
        )
    )

    season_results_df["season_win_pct"] = np.where(
        season_results_df["games_played"] > 0,
        season_results_df["wins_equivalent"] / season_results_df["games_played"],
        np.nan,
    )

    prior_season_lookup_df = season_results_df[["team_abbr", "season", "season_win_pct"]].copy()
    prior_season_lookup_df["season"] = prior_season_lookup_df["season"] + 1
    prior_season_lookup_df = prior_season_lookup_df.rename(
        columns={"season_win_pct": "prior_season_win_pct"}
    )

    team_games_df = team_games_df.merge(
        prior_season_lookup_df,
        on=["team_abbr", "season"],
        how="left",
    )

    return team_games_df


def calculate_team_form_metrics(team_games_df: pd.DataFrame) -> pd.DataFrame:
    team_games_df = team_games_df.sort_values(
        ["team_abbr", "season", "game_date", "game_id", "team_side"]
    ).reset_index(drop=True)

    grouped_by_team_season = team_games_df.groupby(["team_abbr", "season"], sort=False)

    team_games_df["games_played_before_game"] = grouped_by_team_season.cumcount()

    cumulative_result_sum = grouped_by_team_season["game_result_value"].transform(
        lambda series: series.cumsum().shift(1, fill_value=0)
    )

    completed_games_before = grouped_by_team_season["game_result_value"].transform(
        lambda series: series.notna().astype(int).cumsum().shift(1, fill_value=0)
    )

    team_games_df["rolling_win_pct"] = np.where(
        completed_games_before > 0,
        cumulative_result_sum / completed_games_before,
        np.nan,
    )

    grouped_by_team = team_games_df.groupby("team_abbr", sort=False)
    previous_game_date = grouped_by_team["game_date"].shift(1)

    rest_days = (team_games_df["game_date"] - previous_game_date).dt.days - 1
    team_games_df["rest_days"] = rest_days.where(previous_game_date.notna(), np.nan)

    return team_games_df


def finalize_fact_team_form(team_games_df: pd.DataFrame) -> pd.DataFrame:
    fact_df = team_games_df.copy()

    fact_df["team_id"] = pd.to_numeric(fact_df["team_id"], errors="coerce").astype("Int64")
    fact_df["games_played_before_game"] = pd.to_numeric(
        fact_df["games_played_before_game"], errors="coerce"
    ).astype("Int64")
    fact_df["rest_days"] = pd.to_numeric(fact_df["rest_days"], errors="coerce").astype("Int64")

    fact_df["rolling_win_pct"] = pd.to_numeric(fact_df["rolling_win_pct"], errors="coerce")
    fact_df["prior_season_win_pct"] = pd.to_numeric(fact_df["prior_season_win_pct"], errors="coerce")

    fact_df = fact_df[EXPECTED_OUTPUT_COLUMNS].copy()
    fact_df = fact_df.sort_values(["game_id", "team_id"]).reset_index(drop=True)

    return fact_df


def validate_fact_team_form(fact_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    duplicate_key_count = int(fact_df.duplicated(subset=["game_id", "team_id"]).sum())
    if duplicate_key_count > 0:
        raise ValueError(f"Duplicate (game_id, team_id) rows found: {duplicate_key_count}")

    expected_row_count = len(clean_df) * 2
    actual_row_count = len(fact_df)
    if actual_row_count != expected_row_count:
        raise ValueError(
            f"fact_team_form row count mismatch. Expected {expected_row_count}, found {actual_row_count}"
        )

    missing_required = []
    for column in ["game_id", "team_id", "games_played_before_game"]:
        missing_count = int(fact_df[column].isna().sum())
        if missing_count > 0:
            missing_required.append(f"{column}={missing_count}")

    if missing_required:
        raise ValueError(f"Missing required fact_team_form values: {', '.join(missing_required)}")

    negative_games_played = int((fact_df["games_played_before_game"] < 0).sum())
    if negative_games_played > 0:
        raise ValueError(f"Found {negative_games_played} negative games_played_before_game values")

    negative_rest_days = int((fact_df["rest_days"].dropna() < 0).sum())
    if negative_rest_days > 0:
        raise ValueError(f"Found {negative_rest_days} negative rest_days values")

    for column in ["rolling_win_pct", "prior_season_win_pct"]:
        invalid_count = int(
            fact_df[column].dropna().pipe(lambda series: ((series < 0) | (series > 1)).sum())
        )
        if invalid_count > 0:
            raise ValueError(f"Found {invalid_count} invalid values outside [0, 1] in {column}")

    missing_output_columns = [column for column in EXPECTED_OUTPUT_COLUMNS if column not in fact_df.columns]
    if missing_output_columns:
        raise ValueError(f"Missing expected output columns: {', '.join(missing_output_columns)}")


def write_output(fact_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact_df.to_csv(output_path, index=False)
    logging.info("fact_team_form file written: %s", output_path)


def main() -> None:
    configure_logging()

    clean_schedules_df = load_csv(CLEAN_SCHEDULES_PATH)
    raw_schedules_df = load_csv(RAW_SCHEDULES_PATH)
    dim_team_df = load_csv(DIM_TEAM_PATH)

    validate_required_columns(clean_schedules_df, REQUIRED_CLEAN_SCHEDULE_COLUMNS, "clean_schedules")
    validate_required_columns(raw_schedules_df, REQUIRED_RAW_SCHEDULE_COLUMNS, "nfl_schedules_raw")
    validate_required_columns(dim_team_df, REQUIRED_DIM_TEAM_COLUMNS, "dim_team")

    clean_schedules_df = parse_clean_schedule_dates(clean_schedules_df)
    raw_schedules_df = parse_raw_schedule_dates(raw_schedules_df)

    clean_schedules_df = standardize_clean_schedule_fields(clean_schedules_df)
    raw_schedules_df = standardize_raw_schedule_fields(raw_schedules_df)
    dim_team_df = standardize_team_dimension(dim_team_df)

    score_backbone_df = build_score_backbone(clean_schedules_df, raw_schedules_df)
    team_games_df = build_team_game_rows(score_backbone_df)
    team_games_df = add_team_ids(team_games_df, dim_team_df)
    team_games_df = calculate_prior_season_win_pct(team_games_df)
    team_games_df = calculate_team_form_metrics(team_games_df)

    fact_team_form_df = finalize_fact_team_form(team_games_df)
    validate_fact_team_form(fact_team_form_df, clean_schedules_df)
    write_output(fact_team_form_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()