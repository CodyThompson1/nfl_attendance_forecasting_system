"""
File: build_ml_features_attendance.py

Purpose:
Build the final game-level feature table for NFL attendance modeling by using
fact_game as the backbone and joining fact_weather, fact_team_form, dim_date,
dim_venue, and dim_team. The output keeps one row per game, excludes season
2020 from the model-ready dataset, and writes the final feature table to CSV.

Inputs:
- data/processed/warehouse/fact_game.csv
- data/processed/warehouse/fact_weather.csv
- data/processed/warehouse/fact_team_form.csv
- data/processed/warehouse/dim_date.csv
- data/processed/warehouse/dim_venue.csv
- data/processed/warehouse/dim_team.csv

Outputs:
- data/processed/features/ml_features_attendance.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
WAREHOUSE_DIR = BASE_DIR / "data" / "processed" / "warehouse"
FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"

FACT_GAME_PATH = WAREHOUSE_DIR / "fact_game.csv"
FACT_WEATHER_PATH = WAREHOUSE_DIR / "fact_weather.csv"
FACT_TEAM_FORM_PATH = WAREHOUSE_DIR / "fact_team_form.csv"
DIM_DATE_PATH = WAREHOUSE_DIR / "dim_date.csv"
DIM_VENUE_PATH = WAREHOUSE_DIR / "dim_venue.csv"
DIM_TEAM_PATH = WAREHOUSE_DIR / "dim_team.csv"

OUTPUT_PATH = FEATURES_DIR / "ml_features_attendance.csv"

REQUIRED_OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "attendance",
    "temperature",
    "precipitation",
    "wind_speed",
    "home_team_win_pct",
    "away_team_win_pct",
    "weekend_flag",
    "holiday_flag",
    "indoor_flag",
    "holiday_before_flag",
    "holiday_after_flag",
    "holiday_adjacent_flag",
    "divisional_game_flag",
    "rivalry_flag",
    "primetime_flag",
    "week_of_season",
    "month",
    "home_rest_days",
    "away_rest_days",
    "weather_condition",
    "severe_weather_flag",
    "home_prior_season_win_pct",
    "away_prior_season_win_pct",
    "neutral_site_flag",
    "international_flag",
]

TRUE_VALUES = {"true", "t", "1", "yes", "y"}
FALSE_VALUES = {"false", "f", "0", "no", "n"}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    logging.info("Loading file: %s", file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return pd.read_csv(file_path)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def normalize_boolean_series(series: pd.Series) -> pd.Series:
    if series is None:
        return series

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": pd.NA, "none": pd.NA, "": pd.NA})
    )

    result = normalized.map(
        lambda value: True
        if value in TRUE_VALUES
        else False
        if value in FALSE_VALUES
        else pd.NA
    )

    return result.fillna(False)


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def ensure_required_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{df_name} is missing required columns: {missing_text}")


def prepare_fact_game(fact_game_df: pd.DataFrame) -> pd.DataFrame:
    fact_game_df = standardize_column_names(fact_game_df)

    required_columns = [
        "game_id",
        "season",
        "week",
        "game_type",
        "attendance",
        "home_team_id",
        "away_team_id",
        "venue_id",
        "date_id",
    ]
    ensure_required_columns(fact_game_df, required_columns, "fact_game")

    if "neutral_site_flag" not in fact_game_df.columns:
        if "is_neutral_site" in fact_game_df.columns:
            fact_game_df["neutral_site_flag"] = fact_game_df["is_neutral_site"]
        else:
            fact_game_df["neutral_site_flag"] = False

    if "international_flag" not in fact_game_df.columns:
        if "is_international" in fact_game_df.columns:
            fact_game_df["international_flag"] = fact_game_df["is_international"]
        else:
            fact_game_df["international_flag"] = False

    if "divisional_game_flag" not in fact_game_df.columns:
        fact_game_df["divisional_game_flag"] = False

    if "rivalry_flag" not in fact_game_df.columns:
        fact_game_df["rivalry_flag"] = False

    if "primetime_flag" not in fact_game_df.columns:
        fact_game_df["primetime_flag"] = False

    if "week_of_season" not in fact_game_df.columns:
        fact_game_df["week_of_season"] = fact_game_df["week"]

    fact_game_df = coerce_numeric(
        fact_game_df,
        ["season", "week", "attendance", "week_of_season"],
    )

    boolean_columns = [
        "neutral_site_flag",
        "international_flag",
        "divisional_game_flag",
        "rivalry_flag",
        "primetime_flag",
    ]
    for column in boolean_columns:
        fact_game_df[column] = normalize_boolean_series(fact_game_df[column])

    fact_game_df = fact_game_df.drop_duplicates(subset=["game_id"]).copy()

    return fact_game_df


def prepare_fact_weather(fact_weather_df: pd.DataFrame) -> pd.DataFrame:
    fact_weather_df = standardize_column_names(fact_weather_df)

    required_columns = [
        "game_id",
        "temperature",
        "precipitation",
        "wind_speed",
        "weather_condition",
        "severe_weather_flag",
    ]
    ensure_required_columns(fact_weather_df, required_columns, "fact_weather")

    fact_weather_df = coerce_numeric(
        fact_weather_df,
        ["temperature", "precipitation", "wind_speed"],
    )
    fact_weather_df["severe_weather_flag"] = normalize_boolean_series(
        fact_weather_df["severe_weather_flag"]
    )

    fact_weather_df = fact_weather_df[
        [
            "game_id",
            "temperature",
            "precipitation",
            "wind_speed",
            "weather_condition",
            "severe_weather_flag",
        ]
    ].drop_duplicates(subset=["game_id"])

    return fact_weather_df


def prepare_fact_team_form(fact_team_form_df: pd.DataFrame) -> pd.DataFrame:
    fact_team_form_df = standardize_column_names(fact_team_form_df)

    required_columns = ["game_id", "team_id"]
    ensure_required_columns(fact_team_form_df, required_columns, "fact_team_form")

    if "rolling_win_pct" in fact_team_form_df.columns and "team_win_pct" not in fact_team_form_df.columns:
        fact_team_form_df["team_win_pct"] = fact_team_form_df["rolling_win_pct"]

    if "prior_season_win_pct" not in fact_team_form_df.columns:
        fact_team_form_df["prior_season_win_pct"] = pd.NA

    if "rest_days" not in fact_team_form_df.columns:
        fact_team_form_df["rest_days"] = pd.NA

    required_form_columns = [
        "game_id",
        "team_id",
        "team_win_pct",
        "prior_season_win_pct",
        "rest_days",
    ]
    ensure_required_columns(fact_team_form_df, required_form_columns, "fact_team_form")

    fact_team_form_df = coerce_numeric(
        fact_team_form_df,
        ["team_win_pct", "prior_season_win_pct", "rest_days"],
    )

    fact_team_form_df = fact_team_form_df[
        [
            "game_id",
            "team_id",
            "team_win_pct",
            "prior_season_win_pct",
            "rest_days",
        ]
    ].drop_duplicates(subset=["game_id", "team_id"])

    return fact_team_form_df


def prepare_dim_date(dim_date_df: pd.DataFrame) -> pd.DataFrame:
    dim_date_df = standardize_column_names(dim_date_df)

    required_columns = [
        "date_id",
        "weekend_flag",
        "holiday_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
        "month",
    ]
    ensure_required_columns(dim_date_df, required_columns, "dim_date")

    boolean_columns = [
        "weekend_flag",
        "holiday_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
    ]
    for column in boolean_columns:
        dim_date_df[column] = normalize_boolean_series(dim_date_df[column])

    dim_date_df = coerce_numeric(dim_date_df, ["month"])

    dim_date_df = dim_date_df[
        [
            "date_id",
            "weekend_flag",
            "holiday_flag",
            "holiday_before_flag",
            "holiday_after_flag",
            "holiday_adjacent_flag",
            "month",
        ]
    ].drop_duplicates(subset=["date_id"])

    return dim_date_df


def prepare_dim_venue(dim_venue_df: pd.DataFrame) -> pd.DataFrame:
    dim_venue_df = standardize_column_names(dim_venue_df)

    required_columns = ["venue_id", "indoor_flag"]
    ensure_required_columns(dim_venue_df, required_columns, "dim_venue")

    dim_venue_df["indoor_flag"] = normalize_boolean_series(dim_venue_df["indoor_flag"])

    dim_venue_df = dim_venue_df[["venue_id", "indoor_flag"]].drop_duplicates(subset=["venue_id"])

    return dim_venue_df


def prepare_dim_team(dim_team_df: pd.DataFrame) -> pd.DataFrame:
    dim_team_df = standardize_column_names(dim_team_df)

    required_columns = ["team_id"]
    ensure_required_columns(dim_team_df, required_columns, "dim_team")

    if "team_abbr" not in dim_team_df.columns:
        dim_team_df["team_abbr"] = pd.NA

    if "conference" not in dim_team_df.columns:
        dim_team_df["conference"] = pd.NA

    if "division" not in dim_team_df.columns:
        dim_team_df["division"] = pd.NA

    dim_team_df = dim_team_df[
        ["team_id", "team_abbr", "conference", "division"]
    ].drop_duplicates(subset=["team_id"])

    return dim_team_df


def build_home_away_team_form_features(
    base_df: pd.DataFrame,
    fact_team_form_df: pd.DataFrame,
) -> pd.DataFrame:
    home_form_df = fact_team_form_df.rename(
        columns={
            "team_id": "home_team_id",
            "team_win_pct": "home_team_win_pct",
            "prior_season_win_pct": "home_prior_season_win_pct",
            "rest_days": "home_rest_days",
        }
    )

    away_form_df = fact_team_form_df.rename(
        columns={
            "team_id": "away_team_id",
            "team_win_pct": "away_team_win_pct",
            "prior_season_win_pct": "away_prior_season_win_pct",
            "rest_days": "away_rest_days",
        }
    )

    merged_df = base_df.merge(
        home_form_df[
            [
                "game_id",
                "home_team_id",
                "home_team_win_pct",
                "home_prior_season_win_pct",
                "home_rest_days",
            ]
        ],
        how="left",
        on=["game_id", "home_team_id"],
    )

    merged_df = merged_df.merge(
        away_form_df[
            [
                "game_id",
                "away_team_id",
                "away_team_win_pct",
                "away_prior_season_win_pct",
                "away_rest_days",
            ]
        ],
        how="left",
        on=["game_id", "away_team_id"],
    )

    return merged_df


def add_divisional_game_flag_from_team_dim(
    df: pd.DataFrame,
    dim_team_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    home_team_lookup = dim_team_df.rename(
        columns={
            "team_id": "home_team_id",
            "team_abbr": "home_team_abbr",
            "conference": "home_conference",
            "division": "home_division",
        }
    )

    away_team_lookup = dim_team_df.rename(
        columns={
            "team_id": "away_team_id",
            "team_abbr": "away_team_abbr",
            "conference": "away_conference",
            "division": "away_division",
        }
    )

    df = df.merge(home_team_lookup, how="left", on="home_team_id")
    df = df.merge(away_team_lookup, how="left", on="away_team_id")

    missing_or_blank_flag = (
        df["divisional_game_flag"].isna()
        | (df["divisional_game_flag"] == False)
    )

    derived_divisional_flag = (
        df["home_conference"].notna()
        & df["away_conference"].notna()
        & df["home_division"].notna()
        & df["away_division"].notna()
        & (df["home_conference"].astype(str).str.strip() == df["away_conference"].astype(str).str.strip())
        & (df["home_division"].astype(str).str.strip() == df["away_division"].astype(str).str.strip())
    )

    df.loc[missing_or_blank_flag, "divisional_game_flag"] = derived_divisional_flag.loc[
        missing_or_blank_flag
    ]

    return df


def apply_indoor_weather_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    indoor_mask = df["indoor_flag"] == True

    weather_numeric_columns = ["temperature", "precipitation", "wind_speed"]
    for column in weather_numeric_columns:
        if column in df.columns:
            df.loc[indoor_mask, column] = pd.NA

    if "weather_condition" in df.columns:
        df.loc[indoor_mask, "weather_condition"] = pd.NA

    if "severe_weather_flag" in df.columns:
        df.loc[indoor_mask, "severe_weather_flag"] = False

    return df


def validate_one_row_per_game(df: pd.DataFrame) -> None:
    duplicate_count = df["game_id"].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Final feature table has {duplicate_count} duplicate game_id values.")


def validate_output_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_OUTPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Final feature table is missing required output columns: {missing_text}")


def build_ml_features_attendance() -> pd.DataFrame:
    fact_game_df = prepare_fact_game(load_csv(FACT_GAME_PATH))
    fact_weather_df = prepare_fact_weather(load_csv(FACT_WEATHER_PATH))
    fact_team_form_df = prepare_fact_team_form(load_csv(FACT_TEAM_FORM_PATH))
    dim_date_df = prepare_dim_date(load_csv(DIM_DATE_PATH))
    dim_venue_df = prepare_dim_venue(load_csv(DIM_VENUE_PATH))
    dim_team_df = prepare_dim_team(load_csv(DIM_TEAM_PATH))

    model_df = fact_game_df.copy()

    model_df = model_df.merge(
        fact_weather_df,
        how="left",
        on="game_id",
    )

    model_df = build_home_away_team_form_features(model_df, fact_team_form_df)

    model_df = model_df.merge(
        dim_date_df,
        how="left",
        on="date_id",
    )

    model_df = model_df.merge(
        dim_venue_df,
        how="left",
        on="venue_id",
    )

    model_df = add_divisional_game_flag_from_team_dim(model_df, dim_team_df)

    model_df = apply_indoor_weather_logic(model_df)

    model_df = model_df[model_df["season"] != 2020].copy()

    numeric_columns = [
        "season",
        "week",
        "attendance",
        "temperature",
        "precipitation",
        "wind_speed",
        "home_team_win_pct",
        "away_team_win_pct",
        "week_of_season",
        "month",
        "home_rest_days",
        "away_rest_days",
        "home_prior_season_win_pct",
        "away_prior_season_win_pct",
    ]
    model_df = coerce_numeric(model_df, numeric_columns)

    boolean_columns = [
        "weekend_flag",
        "holiday_flag",
        "indoor_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
        "divisional_game_flag",
        "rivalry_flag",
        "primetime_flag",
        "severe_weather_flag",
        "neutral_site_flag",
        "international_flag",
    ]
    for column in boolean_columns:
        model_df[column] = normalize_boolean_series(model_df[column])

    output_df = model_df[REQUIRED_OUTPUT_COLUMNS].copy()

    validate_one_row_per_game(output_df)
    validate_output_columns(output_df)

    return output_df


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Model feature table written: %s", output_path)


def main() -> None:
    configure_logging()
    ml_features_df = build_ml_features_attendance()
    save_output(ml_features_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()