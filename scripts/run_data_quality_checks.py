"""
File: run_data_quality_checks.py

Purpose:
Run data quality checks across final warehouse and feature outputs for the
NFL attendance forecasting pipeline and save a QC summary CSV.

Inputs:
- data/processed/warehouse/fact_game.csv
- data/processed/warehouse/fact_weather.csv
- data/processed/warehouse/fact_team_form.csv
- data/processed/warehouse/dim_team.csv
- data/processed/warehouse/dim_venue.csv
- data/processed/warehouse/dim_date.csv
- data/processed/features/ml_features_attendance.csv

Outputs:
- data/processed/qc/data_quality_summary.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACT_GAME_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_game.csv"
FACT_WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_weather.csv"
FACT_TEAM_FORM_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_team_form.csv"
DIM_TEAM_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_team.csv"
DIM_VENUE_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_venue.csv"
DIM_DATE_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "dim_date.csv"
ML_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "features" / "ml_features_attendance.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "qc" / "data_quality_summary.csv"
)

CORE_MODEL_FIELDS = [
    "game_id",
    "season",
    "attendance",
    "home_team_id",
    "away_team_id",
    "venue_id",
    "date_id",
    "home_win_pct_last_5",
    "away_win_pct_last_5",
    "weekend_flag",
    "holiday_adjacent_flag",
]

MAX_ALLOWED_MISSING_PCT = 5.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    logging.info("Loading file: %s", file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    return pd.read_csv(file_path)


def ensure_output_directory(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def coerce_datetime(df: pd.DataFrame, column_name: str) -> pd.Series:
    return pd.to_datetime(df[column_name], errors="coerce")


def add_result(
    results: list[dict[str, object]],
    check_name: str,
    passed: bool,
    detail: str,
    metric_value: object | None = None,
    threshold: object | None = None,
) -> None:
    results.append(
        {
            "check_name": check_name,
            "passed": passed,
            "detail": detail,
            "metric_value": metric_value,
            "threshold": threshold,
        }
    )


def check_attendance_positive(fact_game_df: pd.DataFrame) -> dict[str, object]:
    if "attendance" not in fact_game_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: attendance",
            "metric_value": None,
            "threshold": "> 0 for all rows",
        }

    attendance_numeric = pd.to_numeric(fact_game_df["attendance"], errors="coerce")
    invalid_count = int((attendance_numeric <= 0).fillna(True).sum())

    passed = invalid_count == 0
    detail = (
        "All fact_game attendance values are greater than 0."
        if passed
        else f"{invalid_count} fact_game rows have attendance <= 0 or missing."
    )

    return {
        "passed": passed,
        "detail": detail,
        "metric_value": invalid_count,
        "threshold": 0,
    }


def check_no_duplicate_game_ids(
    df: pd.DataFrame,
    dataset_name: str,
) -> dict[str, object]:
    if "game_id" not in df.columns:
        return {
            "passed": False,
            "detail": f"Missing required column: game_id in {dataset_name}",
            "metric_value": None,
            "threshold": 0,
        }

    duplicate_count = int(df["game_id"].duplicated().sum())
    passed = duplicate_count == 0
    detail = (
        f"No duplicate game_id values found in {dataset_name}."
        if passed
        else f"{dataset_name} contains {duplicate_count} duplicate game_id values."
    )

    return {
        "passed": passed,
        "detail": detail,
        "metric_value": duplicate_count,
        "threshold": 0,
    }


def check_one_row_per_game(fact_game_df: pd.DataFrame) -> dict[str, object]:
    if "game_id" not in fact_game_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: game_id in fact_game",
            "metric_value": None,
            "threshold": "row_count == unique_game_id_count",
        }

    row_count = int(len(fact_game_df))
    unique_game_count = int(fact_game_df["game_id"].nunique())
    passed = row_count == unique_game_count

    detail = (
        "fact_game has exactly one row per game."
        if passed
        else (
            f"fact_game row count ({row_count}) does not match unique game_id count "
            f"({unique_game_count})."
        )
    )

    return {
        "passed": passed,
        "detail": detail,
        "metric_value": row_count - unique_game_count,
        "threshold": 0,
    }


def check_weather_date_alignment(
    fact_game_df: pd.DataFrame,
    fact_weather_df: pd.DataFrame,
    dim_venue_df: pd.DataFrame,
) -> dict[str, object]:
    if "game_id" not in fact_weather_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: game_id in fact_weather",
            "metric_value": None,
            "threshold": "0 rows with mismatched game_date and weather_date",
        }

    if "game_id" not in fact_game_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: game_id in fact_game",
            "metric_value": None,
            "threshold": "0 rows with mismatched game_date and weather_date",
        }

    if "game_date" not in fact_game_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: game_date in fact_game",
            "metric_value": None,
            "threshold": "0 rows with mismatched game_date and weather_date",
        }

    weather_date_column = find_first_existing_column(
        fact_weather_df,
        ["weather_date", "weather_timestamp", "observation_date", "date"],
    )
    if weather_date_column is None:
        return {
            "passed": False,
            "detail": "Missing weather date column in fact_weather",
            "metric_value": None,
            "threshold": "0 rows with mismatched game_date and weather_date",
        }

    game_subset_columns = ["game_id", "game_date"]
    if "venue_id" in fact_game_df.columns:
        game_subset_columns.append("venue_id")

    game_subset = fact_game_df[game_subset_columns].copy()
    game_subset["game_date_parsed"] = pd.to_datetime(
        game_subset["game_date"], errors="coerce"
    ).dt.date

    weather_subset = fact_weather_df[["game_id", weather_date_column]].copy()
    weather_subset["weather_date_parsed"] = pd.to_datetime(
        weather_subset[weather_date_column], errors="coerce"
    ).dt.date

    merged_df = game_subset.merge(weather_subset, on="game_id", how="left")

    venue_subset = pd.DataFrame()
    if "venue_id" in dim_venue_df.columns:
        venue_subset = dim_venue_df[["venue_id"]].copy()
        indoor_column = find_first_existing_column(
            dim_venue_df,
            ["indoor_flag", "is_indoor", "indoors_flag"],
        )
        if indoor_column is not None:
            venue_subset["indoor_flag"] = dim_venue_df[indoor_column]
        else:
            venue_subset["indoor_flag"] = pd.NA

    if not venue_subset.empty and "venue_id" in merged_df.columns:
        merged_df = merged_df.merge(venue_subset, on="venue_id", how="left")
    else:
        merged_df["indoor_flag"] = pd.NA

    merged_df["indoor_flag"] = (
        merged_df["indoor_flag"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    applicable_df = merged_df[
        merged_df["weather_date_parsed"].notna()
        & merged_df["game_date_parsed"].notna()
        & (merged_df["indoor_flag"] != True)
    ].copy()

    if applicable_df.empty:
        return {
            "passed": True,
            "detail": "No applicable outdoor rows with both game_date and weather_date available.",
            "metric_value": 0,
            "threshold": "0 rows with mismatched game_date and weather_date",
        }

    invalid_count = int(
        (applicable_df["game_date_parsed"] != applicable_df["weather_date_parsed"]).sum()
    )
    passed = invalid_count == 0

    detail = (
        "All applicable weather rows align to the same calendar date as the game."
        if passed
        else f"{invalid_count} applicable rows have weather dates that do not match the game date."
    )

    return {
        "passed": passed,
        "detail": detail,
        "metric_value": invalid_count,
        "threshold": 0,
    }


def check_core_field_missingness(
    ml_features_df: pd.DataFrame,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    working_df = ml_features_df.copy()
    if "season" in working_df.columns:
        working_df = working_df[working_df["season"].astype(str) != "2020"].copy()

    for field_name in CORE_MODEL_FIELDS:
        if field_name not in working_df.columns:
            results.append(
                {
                    "check_name": f"missingness_under_5pct_{field_name}",
                    "passed": False,
                    "detail": f"Missing required core model field: {field_name}",
                    "metric_value": None,
                    "threshold": f"< {MAX_ALLOWED_MISSING_PCT}%",
                }
            )
            continue

        missing_pct = round(float(working_df[field_name].isna().mean() * 100), 4)
        passed = missing_pct < MAX_ALLOWED_MISSING_PCT
        detail = (
            f"{field_name} missingness is {missing_pct:.4f}%."
            if passed
            else (
                f"{field_name} missingness is {missing_pct:.4f}%, which exceeds "
                f"{MAX_ALLOWED_MISSING_PCT}%."
            )
        )

        results.append(
            {
                "check_name": f"missingness_under_5pct_{field_name}",
                "passed": passed,
                "detail": detail,
                "metric_value": missing_pct,
                "threshold": f"< {MAX_ALLOWED_MISSING_PCT}%",
            }
        )

    return results


def check_valid_dimension_joins(
    fact_game_df: pd.DataFrame,
    dim_team_df: pd.DataFrame,
    dim_venue_df: pd.DataFrame,
    dim_date_df: pd.DataFrame,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    join_specs = [
        ("home_team_id", dim_team_df, "team_id"),
        ("away_team_id", dim_team_df, "team_id"),
        ("venue_id", dim_venue_df, "venue_id"),
        ("date_id", dim_date_df, "date_id"),
    ]

    for fact_key, dim_df, dim_key in join_specs:
        check_name = f"valid_join_{fact_key}_to_{dim_key}"

        if fact_key not in fact_game_df.columns:
            results.append(
                {
                    "check_name": check_name,
                    "passed": False,
                    "detail": f"Missing fact_game join key: {fact_key}",
                    "metric_value": None,
                    "threshold": 0,
                }
            )
            continue

        if dim_key not in dim_df.columns:
            results.append(
                {
                    "check_name": check_name,
                    "passed": False,
                    "detail": f"Missing dimension join key: {dim_key}",
                    "metric_value": None,
                    "threshold": 0,
                }
            )
            continue

        unmatched_count = int(
            (~fact_game_df[fact_key].isin(dim_df[dim_key].dropna())).sum()
        )
        passed = unmatched_count == 0
        detail = (
            f"All {fact_key} values in fact_game successfully join to {dim_key}."
            if passed
            else (
                f"{unmatched_count} fact_game rows have {fact_key} values that do not "
                f"match {dim_key}."
            )
        )

        results.append(
            {
                "check_name": check_name,
                "passed": passed,
                "detail": detail,
                "metric_value": unmatched_count,
                "threshold": 0,
            }
        )

    return results


def check_row_count_consistency(
    fact_game_df: pd.DataFrame,
    ml_features_df: pd.DataFrame,
) -> dict[str, object]:
    fact_df = fact_game_df.copy()
    features_df = ml_features_df.copy()

    if "season" not in fact_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: season in fact_game",
            "metric_value": None,
            "threshold": "equal row counts after excluding 2020",
        }

    if "season" not in features_df.columns:
        return {
            "passed": False,
            "detail": "Missing required column: season in ml_features_attendance",
            "metric_value": None,
            "threshold": "equal row counts after excluding 2020",
        }

    fact_df = fact_df[fact_df["season"].astype(str) != "2020"].copy()
    features_df = features_df[features_df["season"].astype(str) != "2020"].copy()

    fact_count = int(len(fact_df))
    features_count = int(len(features_df))
    count_difference = fact_count - features_count

    passed = fact_count == features_count
    detail = (
        "fact_game and ml_features_attendance have consistent row counts after excluding 2020."
        if passed
        else (
            f"Row count mismatch after excluding 2020: fact_game={fact_count}, "
            f"ml_features_attendance={features_count}."
        )
    )

    return {
        "passed": passed,
        "detail": detail,
        "metric_value": count_difference,
        "threshold": 0,
    }


def check_fact_team_form_duplicates(fact_team_form_df: pd.DataFrame) -> dict[str, object]:
    required_columns = ["game_id"]
    missing_columns = [column for column in required_columns if column not in fact_team_form_df.columns]

    if missing_columns:
        return {
            "passed": False,
            "detail": f"Missing required columns in fact_team_form: {', '.join(missing_columns)}",
            "metric_value": None,
            "threshold": 0,
        }

    if "team_id" in fact_team_form_df.columns:
        duplicate_count = int(
            fact_team_form_df.duplicated(subset=["game_id", "team_id"]).sum()
        )
        threshold_text = "0 duplicate (game_id, team_id) rows"
        detail = (
            "No duplicate (game_id, team_id) rows found in fact_team_form."
            if duplicate_count == 0
            else f"fact_team_form contains {duplicate_count} duplicate (game_id, team_id) rows."
        )
    else:
        duplicate_count = int(fact_team_form_df["game_id"].duplicated().sum())
        threshold_text = "0 duplicate game_id rows"
        detail = (
            "No duplicate game_id rows found in fact_team_form."
            if duplicate_count == 0
            else f"fact_team_form contains {duplicate_count} duplicate game_id rows."
        )

    return {
        "passed": duplicate_count == 0,
        "detail": detail,
        "metric_value": duplicate_count,
        "threshold": threshold_text,
    }


def run_quality_checks() -> pd.DataFrame:
    fact_game_df = load_csv(FACT_GAME_PATH)
    fact_weather_df = load_csv(FACT_WEATHER_PATH)
    fact_team_form_df = load_csv(FACT_TEAM_FORM_PATH)
    dim_team_df = load_csv(DIM_TEAM_PATH)
    dim_venue_df = load_csv(DIM_VENUE_PATH)
    dim_date_df = load_csv(DIM_DATE_PATH)
    ml_features_df = load_csv(ML_FEATURES_PATH)

    results: list[dict[str, object]] = []

    attendance_result = check_attendance_positive(fact_game_df)
    add_result(
        results,
        "attendance_greater_than_zero",
        attendance_result["passed"],
        attendance_result["detail"],
        attendance_result["metric_value"],
        attendance_result["threshold"],
    )

    duplicate_fact_game_result = check_no_duplicate_game_ids(fact_game_df, "fact_game")
    add_result(
        results,
        "no_duplicate_game_ids_fact_game",
        duplicate_fact_game_result["passed"],
        duplicate_fact_game_result["detail"],
        duplicate_fact_game_result["metric_value"],
        duplicate_fact_game_result["threshold"],
    )

    one_row_result = check_one_row_per_game(fact_game_df)
    add_result(
        results,
        "one_row_per_game_fact_game",
        one_row_result["passed"],
        one_row_result["detail"],
        one_row_result["metric_value"],
        one_row_result["threshold"],
    )

    weather_result = check_weather_date_alignment(
        fact_game_df,
        fact_weather_df,
        dim_venue_df,
    )
    add_result(
        results,
        "weather_date_matches_game_date_for_applicable_games",
        weather_result["passed"],
        weather_result["detail"],
        weather_result["metric_value"],
        weather_result["threshold"],
    )

    for result in check_core_field_missingness(ml_features_df):
        results.append(result)

    for result in check_valid_dimension_joins(
        fact_game_df,
        dim_team_df,
        dim_venue_df,
        dim_date_df,
    ):
        results.append(result)

    row_count_result = check_row_count_consistency(fact_game_df, ml_features_df)
    add_result(
        results,
        "row_count_consistency_fact_game_vs_ml_features_excluding_2020",
        row_count_result["passed"],
        row_count_result["detail"],
        row_count_result["metric_value"],
        row_count_result["threshold"],
    )

    fact_team_form_result = check_fact_team_form_duplicates(fact_team_form_df)
    add_result(
        results,
        "no_duplicate_rows_fact_team_form",
        fact_team_form_result["passed"],
        fact_team_form_result["detail"],
        fact_team_form_result["metric_value"],
        fact_team_form_result["threshold"],
    )

    return pd.DataFrame(results)


def main() -> None:
    configure_logging()
    summary_df = run_quality_checks()
    ensure_output_directory(OUTPUT_PATH)
    summary_df.to_csv(OUTPUT_PATH, index=False)
    logging.info("Data quality summary written: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()