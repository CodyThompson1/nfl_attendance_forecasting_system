"""
File: build_dim_date.py

Purpose:
Build a reusable date dimension from game dates in clean_schedules.csv. The script
parses game dates carefully, creates one row per unique calendar date, and adds
calendar and holiday-related fields for downstream warehouse and modeling use.

Inputs:
- data/processed/schedules/clean_schedules.csv

Outputs:
- data/processed/warehouse/dim_date.csv
"""

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "schedules" / "clean_schedules.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "warehouse" / "dim_date.csv"

REQUIRED_COLUMNS = ["season"]
DATE_COLUMN_CANDIDATES = ["game_date", "gameday", "date"]


def load_clean_schedules(file_path: Path) -> pd.DataFrame:
    """Load the clean schedules file and validate base requirements."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    schedules_df = pd.read_csv(file_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in schedules_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in clean_schedules.csv: {missing_text}")

    return schedules_df


def get_date_column(schedules_df: pd.DataFrame) -> str:
    """Return the first available date column."""
    for column in DATE_COLUMN_CANDIDATES:
        if column in schedules_df.columns:
            return column

    candidate_text = ", ".join(DATE_COLUMN_CANDIDATES)
    raise ValueError(f"No date column found. Expected one of: {candidate_text}")


def parse_game_dates(date_series: pd.Series) -> pd.Series:
    """Parse game dates with month/day/year handling and fallback parsing."""
    parsed_dates = pd.to_datetime(date_series, format="%m/%d/%Y", errors="coerce")
    fallback_mask = parsed_dates.isna()

    if fallback_mask.any():
        parsed_dates.loc[fallback_mask] = pd.to_datetime(
            date_series.loc[fallback_mask],
            errors="coerce",
        )

    return parsed_dates.dt.normalize()


def build_holiday_frame(min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """Build a holiday lookup frame across the needed date range."""
    calendar = USFederalHolidayCalendar()
    holiday_dates = calendar.holidays(
        start=min_date - pd.Timedelta(days=2),
        end=max_date + pd.Timedelta(days=2),
    )

    holiday_df = pd.DataFrame({"date": pd.to_datetime(holiday_dates).normalize()})
    holiday_df["holiday_flag"] = True

    return holiday_df.drop_duplicates(subset=["date"]).reset_index(drop=True)


def build_dim_date(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Build the date dimension from unique game dates."""
    schedules_df = schedules_df.copy()
    schedules_df["season"] = pd.to_numeric(schedules_df["season"], errors="coerce")
    schedules_df = schedules_df[schedules_df["season"].between(2015, 2025, inclusive="both")].copy()

    date_column = get_date_column(schedules_df)
    schedules_df["date"] = parse_game_dates(schedules_df[date_column])

    invalid_dates_df = schedules_df[schedules_df["date"].isna()]
    if not invalid_dates_df.empty:
        invalid_values = invalid_dates_df[date_column].dropna().astype(str).unique().tolist()
        raise ValueError(f"Unable to parse one or more game dates: {invalid_values[:10]}")

    dim_date_df = pd.DataFrame({"date": schedules_df["date"].drop_duplicates().sort_values()})
    dim_date_df = dim_date_df.reset_index(drop=True)

    min_date = dim_date_df["date"].min()
    max_date = dim_date_df["date"].max()
    holiday_df = build_holiday_frame(min_date=min_date, max_date=max_date)

    dim_date_df = dim_date_df.merge(holiday_df, on="date", how="left")
    dim_date_df["holiday_flag"] = dim_date_df["holiday_flag"].fillna(False)

    holiday_dates = set(dim_date_df.loc[dim_date_df["holiday_flag"], "date"])

    dim_date_df["holiday_before_flag"] = dim_date_df["date"].apply(
        lambda value: (value + pd.Timedelta(days=1)) in holiday_dates
    )
    dim_date_df["holiday_after_flag"] = dim_date_df["date"].apply(
        lambda value: (value - pd.Timedelta(days=1)) in holiday_dates
    )
    dim_date_df["holiday_adjacent_flag"] = (
        dim_date_df["holiday_before_flag"] | dim_date_df["holiday_after_flag"]
    )

    dim_date_df["day_of_week"] = dim_date_df["date"].dt.day_name()
    dim_date_df["weekend_flag"] = dim_date_df["day_of_week"].isin(["Saturday", "Sunday"])
    dim_date_df["month"] = dim_date_df["date"].dt.month
    dim_date_df["quarter"] = dim_date_df["date"].dt.quarter
    dim_date_df["year"] = dim_date_df["date"].dt.year
    dim_date_df["date_id"] = dim_date_df["date"].dt.strftime("%Y%m%d").astype(int)

    dim_date_df = dim_date_df[
        [
            "date_id",
            "date",
            "day_of_week",
            "weekend_flag",
            "holiday_flag",
            "holiday_before_flag",
            "holiday_after_flag",
            "holiday_adjacent_flag",
            "month",
            "quarter",
            "year",
        ]
    ].copy()

    dim_date_df = dim_date_df.sort_values("date").reset_index(drop=True)

    return dim_date_df


def validate_dim_date(dim_date_df: pd.DataFrame) -> None:
    """Validate the final dim_date output."""
    if dim_date_df.empty:
        raise ValueError("dim_date output is empty.")

    required_output_columns = [
        "date_id",
        "date",
        "day_of_week",
        "weekend_flag",
        "holiday_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
        "month",
        "quarter",
        "year",
    ]

    missing_columns = [column for column in required_output_columns if column not in dim_date_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required output columns: {missing_text}")

    if dim_date_df["date"].isna().any():
        raise ValueError("Null date values found in dim_date.")

    if dim_date_df["date"].duplicated().any():
        duplicate_dates = dim_date_df.loc[dim_date_df["date"].duplicated(), "date"].astype(str).tolist()
        raise ValueError(f"Duplicate dates found in dim_date: {duplicate_dates}")

    if dim_date_df["date_id"].duplicated().any():
        raise ValueError("Duplicate date_id values found in dim_date.")


def write_dim_date(dim_date_df: pd.DataFrame, output_path: Path) -> None:
    """Write dim_date to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dim_date_df.to_csv(output_path, index=False)


def main() -> None:
    """Run the dim_date build process."""
    schedules_df = load_clean_schedules(INPUT_PATH)
    dim_date_df = build_dim_date(schedules_df)
    validate_dim_date(dim_date_df)
    write_dim_date(dim_date_df, OUTPUT_PATH)

    print(f"dim_date file written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()