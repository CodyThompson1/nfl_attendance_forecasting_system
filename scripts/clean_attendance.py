"""
File: clean_attendance.py

Purpose:
Clean and standardize Sports Reference attendance data that is already in
team-week format. This version uses the shared team alias reference so the
attendance output aligns with schedule-based team abbreviations and names.

Inputs:
* data/raw/attendance/sportsref_attendance_raw.csv
* data/processed/schedules/team_alias_reference.csv

Outputs:
* data/processed/attendance/team_week_attendance_clean.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "attendance" / "sportsref_attendance_raw.csv"
)
TEAM_ALIAS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "schedules" / "team_alias_reference.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "attendance" / "team_week_attendance_clean.csv"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    logging.info("Loading file: %s", file_path)
    return pd.read_csv(file_path)


def standardize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_alias_reference(alias_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "raw_team_value",
        "team_abbr_standard",
        "team_name_standard",
    ]
    missing_columns = [col for col in required_columns if col not in alias_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required team alias columns: {', '.join(missing_columns)}"
        )

    clean_df = alias_df.copy()

    clean_df["raw_team_value"] = clean_df["raw_team_value"].apply(standardize_text)
    clean_df["team_abbr_standard"] = clean_df["team_abbr_standard"].apply(standardize_text)
    clean_df["team_name_standard"] = clean_df["team_name_standard"].apply(standardize_text)

    clean_df = clean_df.loc[clean_df["raw_team_value"] != ""].copy()
    clean_df = clean_df.drop_duplicates(subset=["raw_team_value"], keep="first")
    clean_df = clean_df.reset_index(drop=True)

    return clean_df


def build_team_alias_lookup(alias_df: pd.DataFrame) -> pd.DataFrame:
    alias_base = clean_alias_reference(alias_df)

    base_rows = alias_base[
        ["raw_team_value", "team_abbr_standard", "team_name_standard"]
    ].copy()

    name_rows = alias_base[
        ["team_name_standard", "team_abbr_standard", "team_name_standard"]
    ].copy()
    name_rows.columns = ["raw_team_value", "team_abbr_standard", "team_name_standard"]

    abbr_rows = alias_base[
        ["team_abbr_standard", "team_abbr_standard", "team_name_standard"]
    ].copy()
    abbr_rows.columns = ["raw_team_value", "team_abbr_standard", "team_name_standard"]

    lookup_df = pd.concat(
        [base_rows, name_rows, abbr_rows],
        ignore_index=True,
    )

    lookup_df["raw_team_value"] = lookup_df["raw_team_value"].apply(standardize_text)
    lookup_df["team_abbr_standard"] = lookup_df["team_abbr_standard"].apply(standardize_text)
    lookup_df["team_name_standard"] = lookup_df["team_name_standard"].apply(standardize_text)

    lookup_df = lookup_df.loc[lookup_df["raw_team_value"] != ""].copy()
    lookup_df = lookup_df.drop_duplicates(subset=["raw_team_value"], keep="first")
    lookup_df = lookup_df.reset_index(drop=True)

    return lookup_df


def clean_attendance(
    attendance_df: pd.DataFrame,
    alias_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "season",
        "week",
        "home_team_name",
        "home_team_abbr",
        "attendance",
    ]
    missing_columns = [col for col in required_columns if col not in attendance_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required attendance columns: {', '.join(missing_columns)}"
        )

    clean_df = attendance_df.copy()

    clean_df["season"] = pd.to_numeric(clean_df["season"], errors="coerce").astype("Int64")
    clean_df["week"] = pd.to_numeric(clean_df["week"], errors="coerce").astype("Int64")
    clean_df["attendance"] = pd.to_numeric(clean_df["attendance"], errors="coerce")

    large_value_mask = clean_df["attendance"] > 200000
    clean_df.loc[large_value_mask, "attendance"] = (
        clean_df.loc[large_value_mask, "attendance"] / 10
    )

    clean_df["attendance"] = clean_df["attendance"].round().astype("Int64")

    clean_df["team_name_raw"] = clean_df["home_team_name"].apply(standardize_text)
    clean_df["team_abbr_raw"] = clean_df["home_team_abbr"].apply(standardize_text)

    abbr_lookup_df = alias_lookup_df.rename(
        columns={
            "raw_team_value": "team_abbr_raw",
            "team_abbr_standard": "team_abbr_from_abbr_lookup",
            "team_name_standard": "team_name_from_abbr_lookup",
        }
    )

    name_lookup_df = alias_lookup_df.rename(
        columns={
            "raw_team_value": "team_name_raw",
            "team_abbr_standard": "team_abbr_from_name_lookup",
            "team_name_standard": "team_name_from_name_lookup",
        }
    )

    clean_df = clean_df.merge(
        abbr_lookup_df,
        how="left",
        on="team_abbr_raw",
    )

    clean_df = clean_df.merge(
        name_lookup_df,
        how="left",
        on="team_name_raw",
    )

    clean_df["team_abbr_std"] = clean_df["team_abbr_from_abbr_lookup"]
    clean_df["team_abbr_std"] = clean_df["team_abbr_std"].fillna(
        clean_df["team_abbr_from_name_lookup"]
    )
    clean_df["team_abbr_std"] = clean_df["team_abbr_std"].fillna(clean_df["team_abbr_raw"])

    clean_df["team_name_std"] = clean_df["team_name_from_abbr_lookup"]
    clean_df["team_name_std"] = clean_df["team_name_std"].fillna(
        clean_df["team_name_from_name_lookup"]
    )
    clean_df["team_name_std"] = clean_df["team_name_std"].fillna(clean_df["team_name_raw"])

    output_columns = [
        "season",
        "week",
        "team_name_raw",
        "team_abbr_raw",
        "team_name_std",
        "team_abbr_std",
        "attendance",
    ]

    clean_df = clean_df[output_columns].copy()
    clean_df = clean_df.sort_values(
        ["season", "team_abbr_std", "week"]
    ).reset_index(drop=True)

    return clean_df


def validate(clean_df: pd.DataFrame, alias_base_df: pd.DataFrame) -> None:
    if clean_df["season"].isna().any():
        raise ValueError("Missing season values found.")

    if clean_df["week"].isna().any():
        raise ValueError("Missing week values found.")

    if clean_df["attendance"].isna().any():
        raise ValueError("Missing attendance values found.")

    if clean_df.duplicated(["season", "team_abbr_std", "week"]).any():
        duplicates = clean_df.loc[
            clean_df.duplicated(["season", "team_abbr_std", "week"], keep=False),
            ["season", "week", "team_abbr_std"],
        ].drop_duplicates()
        raise ValueError(f"Duplicate team-week rows found:\n{duplicates.to_string(index=False)}")

    valid_abbrs = set(alias_base_df["team_abbr_standard"].dropna().astype(str))
    invalid_abbrs = sorted(
        set(clean_df["team_abbr_std"].dropna().astype(str)) - valid_abbrs
    )
    if invalid_abbrs:
        raise ValueError(f"Unmapped standardized team abbreviations found: {invalid_abbrs}")


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved cleaned attendance file to %s", output_path)


def main() -> None:
    configure_logging()

    attendance_df = load_csv(RAW_INPUT_PATH)
    alias_df = load_csv(TEAM_ALIAS_PATH)

    alias_base_df = clean_alias_reference(alias_df)
    alias_lookup_df = build_team_alias_lookup(alias_df)

    clean_df = clean_attendance(attendance_df, alias_lookup_df)
    validate(clean_df, alias_base_df)
    save_csv(clean_df, OUTPUT_PATH)

    logging.info("clean_attendance.py completed successfully.")
    logging.info("Output rows: %s", len(clean_df))


if __name__ == "__main__":
    main()