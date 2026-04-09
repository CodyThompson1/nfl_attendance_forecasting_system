"""
File: build_fact_weather.py

Purpose:
Standardize the raw game-level weather extract into a warehouse-ready fact_weather
table with one row per game. This script creates a weather_id, keeps core weather
fields, derives a severe_weather_flag using reasonable business rules, preserves
useful quality-check columns, and writes the final output to CSV.

Inputs:
- data/raw/weather/game_weather_raw.csv

Outputs:
- data/processed/warehouse/fact_weather.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "weather" / "game_weather_raw.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "warehouse" / "fact_weather.csv"

REQUIRED_COLUMNS = {
    "game_id",
    "weather_timestamp",
    "temperature",
    "precipitation",
    "wind_speed",
    "weather_condition",
}

OUTPUT_COLUMNS = [
    "game_id",
    "weather_id",
    "temperature",
    "precipitation",
    "wind_speed",
    "weather_condition",
    "severe_weather_flag",
    "weather_timestamp",
]

QUALITY_AND_MODELING_COLUMNS = [
    "weather_source_method",
    "venue_name",
    "latitude",
    "longitude",
    "temperature_max",
    "temperature_min",
    "temperature_avg",
    "apparent_temperature",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_avg",
    "rainfall",
    "snowfall",
    "snow_depth",
    "wind_gust",
    "wind_direction",
    "sunshine",
    "weather_code",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_raw_weather() -> pd.DataFrame:
    logging.info("Loading file: %s", RAW_WEATHER_PATH)

    if not RAW_WEATHER_PATH.exists():
        raise FileNotFoundError(f"Missing required input file: {RAW_WEATHER_PATH}")

    return pd.read_csv(RAW_WEATHER_PATH)


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"game_weather_raw.csv is missing required columns: {missing_text}")


def coerce_numeric_series(dataframe: pd.DataFrame, column_name: str) -> None:
    if column_name in dataframe.columns:
        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")


def clean_weather_condition(value: object) -> Optional[str]:
    if pd.isna(value):
        return None

    value_text = str(value).strip()
    if not value_text:
        return None

    return value_text


def coalesce_weather_condition(row: pd.Series) -> Optional[str]:
    weather_condition = clean_weather_condition(row.get("weather_condition"))
    if weather_condition is not None:
        return weather_condition

    weather_code = pd.to_numeric(row.get("weather_code"), errors="coerce")
    if pd.notna(weather_code):
        return f"WMO_{int(weather_code)}"

    precipitation = pd.to_numeric(row.get("precipitation"), errors="coerce")
    snowfall = pd.to_numeric(row.get("snowfall"), errors="coerce")
    wind_speed = pd.to_numeric(row.get("wind_speed"), errors="coerce")

    condition_parts: list[str] = []

    if pd.notna(snowfall) and snowfall > 0:
        condition_parts.append("SNOW")
    elif pd.notna(precipitation) and precipitation > 0:
        condition_parts.append("PRECIP")

    if pd.notna(wind_speed) and wind_speed >= 20:
        condition_parts.append("WINDY")

    if not condition_parts:
        return "UNKNOWN"

    return "|".join(condition_parts)


def derive_severe_weather_flag(row: pd.Series) -> int:
    weather_condition = str(row.get("weather_condition") or "").upper()

    temperature = pd.to_numeric(row.get("temperature"), errors="coerce")
    precipitation = pd.to_numeric(row.get("precipitation"), errors="coerce")
    wind_speed = pd.to_numeric(row.get("wind_speed"), errors="coerce")
    wind_gust = pd.to_numeric(row.get("wind_gust"), errors="coerce")
    snowfall = pd.to_numeric(row.get("snowfall"), errors="coerce")

    severe_precipitation = pd.notna(precipitation) and precipitation >= 0.50
    severe_wind = pd.notna(wind_speed) and wind_speed >= 25
    severe_wind_gust = pd.notna(wind_gust) and wind_gust >= 35
    severe_cold = pd.notna(temperature) and temperature <= 20
    severe_heat = pd.notna(temperature) and temperature >= 95
    severe_snow = pd.notna(snowfall) and snowfall > 0
    severe_condition_text = any(
        keyword in weather_condition
        for keyword in [
            "SNOW",
            "STORM",
            "THUNDER",
            "SQUALL",
            "TORNADO",
            "FREEZING",
            "ICE",
            "BLIZZARD",
            "WINDY",
        ]
    )

    if any(
        [
            severe_precipitation,
            severe_wind,
            severe_wind_gust,
            severe_cold,
            severe_heat,
            severe_snow,
            severe_condition_text,
        ]
    ):
        return 1

    return 0


def build_weather_id(dataframe: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [f"WTH_{index:05d}" for index in range(1, len(dataframe) + 1)],
        index=dataframe.index,
    )


def standardize_weather_timestamp(dataframe: pd.DataFrame) -> None:
    dataframe["weather_timestamp"] = pd.to_datetime(
        dataframe["weather_timestamp"],
        errors="coerce",
    )

    dataframe["weather_timestamp"] = dataframe["weather_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    dataframe.loc[dataframe["weather_timestamp"] == "NaT", "weather_timestamp"] = None


def deduplicate_to_one_row_per_game(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "weather_timestamp" in dataframe.columns:
        dataframe["_weather_timestamp_sort"] = pd.to_datetime(
            dataframe["weather_timestamp"],
            errors="coerce",
        )
    else:
        dataframe["_weather_timestamp_sort"] = pd.NaT

    dataframe["_missing_core_fields"] = dataframe[
        ["temperature", "precipitation", "wind_speed"]
    ].isna().sum(axis=1)

    dataframe = dataframe.sort_values(
        by=["game_id", "_missing_core_fields", "_weather_timestamp_sort"],
        ascending=[True, True, True],
    )

    deduplicated_df = dataframe.drop_duplicates(subset=["game_id"], keep="first").copy()
    deduplicated_df = deduplicated_df.drop(
        columns=["_weather_timestamp_sort", "_missing_core_fields"],
        errors="ignore",
    )

    return deduplicated_df


def validate_output(dataframe: pd.DataFrame) -> None:
    duplicate_game_ids = int(dataframe["game_id"].duplicated().sum())
    if duplicate_game_ids > 0:
        raise ValueError(f"fact_weather.csv contains duplicate game_id values: {duplicate_game_ids}")

    missing_game_ids = int(dataframe["game_id"].isna().sum())
    if missing_game_ids > 0:
        raise ValueError(f"fact_weather.csv contains missing game_id values: {missing_game_ids}")

    missing_weather_ids = int(dataframe["weather_id"].isna().sum())
    if missing_weather_ids > 0:
        raise ValueError(f"fact_weather.csv contains missing weather_id values: {missing_weather_ids}")


def build_fact_weather() -> pd.DataFrame:
    weather_df = load_raw_weather()
    validate_required_columns(weather_df)

    numeric_columns = [
        "temperature",
        "precipitation",
        "wind_speed",
        "temperature_max",
        "temperature_min",
        "temperature_avg",
        "apparent_temperature",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "apparent_temperature_avg",
        "rainfall",
        "snowfall",
        "snow_depth",
        "wind_gust",
        "wind_direction",
        "sunshine",
        "weather_code",
        "latitude",
        "longitude",
    ]

    for column_name in numeric_columns:
        coerce_numeric_series(weather_df, column_name)

    weather_df["weather_condition"] = weather_df.apply(coalesce_weather_condition, axis=1)
    standardize_weather_timestamp(weather_df)

    weather_df = deduplicate_to_one_row_per_game(weather_df)
    weather_df["severe_weather_flag"] = weather_df.apply(derive_severe_weather_flag, axis=1)
    weather_df["weather_id"] = build_weather_id(weather_df)

    available_quality_columns = [
        column_name for column_name in QUALITY_AND_MODELING_COLUMNS
        if column_name in weather_df.columns
    ]

    final_columns = OUTPUT_COLUMNS + available_quality_columns
    fact_weather_df = weather_df[final_columns].copy()

    validate_output(fact_weather_df)
    return fact_weather_df


def save_output(dataframe: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_PATH, index=False)
    logging.info("fact_weather output written: %s", OUTPUT_PATH)


def main() -> None:
    configure_logging()
    fact_weather_df = build_fact_weather()
    save_output(fact_weather_df)


if __name__ == "__main__":
    main()