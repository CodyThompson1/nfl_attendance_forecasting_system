"""
File: extract_weather_open_meteo.py

Purpose:
Extract one raw daily weather record per NFL game using the Open-Meteo Historical
Weather API. This script reads game, date, and venue warehouse tables, uses the
actual game venue latitude and longitude, requests daily weather for the game
date, assigns controlled values for indoor venues, and writes one raw weather
row per game to CSV.

Inputs:
- data/processed/warehouse/fact_game.csv
- data/processed/warehouse/dim_date.csv
- data/processed/warehouse/dim_venue.csv

Outputs:
- data/raw/weather/game_weather_raw.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE_DIR = PROJECT_ROOT / "data" / "processed" / "warehouse"
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"

FACT_GAME_PATH = WAREHOUSE_DIR / "fact_game.csv"
DIM_DATE_PATH = WAREHOUSE_DIR / "dim_date.csv"
DIM_VENUE_PATH = WAREHOUSE_DIR / "dim_venue.csv"
OUTPUT_PATH = RAW_WEATHER_DIR / "game_weather_raw.csv"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_SLEEP_SECONDS = 0.15
MAX_RETRIES = 3

REQUIRED_FACT_GAME_COLUMNS = {
    "game_id",
    "date_id",
    "venue_id",
}

REQUIRED_DIM_DATE_COLUMNS = {
    "date_id",
    "date",
}

REQUIRED_DIM_VENUE_COLUMNS = {
    "venue_id",
    "venue_name",
    "latitude",
    "longitude",
}

OPTIONAL_INDOOR_COLUMNS = [
    "indoor_flag",
    "is_indoor",
    "roof_type",
    "venue_type",
]

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "snow_depth_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "weather_code",
]

INDOOR_TEMPERATURE = 70.0
INDOOR_PRECIPITATION = 0.0
INDOOR_WIND_SPEED = 0.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_csv(file_path: Path) -> pd.DataFrame:
    logging.info("Loading file: %s", file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required input file: {file_path}")
    return pd.read_csv(file_path)


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    file_label: str,
) -> None:
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{file_label} is missing required columns: {missing_text}")


def to_bool_from_value(value: object) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    value_text = str(value).strip().lower()

    truthy_values = {
        "true",
        "t",
        "1",
        "yes",
        "y",
        "indoor",
        "closed",
        "domed",
        "dome",
    }
    falsy_values = {
        "false",
        "f",
        "0",
        "no",
        "n",
        "outdoor",
        "open",
    }

    if value_text in truthy_values:
        return True

    if value_text in falsy_values:
        return False

    return False


def infer_indoor_flag(venue_row: pd.Series) -> bool:
    for column_name in OPTIONAL_INDOOR_COLUMNS:
        if column_name not in venue_row.index:
            continue

        value = venue_row[column_name]

        if column_name in {"indoor_flag", "is_indoor"} and to_bool_from_value(value):
            return True

        if column_name == "roof_type":
            value_text = str(value).strip().lower()
            if value_text in {"closed", "dome", "domed", "retractable_closed", "indoor"}:
                return True

        if column_name == "venue_type":
            value_text = str(value).strip().lower()
            if value_text in {"indoor", "dome", "domed"}:
                return True

    return False


def coerce_numeric(value: object) -> Optional[float]:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return None
    return float(numeric_value)


def classify_weather_condition(
    weather_code: Optional[float],
    precipitation: Optional[float],
    snowfall: Optional[float],
    wind_speed: Optional[float],
) -> Optional[str]:
    condition_parts: list[str] = []

    if weather_code is not None:
        condition_parts.append(f"WMO_{int(weather_code)}")

    if snowfall is not None and snowfall > 0:
        condition_parts.append("SNOW")
    elif precipitation is not None and precipitation > 0:
        condition_parts.append("PRECIP")

    if wind_speed is not None and wind_speed >= 20:
        condition_parts.append("WINDY")

    if not condition_parts:
        return "CLEAR_OR_UNSPECIFIED"

    return "|".join(condition_parts)


def representative_temperature(
    temp_mean: Optional[float],
    temp_max: Optional[float],
    temp_min: Optional[float],
) -> Optional[float]:
    if temp_mean is not None:
        return temp_mean

    if temp_max is not None and temp_min is not None:
        return (temp_max + temp_min) / 2

    if temp_max is not None:
        return temp_max

    if temp_min is not None:
        return temp_min

    return None


def representative_apparent_temperature(
    apparent_mean: Optional[float],
    apparent_max: Optional[float],
    apparent_min: Optional[float],
    fallback_temperature: Optional[float],
) -> Optional[float]:
    if apparent_mean is not None:
        return apparent_mean

    if apparent_max is not None and apparent_min is not None:
        return (apparent_max + apparent_min) / 2

    if apparent_max is not None:
        return apparent_max

    if apparent_min is not None:
        return apparent_min

    return fallback_temperature


def build_missing_weather_row(
    game_row: pd.Series,
    source_method: str,
) -> dict:
    game_date = pd.to_datetime(game_row.get("date"), errors="coerce")

    return {
        "game_id": game_row["game_id"],
        "weather_timestamp": (
            game_date.strftime("%Y-%m-%d 12:00:00")
            if pd.notna(game_date)
            else None
        ),
        "temperature": None,
        "precipitation": None,
        "wind_speed": None,
        "weather_condition": None,
        "weather_source_method": source_method,
        "venue_name": game_row.get("venue_name"),
        "latitude": coerce_numeric(game_row.get("latitude")),
        "longitude": coerce_numeric(game_row.get("longitude")),
        "temperature_max": None,
        "temperature_min": None,
        "temperature_avg": None,
        "apparent_temperature": None,
        "apparent_temperature_max": None,
        "apparent_temperature_min": None,
        "apparent_temperature_avg": None,
        "rainfall": None,
        "snowfall": None,
        "snow_depth": None,
        "wind_gust": None,
        "wind_direction": None,
        "sunshine": None,
        "weather_code": None,
    }


def build_indoor_weather_row(game_row: pd.Series) -> dict:
    game_date = pd.to_datetime(game_row.get("date"), errors="coerce")

    return {
        "game_id": game_row["game_id"],
        "weather_timestamp": (
            game_date.strftime("%Y-%m-%d 12:00:00")
            if pd.notna(game_date)
            else None
        ),
        "temperature": INDOOR_TEMPERATURE,
        "precipitation": INDOOR_PRECIPITATION,
        "wind_speed": INDOOR_WIND_SPEED,
        "weather_condition": "INDOOR",
        "weather_source_method": "open_meteo_indoor_controlled_daily_values",
        "venue_name": game_row.get("venue_name"),
        "latitude": coerce_numeric(game_row.get("latitude")),
        "longitude": coerce_numeric(game_row.get("longitude")),
        "temperature_max": INDOOR_TEMPERATURE,
        "temperature_min": INDOOR_TEMPERATURE,
        "temperature_avg": INDOOR_TEMPERATURE,
        "apparent_temperature": INDOOR_TEMPERATURE,
        "apparent_temperature_max": INDOOR_TEMPERATURE,
        "apparent_temperature_min": INDOOR_TEMPERATURE,
        "apparent_temperature_avg": INDOOR_TEMPERATURE,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "snow_depth": 0.0,
        "wind_gust": 0.0,
        "wind_direction": None,
        "sunshine": None,
        "weather_code": None,
    }


def build_request_params(latitude: float, longitude: float, game_date: str) -> dict:
    return {
        "latitude": round(latitude, 6),
        "longitude": round(longitude, 6),
        "start_date": game_date,
        "end_date": game_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "GMT",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }


def request_open_meteo_daily_weather(latitude: float, longitude: float, game_date: str) -> dict:
    params = build_request_params(latitude=latitude, longitude=longitude, game_date=game_date)
    last_exception: Optional[Exception] = None

    for attempt_number in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict) and payload.get("error") is True:
                reason_text = payload.get("reason", "unknown_api_error")
                raise ValueError(f"Open-Meteo API error: {reason_text}")

            return payload

        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            logging.warning(
                "Open-Meteo request failed for lat=%s lon=%s date=%s attempt=%s/%s: %s",
                latitude,
                longitude,
                game_date,
                attempt_number,
                MAX_RETRIES,
                exc,
            )
            time.sleep(REQUEST_SLEEP_SECONDS * attempt_number)

    raise RuntimeError(
        f"Open-Meteo request failed after {MAX_RETRIES} attempts: {last_exception}"
    )


def extract_daily_value(daily_payload: dict, field_name: str) -> Optional[float]:
    daily_section = daily_payload.get("daily", {})
    if not isinstance(daily_section, dict):
        return None

    field_values = daily_section.get(field_name)
    if not isinstance(field_values, list) or len(field_values) == 0:
        return None

    return coerce_numeric(field_values[0])


def extract_daily_time(daily_payload: dict) -> Optional[str]:
    daily_section = daily_payload.get("daily", {})
    if not isinstance(daily_section, dict):
        return None

    time_values = daily_section.get("time")
    if not isinstance(time_values, list) or len(time_values) == 0:
        return None

    time_value = pd.to_datetime(time_values[0], errors="coerce")
    if pd.isna(time_value):
        return None

    return time_value.strftime("%Y-%m-%d 12:00:00")


def fetch_daily_weather_for_game(game_row: pd.Series) -> dict:
    latitude = coerce_numeric(game_row.get("latitude"))
    longitude = coerce_numeric(game_row.get("longitude"))
    game_date = pd.to_datetime(game_row.get("date"), errors="coerce")

    if pd.isna(game_date):
        return build_missing_weather_row(game_row, "missing_game_date")

    if latitude is None or longitude is None:
        return build_missing_weather_row(game_row, "missing_venue_coordinates")

    game_date_text = game_date.strftime("%Y-%m-%d")

    try:
        payload = request_open_meteo_daily_weather(
            latitude=latitude,
            longitude=longitude,
            game_date=game_date_text,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Weather pull failed for game_id=%s: %s", game_row["game_id"], exc)
        return build_missing_weather_row(game_row, "open_meteo_request_error")

    temp_max = extract_daily_value(payload, "temperature_2m_max")
    temp_min = extract_daily_value(payload, "temperature_2m_min")
    temp_mean = extract_daily_value(payload, "temperature_2m_mean")

    apparent_max = extract_daily_value(payload, "apparent_temperature_max")
    apparent_min = extract_daily_value(payload, "apparent_temperature_min")
    apparent_mean = extract_daily_value(payload, "apparent_temperature_mean")

    precipitation_sum = extract_daily_value(payload, "precipitation_sum")
    rain_sum = extract_daily_value(payload, "rain_sum")
    snowfall_sum = extract_daily_value(payload, "snowfall_sum")
    snow_depth_max = extract_daily_value(payload, "snow_depth_max")
    wind_speed_max = extract_daily_value(payload, "wind_speed_10m_max")
    wind_gust_max = extract_daily_value(payload, "wind_gusts_10m_max")
    wind_direction = extract_daily_value(payload, "wind_direction_10m_dominant")
    sunshine = extract_daily_value(payload, "shortwave_radiation_sum")
    weather_code = extract_daily_value(payload, "weather_code")

    temperature = representative_temperature(
        temp_mean=temp_mean,
        temp_max=temp_max,
        temp_min=temp_min,
    )

    apparent_temperature = representative_apparent_temperature(
        apparent_mean=apparent_mean,
        apparent_max=apparent_max,
        apparent_min=apparent_min,
        fallback_temperature=temperature,
    )

    weather_timestamp = extract_daily_time(payload)
    if weather_timestamp is None:
        weather_timestamp = game_date.strftime("%Y-%m-%d 12:00:00")

    weather_condition = classify_weather_condition(
        weather_code=weather_code,
        precipitation=precipitation_sum,
        snowfall=snowfall_sum,
        wind_speed=wind_speed_max,
    )

    return {
        "game_id": game_row["game_id"],
        "weather_timestamp": weather_timestamp,
        "temperature": temperature,
        "precipitation": precipitation_sum,
        "wind_speed": wind_speed_max,
        "weather_condition": weather_condition,
        "weather_source_method": "open_meteo_daily_by_venue_coordinates",
        "venue_name": game_row.get("venue_name"),
        "latitude": latitude,
        "longitude": longitude,
        "temperature_max": temp_max,
        "temperature_min": temp_min,
        "temperature_avg": temp_mean,
        "apparent_temperature": apparent_temperature,
        "apparent_temperature_max": apparent_max,
        "apparent_temperature_min": apparent_min,
        "apparent_temperature_avg": apparent_mean,
        "rainfall": rain_sum,
        "snowfall": snowfall_sum,
        "snow_depth": snow_depth_max,
        "wind_gust": wind_gust_max,
        "wind_direction": wind_direction,
        "sunshine": sunshine,
        "weather_code": weather_code,
    }


def prepare_game_weather_base() -> pd.DataFrame:
    fact_game_df = load_csv(FACT_GAME_PATH)
    dim_date_df = load_csv(DIM_DATE_PATH)
    dim_venue_df = load_csv(DIM_VENUE_PATH)

    validate_required_columns(fact_game_df, REQUIRED_FACT_GAME_COLUMNS, "fact_game.csv")
    validate_required_columns(dim_date_df, REQUIRED_DIM_DATE_COLUMNS, "dim_date.csv")
    validate_required_columns(dim_venue_df, REQUIRED_DIM_VENUE_COLUMNS, "dim_venue.csv")

    merged_df = fact_game_df.merge(
        dim_date_df[["date_id", "date"]],
        on="date_id",
        how="left",
        validate="many_to_one",
    )

    venue_columns = [
        "venue_id",
        "venue_name",
        "latitude",
        "longitude",
    ]
    for optional_column in OPTIONAL_INDOOR_COLUMNS:
        if optional_column in dim_venue_df.columns and optional_column not in venue_columns:
            venue_columns.append(optional_column)

    merged_df = merged_df.merge(
        dim_venue_df[venue_columns],
        on="venue_id",
        how="left",
        validate="many_to_one",
    )

    merged_df["indoor_inferred_flag"] = merged_df.apply(infer_indoor_flag, axis=1)

    duplicate_game_ids = int(merged_df["game_id"].duplicated().sum())
    if duplicate_game_ids > 0:
        raise ValueError(f"Duplicate game_id values found in fact_game.csv: {duplicate_game_ids}")

    return merged_df


def extract_game_weather() -> pd.DataFrame:
    game_weather_base_df = prepare_game_weather_base()
    weather_rows: list[dict] = []

    total_games = len(game_weather_base_df)
    logging.info("Extracting Open-Meteo daily weather for %s games.", total_games)

    for row_number, (_, game_row) in enumerate(game_weather_base_df.iterrows(), start=1):
        if row_number % 25 == 0 or row_number == total_games:
            logging.info(
                "Processing game %s of %s: %s",
                row_number,
                total_games,
                game_row["game_id"],
            )

        if game_row["indoor_inferred_flag"]:
            weather_rows.append(build_indoor_weather_row(game_row))
        else:
            weather_rows.append(fetch_daily_weather_for_game(game_row))

        time.sleep(REQUEST_SLEEP_SECONDS)

    weather_df = pd.DataFrame(weather_rows)

    expected_output_columns = [
        "game_id",
        "weather_timestamp",
        "temperature",
        "precipitation",
        "wind_speed",
        "weather_condition",
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

    for column_name in expected_output_columns:
        if column_name not in weather_df.columns:
            weather_df[column_name] = None

    weather_df = weather_df[expected_output_columns]
    return weather_df


def save_output(weather_df: pd.DataFrame) -> None:
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUTPUT_PATH, index=False)
    logging.info("Weather output written: %s", OUTPUT_PATH)


def main() -> None:
    configure_logging()
    weather_df = extract_game_weather()
    save_output(weather_df)


if __name__ == "__main__":
    main()