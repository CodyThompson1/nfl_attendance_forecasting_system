"""
File: build_dim_venue.py

Purpose:
Build the dim_venue warehouse table from clean_schedules.csv, venue_reference.csv,
and nfl_schedules_raw.csv. The script creates one row per unique cleaned venue and
fills missing venue metadata with a hard-coded venue lookup so the final output is
complete, warehouse-ready, and weather-ready.

Inputs:
- data/processed/schedules/clean_schedules.csv
- data/processed/schedules/venue_reference.csv
- data/raw/schedules/nfl_schedules_raw.csv

Outputs:
- data/processed/warehouse/dim_venue.csv
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

CLEAN_SCHEDULES_PATH = BASE_DIR / "data" / "processed" / "schedules" / "clean_schedules.csv"
VENUE_REFERENCE_PATH = BASE_DIR / "data" / "processed" / "schedules" / "venue_reference.csv"
RAW_SCHEDULE_PATH = BASE_DIR / "data" / "raw" / "schedules" / "nfl_schedules_raw.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "warehouse" / "dim_venue.csv"

REQUIRED_CLEAN_COLUMNS = [
    "season",
    "venue_clean",
    "game_country",
    "is_neutral_site",
    "is_international",
]

VENUE_LOOKUP = {
    "AT&T Stadium": {
        "city": "Arlington",
        "state_or_region": "Texas",
        "country": "United States",
        "roof_type": "retractable",
        "timezone_str": "America/Chicago",
        "latitude": 32.7473,
        "longitude": -97.0945,
    },
    "Acrisure Stadium": {
        "city": "Pittsburgh",
        "state_or_region": "Pennsylvania",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 40.4468,
        "longitude": -80.0158,
    },
    "Allegiant Stadium": {
        "city": "Las Vegas",
        "state_or_region": "Nevada",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/Los_Angeles",
        "latitude": 36.0908,
        "longitude": -115.1830,
    },
    "Allianz Arena": {
        "city": "Munich",
        "state_or_region": "Bavaria",
        "country": "Germany",
        "roof_type": "outdoors",
        "timezone_str": "Europe/Berlin",
        "latitude": 48.2188,
        "longitude": 11.6247,
    },
    "Arena Corinthians": {
        "city": "Sao Paulo",
        "state_or_region": "Sao Paulo",
        "country": "Brazil",
        "roof_type": "outdoors",
        "timezone_str": "America/Sao_Paulo",
        "latitude": -23.5453,
        "longitude": -46.4740,
    },
    "Arrowhead Stadium": {
        "city": "Kansas City",
        "state_or_region": "Missouri",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Chicago",
        "latitude": 39.0489,
        "longitude": -94.4839,
    },
    "Bank of America Stadium": {
        "city": "Charlotte",
        "state_or_region": "North Carolina",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 35.2258,
        "longitude": -80.8528,
    },
    "Caesars Superdome": {
        "city": "New Orleans",
        "state_or_region": "Louisiana",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/Chicago",
        "latitude": 29.9511,
        "longitude": -90.0812,
    },
    "Deutsche Bank Park": {
        "city": "Frankfurt",
        "state_or_region": "Hesse",
        "country": "Germany",
        "roof_type": "outdoors",
        "timezone_str": "Europe/Berlin",
        "latitude": 50.0686,
        "longitude": 8.6455,
    },
    "Edward Jones Dome": {
        "city": "St. Louis",
        "state_or_region": "Missouri",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/Chicago",
        "latitude": 38.6329,
        "longitude": -90.1887,
    },
    "Empower Field at Mile High": {
        "city": "Denver",
        "state_or_region": "Colorado",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Denver",
        "latitude": 39.7439,
        "longitude": -105.0201,
    },
    "Estadio Azteca": {
        "city": "Mexico City",
        "state_or_region": "Mexico City",
        "country": "Mexico",
        "roof_type": "outdoors",
        "timezone_str": "America/Mexico_City",
        "latitude": 19.3029,
        "longitude": -99.1505,
    },
    "EverBank Stadium": {
        "city": "Jacksonville",
        "state_or_region": "Florida",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 30.3239,
        "longitude": -81.6373,
    },
    "Ford Field": {
        "city": "Detroit",
        "state_or_region": "Michigan",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/Detroit",
        "latitude": 42.3400,
        "longitude": -83.0456,
    },
    "Georgia Dome": {
        "city": "Atlanta",
        "state_or_region": "Georgia",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/New_York",
        "latitude": 33.7573,
        "longitude": -84.4008,
    },
    "Gillette Stadium": {
        "city": "Foxborough",
        "state_or_region": "Massachusetts",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 42.0909,
        "longitude": -71.2643,
    },
    "Hard Rock Stadium": {
        "city": "Miami Gardens",
        "state_or_region": "Florida",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 25.9580,
        "longitude": -80.2389,
    },
    "Highmark Stadium": {
        "city": "Orchard Park",
        "state_or_region": "New York",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 42.7738,
        "longitude": -78.7868,
    },
    "Huntington Bank Field": {
        "city": "Cleveland",
        "state_or_region": "Ohio",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 41.5061,
        "longitude": -81.6995,
    },
    "Lambeau Field": {
        "city": "Green Bay",
        "state_or_region": "Wisconsin",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Chicago",
        "latitude": 44.5013,
        "longitude": -88.0622,
    },
    "Levi's Stadium": {
        "city": "Santa Clara",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 37.4030,
        "longitude": -121.9700,
    },
    "Lincoln Financial Field": {
        "city": "Philadelphia",
        "state_or_region": "Pennsylvania",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 39.9008,
        "longitude": -75.1675,
    },
    "Los Angeles Memorial Coliseum": {
        "city": "Los Angeles",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 34.0141,
        "longitude": -118.2879,
    },
    "Lucas Oil Stadium": {
        "city": "Indianapolis",
        "state_or_region": "Indiana",
        "country": "United States",
        "roof_type": "retractable",
        "timezone_str": "America/Indiana/Indianapolis",
        "latitude": 39.7601,
        "longitude": -86.1639,
    },
    "Lumen Field": {
        "city": "Seattle",
        "state_or_region": "Washington",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 47.5952,
        "longitude": -122.3316,
    },
    "M&T Bank Stadium": {
        "city": "Baltimore",
        "state_or_region": "Maryland",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 39.2780,
        "longitude": -76.6227,
    },
    "Mercedes-Benz Stadium": {
        "city": "Atlanta",
        "state_or_region": "Georgia",
        "country": "United States",
        "roof_type": "retractable",
        "timezone_str": "America/New_York",
        "latitude": 33.7554,
        "longitude": -84.4009,
    },
    "MetLife Stadium": {
        "city": "East Rutherford",
        "state_or_region": "New Jersey",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 40.8135,
        "longitude": -74.0745,
    },
    "NRG Stadium": {
        "city": "Houston",
        "state_or_region": "Texas",
        "country": "United States",
        "roof_type": "retractable",
        "timezone_str": "America/Chicago",
        "latitude": 29.6847,
        "longitude": -95.4107,
    },
    "Nissan Stadium": {
        "city": "Nashville",
        "state_or_region": "Tennessee",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Chicago",
        "latitude": 36.1665,
        "longitude": -86.7713,
    },
    "Northwest Stadium": {
        "city": "Landover",
        "state_or_region": "Maryland",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 38.9078,
        "longitude": -76.8644,
    },
    "O.co Coliseum": {
        "city": "Oakland",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 37.7516,
        "longitude": -122.2005,
    },
    "Oakland Coliseum": {
        "city": "Oakland",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 37.7516,
        "longitude": -122.2005,
    },
    "Paycor Stadium": {
        "city": "Cincinnati",
        "state_or_region": "Ohio",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 39.0955,
        "longitude": -84.5160,
    },
    "Qualcomm Stadium": {
        "city": "San Diego",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 32.7831,
        "longitude": -117.1195,
    },
    "Raymond James Stadium": {
        "city": "Tampa",
        "state_or_region": "Florida",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 27.9759,
        "longitude": -82.5033,
    },
    "Ring Central Coliseum": {
        "city": "Oakland",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 37.7516,
        "longitude": -122.2005,
    },
    "SoFi Stadium": {
        "city": "Inglewood",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "covered",
        "timezone_str": "America/Los_Angeles",
        "latitude": 33.9535,
        "longitude": -118.3392,
    },
    "Soldier Field": {
        "city": "Chicago",
        "state_or_region": "Illinois",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Chicago",
        "latitude": 41.8623,
        "longitude": -87.6167,
    },
    "State Farm Stadium": {
        "city": "Glendale",
        "state_or_region": "Arizona",
        "country": "United States",
        "roof_type": "retractable",
        "timezone_str": "America/Phoenix",
        "latitude": 33.5276,
        "longitude": -112.2626,
    },
    "StubHub Center": {
        "city": "Carson",
        "state_or_region": "California",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Los_Angeles",
        "latitude": 33.8644,
        "longitude": -118.2611,
    },
    "TCF Bank Stadium": {
        "city": "Minneapolis",
        "state_or_region": "Minnesota",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/Chicago",
        "latitude": 44.9760,
        "longitude": -93.2247,
    },
    "TIAA Bank Stadium": {
        "city": "Jacksonville",
        "state_or_region": "Florida",
        "country": "United States",
        "roof_type": "outdoors",
        "timezone_str": "America/New_York",
        "latitude": 30.3239,
        "longitude": -81.6373,
    },
    "Tottenham Hotspur Stadium": {
        "city": "London",
        "state_or_region": "England",
        "country": "United Kingdom",
        "roof_type": "outdoors",
        "timezone_str": "Europe/London",
        "latitude": 51.6043,
        "longitude": -0.0664,
    },
    "Twickenham Stadium": {
        "city": "London",
        "state_or_region": "England",
        "country": "United Kingdom",
        "roof_type": "outdoors",
        "timezone_str": "Europe/London",
        "latitude": 51.4559,
        "longitude": -0.3417,
    },
    "U.S. Bank Stadium": {
        "city": "Minneapolis",
        "state_or_region": "Minnesota",
        "country": "United States",
        "roof_type": "dome",
        "timezone_str": "America/Chicago",
        "latitude": 44.9738,
        "longitude": -93.2575,
    },
    "Wembley Stadium": {
        "city": "London",
        "state_or_region": "England",
        "country": "United Kingdom",
        "roof_type": "outdoors",
        "timezone_str": "Europe/London",
        "latitude": 51.5560,
        "longitude": -0.2796,
    },
}

VENUE_NAME_REMAP = {
    "Cleveland Browns Stadium": "Huntington Bank Field",
    "FirstEnergy Stadium": "Huntington Bank Field",
    "FedExField": "Northwest Stadium",
    "Northwest Field": "Northwest Stadium",
    "Heinz Field": "Acrisure Stadium",
    "New Era Field": "Highmark Stadium",
    "Ralph Wilson Stadium": "Highmark Stadium",
    "Bills Stadium": "Highmark Stadium",
    "Paul Brown Stadium": "Paycor Stadium",
    "Sports Authority Field at Mile High": "Empower Field at Mile High",
    "Broncos Stadium at Mile High": "Empower Field at Mile High",
    "Invesco Field at Mile High": "Empower Field at Mile High",
    "Oakland-Alameda County Coliseum": "Oakland Coliseum",
}


def normalize_string(value: object) -> str | None:
    """Return a cleaned string or None."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None

    return text


def normalize_bool(value: object) -> bool:
    """Convert common boolean-like values to bool."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    return text in {"true", "t", "1", "yes", "y"}


def load_required_csv(file_path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a required CSV and validate required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {file_path.name}: {missing_text}")

    return df


def load_optional_csv(file_path: Path) -> pd.DataFrame:
    """Load an optional CSV if present."""
    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def normalize_venue_name(venue_name: object) -> str | None:
    """Normalize venue names to a consistent cleaned value."""
    text = normalize_string(venue_name)
    if text is None:
        return None

    return VENUE_NAME_REMAP.get(text, text)


def standardize_roof_type(roof_value: object) -> str | None:
    """Convert raw roof values into a stable venue-level roof type."""
    text = normalize_string(roof_value)
    if text is None:
        return None

    roof = text.lower()

    if roof in {"outdoors", "outdoor", "open"}:
        return "outdoors"

    if roof in {"indoors", "indoor", "closed", "fixed", "dome"}:
        return "dome"

    if roof in {"retractable", "retractable roof"}:
        return "retractable"

    if roof in {"covered"}:
        return "covered"

    return roof


def roof_type_to_indoor_flag(roof_type: object) -> bool:
    """Set indoor_flag from normalized roof_type."""
    text = normalize_string(roof_type)
    if text is None:
        return False

    return text.lower() in {"dome", "covered"}


def build_schedule_venue_base(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Build one record per venue from clean schedules."""
    clean_df = clean_df.copy()
    clean_df["season"] = pd.to_numeric(clean_df["season"], errors="coerce")
    clean_df = clean_df[clean_df["season"].between(2015, 2025, inclusive="both")].copy()

    clean_df["venue_name"] = clean_df["venue_clean"].apply(normalize_venue_name)
    clean_df["game_country"] = clean_df["game_country"].apply(normalize_string)
    clean_df["is_neutral_site"] = clean_df["is_neutral_site"].apply(normalize_bool)
    clean_df["is_international"] = clean_df["is_international"].apply(normalize_bool)

    clean_df = clean_df[clean_df["venue_name"].notna()].copy()

    base_df = (
        clean_df.groupby("venue_name", as_index=False)
        .agg(
            country=("game_country", "first"),
            neutral_site_flag=("is_neutral_site", "max"),
            international_flag=("is_international", "max"),
        )
    )

    return base_df


def build_raw_venue_attributes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract venue-level attributes from raw schedules."""
    if raw_df.empty:
        return pd.DataFrame(columns=["venue_name", "roof_type"])

    raw_df = raw_df.copy()

    if "game_stadium" in raw_df.columns:
        raw_df["venue_name"] = raw_df["game_stadium"].apply(normalize_venue_name)
    elif "stadium" in raw_df.columns:
        raw_df["venue_name"] = raw_df["stadium"].apply(normalize_venue_name)
    else:
        raw_df["venue_name"] = None

    if "roof" in raw_df.columns:
        raw_df["roof_type"] = raw_df["roof"].apply(standardize_roof_type)
    else:
        raw_df["roof_type"] = None

    raw_df = raw_df[raw_df["venue_name"].notna()].copy()

    roof_priority = {
        "retractable": 1,
        "dome": 2,
        "covered": 3,
        "outdoors": 4,
        None: 99,
    }

    raw_df["roof_rank"] = raw_df["roof_type"].map(roof_priority).fillna(50)

    venue_roof_df = (
        raw_df.sort_values(["venue_name", "roof_rank"])
        .drop_duplicates(subset=["venue_name"], keep="first")
        [["venue_name", "roof_type"]]
        .reset_index(drop=True)
    )

    return venue_roof_df


def build_reference_venue_attributes(venue_ref_df: pd.DataFrame) -> pd.DataFrame:
    """Extract optional attributes from venue_reference.csv."""
    if venue_ref_df.empty:
        return pd.DataFrame(columns=["venue_name", "country_ref"])

    venue_ref_df = venue_ref_df.copy()
    venue_ref_df["venue_name"] = venue_ref_df["venue_clean"].apply(normalize_venue_name)

    if "game_country" in venue_ref_df.columns:
        venue_ref_df["country_ref"] = venue_ref_df["game_country"].apply(normalize_string)
    else:
        venue_ref_df["country_ref"] = None

    venue_ref_df = venue_ref_df[venue_ref_df["venue_name"].notna()].copy()

    venue_ref_df = (
        venue_ref_df.sort_values("venue_name")
        .drop_duplicates(subset=["venue_name"], keep="first")
        [["venue_name", "country_ref"]]
        .reset_index(drop=True)
    )

    return venue_ref_df


def apply_hard_coded_venue_metadata(dim_venue_df: pd.DataFrame) -> pd.DataFrame:
    """Fill venue metadata from the hard-coded lookup."""
    dim_venue_df = dim_venue_df.copy()

    dim_venue_df["city"] = None
    dim_venue_df["state_or_region"] = None
    dim_venue_df["timezone_str"] = None
    dim_venue_df["latitude"] = None
    dim_venue_df["longitude"] = None
    dim_venue_df["country_lookup"] = None
    dim_venue_df["roof_type_lookup"] = None

    for venue_name, metadata in VENUE_LOOKUP.items():
        mask = dim_venue_df["venue_name"] == venue_name
        if not mask.any():
            continue

        dim_venue_df.loc[mask, "city"] = metadata["city"]
        dim_venue_df.loc[mask, "state_or_region"] = metadata["state_or_region"]
        dim_venue_df.loc[mask, "timezone_str"] = metadata["timezone_str"]
        dim_venue_df.loc[mask, "latitude"] = metadata["latitude"]
        dim_venue_df.loc[mask, "longitude"] = metadata["longitude"]
        dim_venue_df.loc[mask, "country_lookup"] = metadata["country"]
        dim_venue_df.loc[mask, "roof_type_lookup"] = metadata["roof_type"]

    return dim_venue_df


def build_dim_venue(
    clean_df: pd.DataFrame,
    venue_ref_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final dim_venue table."""
    schedule_base_df = build_schedule_venue_base(clean_df)
    reference_df = build_reference_venue_attributes(venue_ref_df)
    raw_attr_df = build_raw_venue_attributes(raw_df)

    dim_venue_df = schedule_base_df.merge(reference_df, on="venue_name", how="left")
    dim_venue_df = dim_venue_df.merge(raw_attr_df, on="venue_name", how="left")
    dim_venue_df = apply_hard_coded_venue_metadata(dim_venue_df)

    dim_venue_df["country"] = dim_venue_df["country"].combine_first(dim_venue_df["country_ref"])
    dim_venue_df["country"] = dim_venue_df["country"].combine_first(dim_venue_df["country_lookup"])
    dim_venue_df["roof_type"] = dim_venue_df["roof_type"].combine_first(dim_venue_df["roof_type_lookup"])

    dim_venue_df["country"] = dim_venue_df["country"].fillna("United States")
    dim_venue_df["roof_type"] = dim_venue_df["roof_type"].fillna("outdoors")

    dim_venue_df["indoor_flag"] = dim_venue_df["roof_type"].apply(roof_type_to_indoor_flag)
    dim_venue_df["neutral_site_flag"] = dim_venue_df["neutral_site_flag"].fillna(False)
    dim_venue_df["international_flag"] = dim_venue_df["international_flag"].fillna(False)

    dim_venue_df = dim_venue_df[
        [
            "venue_name",
            "city",
            "state_or_region",
            "country",
            "indoor_flag",
            "roof_type",
            "timezone_str",
            "latitude",
            "longitude",
            "neutral_site_flag",
            "international_flag",
        ]
    ].copy()

    dim_venue_df = (
        dim_venue_df.sort_values("venue_name")
        .drop_duplicates(subset=["venue_name"], keep="first")
        .reset_index(drop=True)
    )

    dim_venue_df.insert(0, "venue_id", range(1, len(dim_venue_df) + 1))

    return dim_venue_df


def validate_dim_venue(dim_venue_df: pd.DataFrame) -> None:
    """Validate final dim_venue output."""
    if dim_venue_df.empty:
        raise ValueError("dim_venue output is empty.")

    if dim_venue_df["venue_name"].isna().any():
        raise ValueError("Null venue_name values found in dim_venue.")

    if dim_venue_df["venue_name"].duplicated().any():
        duplicate_names = dim_venue_df.loc[
            dim_venue_df["venue_name"].duplicated(), "venue_name"
        ].tolist()
        raise ValueError(f"Duplicate venue_name values found in dim_venue: {duplicate_names}")

    missing_columns = [column for column in [
        "venue_id",
        "venue_name",
        "city",
        "state_or_region",
        "country",
        "indoor_flag",
        "roof_type",
        "timezone_str",
        "latitude",
        "longitude",
        "neutral_site_flag",
        "international_flag",
    ] if column not in dim_venue_df.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing output columns: {missing_text}")

    required_non_null_columns = [
        "venue_name",
        "city",
        "state_or_region",
        "country",
        "roof_type",
        "timezone_str",
        "latitude",
        "longitude",
    ]

    for column in required_non_null_columns:
        if dim_venue_df[column].isna().any():
            missing_venues = dim_venue_df.loc[dim_venue_df[column].isna(), "venue_name"].tolist()
            raise ValueError(f"Missing values found in {column} for venues: {missing_venues}")


def write_dim_venue(dim_venue_df: pd.DataFrame, output_path: Path) -> None:
    """Write dim_venue to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dim_venue_df.to_csv(output_path, index=False)


def main() -> None:
    """Run the dim_venue build process."""
    clean_df = load_required_csv(CLEAN_SCHEDULES_PATH, REQUIRED_CLEAN_COLUMNS)
    venue_ref_df = load_optional_csv(VENUE_REFERENCE_PATH)
    raw_df = load_optional_csv(RAW_SCHEDULE_PATH)

    dim_venue_df = build_dim_venue(
        clean_df=clean_df,
        venue_ref_df=venue_ref_df,
        raw_df=raw_df,
    )

    validate_dim_venue(dim_venue_df)
    write_dim_venue(dim_venue_df, OUTPUT_PATH)

    print(f"dim_venue written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()