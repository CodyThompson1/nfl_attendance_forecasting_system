"""
File: extract_sportsref_attendance.py

Purpose: Read cleaned Sports Reference attendance CSV files stored locally by
season, reshape weekly attendance data into a single game-level raw attendance
file, and write the combined output to CSV for later cleaning and warehouse
loading.

Inputs:

* data/raw/attendance/attendance_2015_clean.csv through
  data/raw/attendance/attendance_2025_clean.csv
* Or any *_clean.csv attendance files in data/raw/attendance

Outputs:

* data/raw/attendance/sportsref_attendance_raw.csv
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "attendance"
OUTPUT_PATH = INPUT_DIR / "sportsref_attendance_raw.csv"
START_YEAR = 2015
END_YEAR = 2025


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_column_name(column_name: str) -> str:
    """Normalize a column name for easier matching."""
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def clean_text(value: object) -> str | None:
    """Clean text values and convert blanks to None."""
    if pd.isna(value):
        return None

    cleaned = str(value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned == "":
        return None

    return cleaned


def clean_attendance_value(value: object) -> int | None:
    """Convert attendance values to integers where possible."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    text = text.replace(",", "")
    text = re.sub(r"[^0-9]", "", text)

    if text == "":
        return None

    return int(text)


def clean_week_value(value: object) -> int | None:
    """Convert a week-like value to an integer when possible."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    match = re.search(r"([0-9]{1,2})", text)

    if not match:
        return None

    return int(match.group(1))


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column from a list of candidates."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def discover_input_files(input_dir: Path, start_year: int, end_year: int) -> list[Path]:
    """Discover cleaned attendance CSV files in the attendance input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    discovered_files: list[tuple[int, Path]] = []

    for file_path in sorted(input_dir.glob("*_clean.csv")):
        if file_path.name == OUTPUT_PATH.name:
            continue

        match = re.search(r"(20[0-9]{2})", file_path.stem)
        if not match:
            logging.info("Skipping cleaned file without season in name: %s", file_path.name)
            continue

        season = int(match.group(1))
        if start_year <= season <= end_year:
            discovered_files.append((season, file_path))

    if not discovered_files:
        raise FileNotFoundError(
            "No cleaned attendance CSV files were found in "
            f"{input_dir}. Expected files like attendance_2015_clean.csv."
        )

    discovered_files.sort(key=lambda item: (item[0], item[1].name))
    return [file_path for _, file_path in discovered_files]


def load_attendance_file(file_path: Path) -> pd.DataFrame:
    """Load a single cleaned attendance CSV file."""
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        logging.warning("File is empty after parsing: %s", file_path)

    return df


def identify_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Identify common columns in a cleaned Sports Reference attendance file."""
    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }
    available = list(normalized_columns.keys())

    season_col = find_column(
        available,
        ["season", "year"],
    )
    team_col = find_column(
        available,
        [
            "tm",
            "team",
            "home_team",
            "home_team_name",
            "franchise",
        ],
    )
    total_col = find_column(
        available,
        ["total"],
    )
    home_col = find_column(
        available,
        ["home"],
    )
    away_col = find_column(
        available,
        ["away"],
    )
    avg_col = find_column(
        available,
        ["avg", "average"],
    )
    pct_col = find_column(
        available,
        ["pct", "percentage"],
    )
    rank_col = find_column(
        available,
        ["rank", "rk"],
    )

    return {
        "season": normalized_columns.get(season_col) if season_col else None,
        "team": normalized_columns.get(team_col) if team_col else None,
        "total": normalized_columns.get(total_col) if total_col else None,
        "home": normalized_columns.get(home_col) if home_col else None,
        "away": normalized_columns.get(away_col) if away_col else None,
        "avg": normalized_columns.get(avg_col) if avg_col else None,
        "pct": normalized_columns.get(pct_col) if pct_col else None,
        "rank": normalized_columns.get(rank_col) if rank_col else None,
    }


def identify_week_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Identify week columns such as Week 1, Week 2, and so on."""
    week_columns: list[tuple[int, str]] = []

    for column in df.columns:
        column_text = str(column).strip()
        match = re.search(r"week\s*([0-9]{1,2})", column_text, flags=re.IGNORECASE)
        if match:
            week_number = int(match.group(1))
            week_columns.append((week_number, column))

    week_columns.sort(key=lambda item: item[0])
    return week_columns


def get_season_from_file_path(file_path: Path) -> int:
    """Extract the season year from a file name."""
    match = re.search(r"(20[0-9]{2})", file_path.stem)
    if not match:
        raise ValueError(f"Could not determine season from file name: {file_path.name}")

    return int(match.group(1))


def build_team_lookup() -> pd.DataFrame:
    """Create a lookup table for team names and abbreviations across relocations."""
    team_rows = [
        {"team_key": "Arizona Cardinals", "team_abbr": "ARI"},
        {"team_key": "Atlanta Falcons", "team_abbr": "ATL"},
        {"team_key": "Baltimore Ravens", "team_abbr": "BAL"},
        {"team_key": "Buffalo Bills", "team_abbr": "BUF"},
        {"team_key": "Carolina Panthers", "team_abbr": "CAR"},
        {"team_key": "Chicago Bears", "team_abbr": "CHI"},
        {"team_key": "Cincinnati Bengals", "team_abbr": "CIN"},
        {"team_key": "Cleveland Browns", "team_abbr": "CLE"},
        {"team_key": "Dallas Cowboys", "team_abbr": "DAL"},
        {"team_key": "Denver Broncos", "team_abbr": "DEN"},
        {"team_key": "Detroit Lions", "team_abbr": "DET"},
        {"team_key": "Green Bay Packers", "team_abbr": "GB"},
        {"team_key": "Houston Texans", "team_abbr": "HOU"},
        {"team_key": "Indianapolis Colts", "team_abbr": "IND"},
        {"team_key": "Jacksonville Jaguars", "team_abbr": "JAX"},
        {"team_key": "Kansas City Chiefs", "team_abbr": "KC"},
        {"team_key": "Las Vegas Raiders", "team_abbr": "LV"},
        {"team_key": "Oakland Raiders", "team_abbr": "LV"},
        {"team_key": "Los Angeles Chargers", "team_abbr": "LAC"},
        {"team_key": "San Diego Chargers", "team_abbr": "LAC"},
        {"team_key": "Los Angeles Rams", "team_abbr": "LAR"},
        {"team_key": "St. Louis Rams", "team_abbr": "LAR"},
        {"team_key": "Miami Dolphins", "team_abbr": "MIA"},
        {"team_key": "Minnesota Vikings", "team_abbr": "MIN"},
        {"team_key": "New England Patriots", "team_abbr": "NE"},
        {"team_key": "New Orleans Saints", "team_abbr": "NO"},
        {"team_key": "New York Giants", "team_abbr": "NYG"},
        {"team_key": "New York Jets", "team_abbr": "NYJ"},
        {"team_key": "Philadelphia Eagles", "team_abbr": "PHI"},
        {"team_key": "Pittsburgh Steelers", "team_abbr": "PIT"},
        {"team_key": "San Francisco 49ers", "team_abbr": "SF"},
        {"team_key": "Seattle Seahawks", "team_abbr": "SEA"},
        {"team_key": "Tampa Bay Buccaneers", "team_abbr": "TB"},
        {"team_key": "Tennessee Titans", "team_abbr": "TEN"},
        {"team_key": "Washington Commanders", "team_abbr": "WAS"},
        {"team_key": "Washington Football Team", "team_abbr": "WAS"},
        {"team_key": "Washington Redskins", "team_abbr": "WAS"},
    ]

    lookup_df = pd.DataFrame(team_rows)
    lookup_df["team_key_normalized"] = lookup_df["team_key"].map(normalize_team_name)
    return lookup_df


def normalize_team_name(team_name: str | None) -> str | None:
    """Normalize team names for consistent matching."""
    if team_name is None:
        return None

    normalized = clean_text(team_name)
    if normalized is None:
        return None

    normalized = normalized.replace(".", "")
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def derive_home_games_from_team_totals(
    source_df: pd.DataFrame,
    column_map: dict[str, str | None],
    file_path: Path,
) -> pd.DataFrame:
    """Fallback path that derives one row per home game from team week columns."""
    team_lookup_df = build_team_lookup()
    season_value_from_filename = get_season_from_file_path(file_path)
    week_columns = identify_week_columns(source_df)

    if not week_columns:
        raise ValueError(
            f"No week columns were found in {file_path}. "
            "Expected columns like 'Week 1', 'Week 2', and so on."
        )

    working_df = source_df.copy()

    if column_map["team"] is None:
        raise ValueError(
            f"Could not identify a team column in {file_path.name}. "
            "Expected a column such as 'Tm' or 'Team'."
        )

    working_df["home_team_name"] = working_df[column_map["team"]].map(clean_text)
    working_df["team_key_normalized"] = working_df["home_team_name"].map(normalize_team_name)

    working_df = working_df.merge(
        team_lookup_df[["team_key_normalized", "team_abbr"]],
        on="team_key_normalized",
        how="left",
    )

    rows: list[dict] = []

    for source_row in working_df.to_dict(orient="records"):
        source_season = (
            source_row.get(column_map["season"])
            if column_map["season"]
            else None
        )
        if pd.isna(source_season) or source_season is None:
            season = season_value_from_filename
        else:
            season = int(source_season)

        home_team_name = clean_text(source_row.get("home_team_name"))
        home_team_abbr = clean_text(source_row.get("team_abbr"))

        total_attendance = clean_attendance_value(
            source_row.get(column_map["total"]) if column_map["total"] else None
        )
        home_games = clean_attendance_value(
            source_row.get(column_map["home"]) if column_map["home"] else None
        )
        away_games = clean_attendance_value(
            source_row.get(column_map["away"]) if column_map["away"] else None
        )
        avg_attendance = clean_attendance_value(
            source_row.get(column_map["avg"]) if column_map["avg"] else None
        )
        attendance_pct = clean_text(
            source_row.get(column_map["pct"]) if column_map["pct"] else None
        )
        attendance_rank = clean_text(
            source_row.get(column_map["rank"]) if column_map["rank"] else None
        )

        for week_number, week_column in week_columns:
            attendance = clean_attendance_value(source_row.get(week_column))

            if attendance is None:
                continue

            rows.append(
                {
                    "season": season,
                    "week": week_number,
                    "home_team_name": home_team_name,
                    "home_team_abbr": home_team_abbr,
                    "attendance": attendance,
                    "team_season_total_attendance": total_attendance,
                    "team_home_games": home_games,
                    "team_away_games": away_games,
                    "team_avg_attendance": avg_attendance,
                    "team_attendance_pct": attendance_pct,
                    "team_attendance_rank": attendance_rank,
                    "attendance_source_file": file_path.name,
                    "attendance_source_column": week_column,
                    "source_format": "team_week_totals",
                    "source": "sports_reference_local_clean_csv",
                }
            )

    output_df = pd.DataFrame(rows)

    if output_df.empty:
        logging.warning("No attendance rows were created from %s", file_path)

    return output_df


def reshape_attendance_file(file_path: Path) -> pd.DataFrame:
    """Reshape a cleaned season attendance file to game-level rows."""
    season_df = load_attendance_file(file_path)
    column_map = identify_columns(season_df)

    reshaped_df = derive_home_games_from_team_totals(
        source_df=season_df,
        column_map=column_map,
        file_path=file_path,
    )

    return reshaped_df


def combine_attendance_files(start_year: int, end_year: int) -> pd.DataFrame:
    """Combine all season attendance files into one game-level raw dataset."""
    input_paths = discover_input_files(INPUT_DIR, start_year, end_year)
    all_frames: list[pd.DataFrame] = []

    logging.info("Using attendance directory: %s", INPUT_DIR)

    for file_path in input_paths:
        logging.info("Reading attendance file: %s", file_path)
        reshaped_df = reshape_attendance_file(file_path)
        all_frames.append(reshaped_df)

    if not all_frames:
        return pd.DataFrame()

    combined_df = pd.concat(all_frames, ignore_index=True)
    return combined_df


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize output data types, quality checks, and sort order."""
    if df.empty:
        return df

    output_df = df.copy()

    output_df["season"] = pd.to_numeric(output_df["season"], errors="coerce").astype("Int64")
    output_df["week"] = pd.to_numeric(output_df["week"], errors="coerce").astype("Int64")
    output_df["attendance"] = pd.to_numeric(output_df["attendance"], errors="coerce").astype("Int64")
    output_df["team_season_total_attendance"] = pd.to_numeric(
        output_df["team_season_total_attendance"],
        errors="coerce",
    ).astype("Int64")
    output_df["team_home_games"] = pd.to_numeric(
        output_df["team_home_games"],
        errors="coerce",
    ).astype("Int64")
    output_df["team_away_games"] = pd.to_numeric(
        output_df["team_away_games"],
        errors="coerce",
    ).astype("Int64")
    output_df["team_avg_attendance"] = pd.to_numeric(
        output_df["team_avg_attendance"],
        errors="coerce",
    ).astype("Int64")
    output_df["team_attendance_rank"] = pd.to_numeric(
        output_df["team_attendance_rank"],
        errors="coerce",
    ).astype("Int64")

    output_df["is_home_game_candidate"] = True
    output_df["record_created_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    ordered_columns = [
        "season",
        "week",
        "home_team_name",
        "home_team_abbr",
        "attendance",
        "team_season_total_attendance",
        "team_home_games",
        "team_away_games",
        "team_avg_attendance",
        "team_attendance_pct",
        "team_attendance_rank",
        "is_home_game_candidate",
        "attendance_source_file",
        "attendance_source_column",
        "source_format",
        "source",
        "record_created_utc",
    ]

    output_df = output_df[ordered_columns]
    output_df = output_df.sort_values(
        by=["season", "home_team_abbr", "week"],
        na_position="last",
    ).reset_index(drop=True)

    return output_df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write combined attendance data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Wrote %s rows to %s", len(df), output_path)


def main() -> None:
    """Run the local attendance extraction process."""
    setup_logging()

    combined_df = combine_attendance_files(
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    final_df = finalize_output(combined_df)
    write_output(final_df, OUTPUT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("extract_sportsref_attendance.py failed: %s", exc)
        sys.exit(1)