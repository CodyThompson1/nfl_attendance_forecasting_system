"""
weather_loader.py

Purpose:
    Load historical daily weather for NFL home games using Meteostat.

Implementation:
    Uses the Meteostat API style shown in their docs:
        import meteostat as ms
        stations = ms.stations.nearby(POINT, limit=4)
        ts = ms.daily(stations, START, END)
        df = ms.interpolate(ts, POINT).fetch()

Output Grain:
    Home-game level (season, week, home_team, gameday + weather fields)
"""

from datetime import date
import pandas as pd
import nflreadpy as nfl
import meteostat as ms

from .config import ALL_YEARS, SCOPE_TEAM_ABBRS


# ---------------------------------------------------------------------
# City coordinates (simple + reliable for Module 1)
# ---------------------------------------------------------------------
TEAM_CITY_COORDS = {
    "ATL": (33.7490, -84.3880, 320),   # Atlanta (approx elevation meters)
    "CAR": (35.2271, -80.8431, 230),   # Charlotte
    "DEN": (39.7392, -104.9903, 1609), # Denver
    "KC":  (39.0997, -94.5786, 270),   # Kansas City
    "LAC": (34.0522, -118.2437, 90),   # Los Angeles
    "LV":  (36.1699, -115.1398, 610),  # Las Vegas
    "NO":  (29.9511, -90.0715, 2),     # New Orleans
    "TB":  (27.9506, -82.4572, 15),    # Tampa
}


def _fetch_team_season_daily(team: str, season: int, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame | None:
    """
    Fetch interpolated daily weather for one team and one season.

    Returns a DataFrame indexed by date with Meteostat columns like:
        temp, tmin, tmax, prcp, wspd, snwd, ...
    """
    lat, lon, elev = TEAM_CITY_COORDS[team]
    point = ms.Point(lat, lon, elev)

    start = date(start_dt.year, start_dt.month, start_dt.day)
    end = date(end_dt.year, end_dt.month, end_dt.day)

    try:
        # Nearby stations (Meteostat docs style)
        stns = ms.stations.nearby(point, limit=4)

        # Daily time series (stations can be a station set / query object)
        ts = ms.daily(stns, start, end)

        # Interpolate to the point (more accurate than using a single station)
        df = ms.interpolate(ts, point).fetch()

        if df is None or df.empty:
            return None

        # Convert the index to a date column
        df = df.reset_index()  # index name is usually 'time'
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"]).dt.date
        else:
            # Defensive: find any date-like column
            df["date"] = pd.to_datetime(df.iloc[:, 0]).dt.date

        df["season"] = season
        df["home_team"] = team
        return df

    except Exception as e:
        print(f"[WARN] Meteostat failed for {team} season {season}: {e}")
        return None


def load_weather() -> pd.DataFrame:
    """
    Data Source #3: Weather data via Meteostat.

    Steps:
        1) Load nfl schedules (regular season only) for project years
        2) Filter to scope home teams
        3) For each team-season, fetch interpolated daily weather using Meteostat
        4) Merge daily weather onto game dates

    Returns:
        season, week, home_team, gameday,
        tavg, tmin, tmax, prcp, wspd, snow,
        severe_weather_flag
    """
    schedules = nfl.load_schedules(ALL_YEARS).to_pandas()

    # Regular season, scope home teams
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    schedules = schedules[schedules["home_team"].isin(SCOPE_TEAM_ABBRS)].copy()

    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    schedules["date"] = schedules["gameday"].dt.date

    weather_frames = []

    for team in sorted(SCOPE_TEAM_ABBRS):
        if team not in TEAM_CITY_COORDS:
            print(f"[WARN] No coords configured for {team}. Skipping weather.")
            continue

        team_sched = schedules[schedules["home_team"] == team]
        if team_sched.empty:
            continue

        for season in sorted(team_sched["season"].unique()):
            season_sched = team_sched[team_sched["season"] == season]
            start_dt = season_sched["gameday"].min().normalize()
            end_dt = season_sched["gameday"].max().normalize()

            df_daily = _fetch_team_season_daily(team, int(season), start_dt, end_dt)
            if df_daily is not None:
                weather_frames.append(df_daily)
            else:
                print(f"[WARN] No daily weather returned for {team} season {season}.")

    # If still nothing, return a correctly-shaped empty frame (pipeline-safe)
    cols = [
        "season", "week", "home_team", "gameday",
        "tavg", "tmin", "tmax", "prcp", "wspd", "snow",
        "severe_weather_flag"
    ]
    if not weather_frames:
        return pd.DataFrame(columns=cols)

    weather_daily = pd.concat(weather_frames, ignore_index=True)

    # Meteostat column names (common):
    # temp (avg), tmin, tmax, prcp, wspd, snwd
    rename_map = {
        "temp": "tavg",
        "snwd": "snow",
    }
    weather_daily = weather_daily.rename(columns=rename_map)

    # Keep only columns we care about (if present)
    keep_cols = ["season", "home_team", "date", "tavg", "tmin", "tmax", "prcp", "wspd", "snow"]
    weather_daily = weather_daily[[c for c in keep_cols if c in weather_daily.columns]].copy()

    # Merge onto game dates
    out = schedules.merge(weather_daily, on=["season", "home_team", "date"], how="left")

    # Severe weather heuristic (meeting-ready; refine later)
    out["severe_weather_flag"] = (
        (out.get("prcp", 0).fillna(0) >= 10) |
        (out.get("wspd", 0).fillna(0) >= 35) |
        (out.get("snow", 0).fillna(0) >= 5)
    )

    out = out[cols].copy()
    return out
