"""
WeatherSamplePull.py

Purpose:
    Simple, reliable proof-of-concept for pulling historical weather data using Meteostat.
    This is intentionally scoped small for Module 1 (Weeks 1–3) deliverable.

What it does:
    - Pulls daily weather for a small date range for selected team locations
    - Saves results to CSV for inspection
    - Does NOT attempt to join to attendance/schedules yet

Why:
    - Avoid long multi-year downloads and network timeouts before the first meeting
"""

from datetime import date
import time
import pandas as pd
import meteostat as ms


# ---------------------------------------------------------------------
# Small scope for meeting demo (edit if you want)
# ---------------------------------------------------------------------
TEAM_CITY_COORDS = {
    "DEN": (39.7392, -104.9903, 1609),  # Denver
    "KC":  (39.0997,  -94.5786, 270),   # Kansas City
    "ATL": (33.7490,  -84.3880, 320),   # Atlanta
    "NO":  (29.9511,  -90.0715, 2),     # New Orleans
}

# Short range (safe + fast)
START = date(2023, 9, 1)
END = date(2023, 12, 31)

# Retry settings (simple)
MAX_RETRIES = 3
SLEEP_SECONDS = 2

OUTFILE = "C:/AAM_BMKT673/data/weather/weather_sample_2023.csv"


def fetch_daily_weather_for_team(team: str, lat: float, lon: float, elev: float) -> pd.DataFrame:
    """
    Pull interpolated daily weather from Meteostat for a given location and date range.
    Retries a few times to handle transient network timeouts.
    """
    point = ms.Point(lat, lon, elev)

    # Nearby stations + daily time series
    stations = ms.stations.nearby(point, limit=4)
    ts = ms.daily(stations, START, END)

    # Retry wrapper around fetch
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ms.interpolate(ts, point).fetch()
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index()  # 'time' index -> column
            df["team"] = team
            return df
        except Exception as e:
            last_err = e
            print(f"[WARN] {team}: attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(SLEEP_SECONDS)

    print(f"[ERROR] {team}: failed after {MAX_RETRIES} retries. Last error: {last_err}")
    return pd.DataFrame()


def main():
    # Ensure output folder exists
    import os
    os.makedirs("C:/AAM_BMKT673/data/weather", exist_ok=True)

    frames = []
    for team, (lat, lon, elev) in TEAM_CITY_COORDS.items():
        print(f"Pulling weather for {team} from {START} to {END}...")
        df = fetch_daily_weather_for_team(team, lat, lon, elev)

        if df.empty:
            print(f"[WARN] No data returned for {team}.")
        else:
            print(f"  Rows: {len(df)}  Columns: {list(df.columns)}")
            frames.append(df)

    if not frames:
        print("[ERROR] No weather data retrieved. Check network or try a shorter date range.")
        return

    out = pd.concat(frames, ignore_index=True)

    # Keep only the most relevant columns for the deliverable
    keep_cols = [c for c in ["team", "time", "temp", "tmin", "tmax", "prcp", "wspd", "snwd"] if c in out.columns]
    out = out[keep_cols].copy()

    out.to_csv(OUTFILE, index=False)
    print(f"\nSaved weather sample to: {OUTFILE}")
    print(out.head(10))


if __name__ == "__main__":
    main()
