from src.config import LEAGUE, DIVISIONS, SCOPE_TEAMS
from src.attendance_loader import load_attendance_scope
from src.performance_loader import load_team_performance
from src.weather_loader import load_weather
from src.calendar_features import load_calendar_features


def main():
    print("=== Week 1–3 Deliverable: League, Data Sources, Database Design ===\n")
    print(f"League selected: {LEAGUE}")
    print("Team scope:")
    for div, teams in DIVISIONS.items():
        print(f"  {div}: {teams}")
    print(f"\nTotal teams in scope: {len(SCOPE_TEAMS)}\n")

    # -----------------------------------------------------------------
    # Data Source 1: Attendance (Sports Reference CSV exports)
    # -----------------------------------------------------------------
    attendance = load_attendance_scope()
    print("Data Source 1: Attendance (Sports Reference exports)")
    print("  Loaded rows:", len(attendance))
    print("  Columns:", list(attendance.columns))
    print(attendance.head(), "\n")

    # -----------------------------------------------------------------
    # Data Source 2: Team Performance / Quality (derived from schedules)
    # -----------------------------------------------------------------
    perf = load_team_performance()
    print("Data Source 2: Team Performance / Quality (derived from game results)")
    print("  Loaded rows:", len(perf))
    print("  Columns:", list(perf.columns))
    print(perf.head(), "\n")

    # -----------------------------------------------------------------
    # Data Source 3: Weather (Meteostat)
    # -----------------------------------------------------------------
    import pandas as pd
    from pathlib import Path

    weather_file = Path("C:/AAM_BMKT673/data/weather/weather_sample_2023.csv")

    print("Data Source 3: Weather (Meteostat)")
    if weather_file.exists():
        weather = pd.read_csv(weather_file)
        print("  Loaded rows:", len(weather))
        print("  Columns:", list(weather.columns))
        print(weather.head(), "\n")
    else:
        print("  Weather sample file not found yet.")
        print("  Run: python WeatherSamplePull.py\n")
    # -----------------------------------------------------------------
    # Derived: Calendar features (from schedules)
    # -----------------------------------------------------------------
    calendar = load_calendar_features()
    print("Derived: Calendar features")
    print("  Loaded rows:", len(calendar))
    print("  Columns:", list(calendar.columns))
    print(calendar.head(), "\n")

    print("Done. All Week 1–3 sources are loading with real data.")

if __name__ == "__main__":
    main()
