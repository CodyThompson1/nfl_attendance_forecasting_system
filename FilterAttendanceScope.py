"""
FilterAttendanceScope.py

Reads the combined attendance long dataset created by DataLoading.py,
filters to the 8-team scope (AFC West + NFC South),
and saves a scoped long file for easier downstream work.

Input:
- data/attendance/attendance_all_2015_2024_long.csv

Output:
- data/attendance/attendance_scope_8teams_2015_2024_long.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("C:/AAM_BMKT673/data/attendance")

INFILE = DATA_DIR / "attendance_all_2015_2024_long.csv"
OUTFILE = DATA_DIR / "attendance_scope_8teams_2015_2024_long.csv"

SCOPE_TEAMS = [
    "Denver Broncos",
    "Kansas City Chiefs",
    "Los Angeles Chargers",
    "Las Vegas Raiders",
    "Atlanta Falcons",
    "Carolina Panthers",
    "New Orleans Saints",
    "Tampa Bay Buccaneers",
]

def main():
    if not INFILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INFILE}\n"
            "Run DataLoading.py first to generate attendance_all_2015_2024_long.csv"
        )

    print(f"Loading: {INFILE}")
    df = pd.read_csv(INFILE)

    # Filter to the 8-team scope
    df_scope = df[df["Tm"].isin(SCOPE_TEAMS)].copy()

    # Save scoped file
    df_scope.to_csv(OUTFILE, index=False)

    # Quick validation prints
    print("\n--- Scoped Attendance Saved ---")
    print(f"Output: {OUTFILE}")
    print(f"Scoped shape: {df_scope.shape}")

    # Clean weeks display
    weeks_present = sorted(df_scope["week"].dropna().astype(int).unique().tolist())
    print("Weeks present:", weeks_present)

    # Missing attendance is expected (byes + week 18 not in older seasons)
    missing_rate = df_scope["attendance"].isna().mean()
    print(f"Missing attendance rate: {missing_rate:.2%}")

    # Sanity check: rows by season
    counts_by_season = df_scope.groupby("season")["attendance"].count()
    print("\nNon-missing attendance rows by season:")
    print(counts_by_season)

    print("\nPreview:")
    print(df_scope.head(10))

if __name__ == "__main__":
    main()
