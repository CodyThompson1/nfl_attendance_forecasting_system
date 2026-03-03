import pandas as pd
from pathlib import Path

DATA_DIR = Path("C:/AAM_BMKT673/data/attendance")
INFILE = DATA_DIR / "attendance_scope_8teams_2015_2024_long.csv"
OUTFILE = DATA_DIR / "attendance_scope_qc_summary.csv"

def main():
    df = pd.read_csv(INFILE)

    summary = (
        df.groupby(["season", "Tm"])
          .agg(
              weeks_present=("week", "nunique"),
              non_missing_attendance=("attendance", lambda x: x.notna().sum()),
              missing_attendance=("attendance", lambda x: x.isna().sum()),
              avg_attendance=("attendance", "mean"),
              min_attendance=("attendance", "min"),
              max_attendance=("attendance", "max"),
          )
          .reset_index()
    )

    summary.to_csv(OUTFILE, index=False)

    print(f"Saved QC summary: {OUTFILE}")
    print(summary.head(10))

if __name__ == "__main__":
    main()
