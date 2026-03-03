'''# DataLoading.py
# Clean ALL Sports Reference attendance files (2015-2024)

import pandas as pd
from pathlib import Path

DATA_DIR = Path("C:/AAM_BMKT673/data/attendance")
SEASONS = list(range(2015, 2024 + 1))  # 2015-2024 inclusive

def load_one(file_path: Path) -> pd.DataFrame:
    """Loads one Sports Reference export that may be HTML-wrapped or a normal CSV."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().lower()

    if first_line.startswith("<html"):
        tables = pd.read_html(file_path)
        df = tables[0]
    else:
        df = pd.read_csv(file_path)

    return df

def clean_attendance_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the attendance table:
    - Keeps only rows where Tm looks like a real team (drops league total/avg row)
    - Ensures numeric columns are numeric
    """
    # Drop rows where Tm is missing
    df = df[df["Tm"].notna()].copy()

    # Sports Reference sometimes includes a final summary row not representing a team.
    # Heuristic: keep rows where "Tm" contains letters and a space (team names)
    # Also drop common summary labels if they appear.
    drop_labels = {"League Average", "Average", "Total", "Tm"}
    df = df[~df["Tm"].astype(str).str.strip().isin(drop_labels)].copy()

    # Convert Total/Home/Away to numeric (they can come in as floats/strings)
    for col in ["Total", "Home", "Away"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Week columns to numeric; "Bye" becomes NaN
    week_cols = [c for c in df.columns if str(c).startswith("Week ")]
    for col in week_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("Bye", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def main():
    for year in SEASONS:
        src = DATA_DIR / f"attendance_{year}.csv"
        if not src.exists():
            print(f"[WARN] Missing file: {src}")
            continue

        print(f"Loading: {src}")
        df = load_one(src)
        df = clean_attendance_df(df)

        # Save clean output
        out = DATA_DIR / f"attendance_{year}_clean.csv"
        df.to_csv(out, index=False)

        print(f"  Saved: {out} | shape={df.shape}")

    print("\nDone. All available seasons cleaned.")

if __name__ == "__main__":
    main()
'''
# DataLoading.py
# Combine cleaned attendance files into one master dataset (wide + long)

import pandas as pd
from pathlib import Path
import re

DATA_DIR = Path("C:/AAM_BMKT673/data/attendance")
SEASONS = list(range(2015, 2024 + 1))  # 2015-2024 inclusive

def main():
    wide_frames = []

    for year in SEASONS:
        fp = DATA_DIR / f"attendance_{year}_clean.csv"
        if not fp.exists():
            print(f"[WARN] Missing cleaned file: {fp}")
            continue

        df = pd.read_csv(fp)
        df["season"] = year
        wide_frames.append(df)

    if not wide_frames:
        raise FileNotFoundError("No cleaned attendance files found to combine.")

    # Combine into one wide table (keeps Week 1..Week 17/18 columns)
    attendance_wide = pd.concat(wide_frames, ignore_index=True)

    # Identify all week columns present across years
    week_cols = [c for c in attendance_wide.columns if re.match(r"^Week\s+\d+$", str(c))]
    if not week_cols:
        raise ValueError("No 'Week N' columns found in the cleaned files.")

    # Save wide version (optional but useful for inspection)
    out_wide = DATA_DIR / "attendance_all_2015_2024_wide.csv"
    attendance_wide.to_csv(out_wide, index=False)

    # Melt to long format
    id_cols = [c for c in ["season", "Tm", "Total", "Home", "Away"] if c in attendance_wide.columns]
    attendance_long = attendance_wide.melt(
        id_vars=id_cols,
        value_vars=week_cols,
        var_name="week_label",
        value_name="attendance"
    )

    # Parse week number
    attendance_long["week"] = attendance_long["week_label"].str.extract(r"(\d+)").astype(int)

    # Clean up
    attendance_long = attendance_long.drop(columns=["week_label"])

    # Reorder columns nicely
    cols_order = ["season", "week", "Tm", "attendance"]
    extras = [c for c in ["Total", "Home", "Away"] if c in attendance_long.columns]
    attendance_long = attendance_long[cols_order + extras]

    # Save long version (this is what you’ll use later)
    out_long = DATA_DIR / "attendance_all_2015_2024_long.csv"
    attendance_long.to_csv(out_long, index=False)

    # Show proof it worked
    print("\n--- Combined Attendance Created ---")
    print(f"Wide shape: {attendance_wide.shape} -> saved: {out_wide}")
    print(f"Long shape: {attendance_long.shape} -> saved: {out_long}")
    print("\nPreview (long):")
    print(attendance_long.head(10))

    # Quick diagnostics
    print("\nWeeks present:", sorted(attendance_long["week"].unique()))
    print("Missing attendance rate:", f"{attendance_long['attendance'].isna().mean():.2%}")

if __name__ == "__main__":
    main()
