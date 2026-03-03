import pandas as pd
from .config import ATTENDANCE_DIR, SCOPE_TEAMS

def load_attendance_scope() -> pd.DataFrame:
    """
    Data Source #1: Historical Attendance (Sports Reference exports cleaned/combined)
    Current input file is already prepared from DataLoading/FilterAttenedanceScope.
    """
    fp = ATTENDANCE_DIR / "attendance_scope_8teams_2015_2024_long.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing attendance scope file: {fp}")

    df = pd.read_csv(fp)

    # Basic validation
    required_cols = {"season", "week", "Tm", "attendance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Attendance file missing columns: {missing}")

    # Filter again (safety)
    df = df[df["Tm"].isin(SCOPE_TEAMS)].copy()

    return df
