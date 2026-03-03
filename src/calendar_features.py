import pandas as pd
import nflreadpy as nfl
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import ALL_YEARS, SCOPE_TEAM_ABBRS


def load_calendar_features() -> pd.DataFrame:
    """
    Derived Feature Set: Calendar features from game dates.

    Source:
        nflverse schedules via nflreadpy (gameday)

    Returns (home-game level):
        season, week, home_team, gameday,
        day_of_week, weekend_flag, holiday_flag
    """
    schedules = nfl.load_schedules(ALL_YEARS).to_pandas()


    # Keep only regular season home games for our scope teams
    schedules = schedules[schedules["game_type"] == "REG"].copy()
    schedules = schedules[schedules["home_team"].isin(SCOPE_TEAM_ABBRS)].copy()

    schedules["gameday"] = pd.to_datetime(schedules["gameday"])
    schedules["day_of_week"] = schedules["gameday"].dt.day_name()
    schedules["weekend_flag"] = schedules["gameday"].dt.weekday >= 5  # Sat/Sun

    # Holiday flag (US federal holidays)
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=schedules["gameday"].min(), end=schedules["gameday"].max())
    holiday_dates = set(pd.to_datetime(holidays).date)

    schedules["holiday_flag"] = schedules["gameday"].dt.date.isin(holiday_dates)

    out = schedules[[
        "season", "week", "home_team", "gameday",
        "day_of_week", "weekend_flag", "holiday_flag"
    ]].copy()

    return out
