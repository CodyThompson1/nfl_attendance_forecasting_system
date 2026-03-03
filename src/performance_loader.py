import pandas as pd
import numpy as np
import nflreadpy as nfl

from .config import ALL_YEARS, SCOPE_TEAM_ABBRS


def _build_team_game_log(schedules: pd.DataFrame, team_abbrs: set[str]) -> pd.DataFrame:
    """
    Converts game-level schedules into a team-game log (two rows per game: one per team).

    Returns columns:
        season, week, gameday, team, opponent, is_home, points_for, points_against, win
    """
    keep_cols = [
        "season", "week", "gameday",
        "home_team", "away_team",
        "home_score", "away_score",
        "game_type",
    ]
    schedules = schedules[[c for c in keep_cols if c in schedules.columns]].copy()
    schedules = schedules[schedules["game_type"] == "REG"].copy()

    # Home rows
    home = schedules.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["is_home"] = True
    home["points_for"] = home["home_score"]
    home["points_against"] = home["away_score"]

    # Away rows
    away = schedules.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["is_home"] = False
    away["points_for"] = away["away_score"]
    away["points_against"] = away["home_score"]

    team_log = pd.concat([home, away], ignore_index=True)

    # Filter to scope teams
    team_log = team_log[team_log["team"].isin(team_abbrs)].copy()

    # Win indicator (ties treated as 0.5)
    team_log["win"] = np.where(
        team_log["points_for"] > team_log["points_against"], 1.0,
        np.where(team_log["points_for"] < team_log["points_against"], 0.0, 0.5)
    )

    # Sort for rolling calcs
    team_log["gameday"] = pd.to_datetime(team_log["gameday"])
    team_log = team_log.sort_values(["team", "season", "week", "gameday"]).reset_index(drop=True)

    return team_log


def load_team_performance() -> pd.DataFrame:
    """
    Data Source #2: Team performance / quality (derived)

    Source:
        nflverse schedules/results via nflreadpy

    Outputs (team-week level):
        season, week, team,
        win_pct_to_date, rolling_5_win_pct,
        rolling_5_point_diff
    """
    schedules = nfl.load_schedules(ALL_YEARS).to_pandas()

    team_log = _build_team_game_log(schedules, SCOPE_TEAM_ABBRS)

    # Point differential
    team_log["point_diff"] = team_log["points_for"] - team_log["points_against"]

    # Cumulative win pct to date (through that game week)
    team_log["games_played"] = team_log.groupby(["team", "season"]).cumcount() + 1
    team_log["wins_cum"] = team_log.groupby(["team", "season"])["win"].cumsum()
    team_log["win_pct_to_date"] = team_log["wins_cum"] / team_log["games_played"]

    # Rolling 5 metrics (within season)
    team_log["rolling_5_win_pct"] = (
        team_log.groupby(["team", "season"])["win"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    team_log["rolling_5_point_diff"] = (
        team_log.groupby(["team", "season"])["point_diff"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # If multiple rows per week ever exist take last in week
    out = (
        team_log.sort_values(["team", "season", "week", "gameday"])
        .groupby(["season", "week", "team"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    out = out[[
        "season", "week", "team",
        "win_pct_to_date", "rolling_5_win_pct", "rolling_5_point_diff"
    ]].copy()

    return out
