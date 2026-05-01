# Database Schema

This folder contains the SQLite database for the NFL Attendance Forecasting System. The database uses a **star schema** design, with dimension tables for reference data and fact tables for measurable events.

---

## Schema Overview

### Dimension Tables

**`dim_team`**
Stores metadata for each NFL team in scope.

| Column | Type | Description |
|---|---|---|
| team_id | INTEGER PK | Unique team identifier |
| team_abbr | TEXT | Team abbreviation (e.g. KC, DEN) |
| team_name | TEXT | Full team name |
| conference | TEXT | AFC or NFC |
| division | TEXT | Division name (e.g. AFC West) |
| stadium_name | TEXT | Home stadium name |
| city | TEXT | Home city |

---

**`dim_venue`**
Stores venue/stadium information used for game context.

| Column | Type | Description |
|---|---|---|
| venue_id | INTEGER PK | Unique venue identifier |
| venue_name | TEXT | Stadium name |
| city | TEXT | City |
| state | TEXT | State |
| capacity | INTEGER | Stadium seating capacity |
| surface | TEXT | Playing surface type |
| roof_type | TEXT | Open, retractable, or dome |

---

**`dim_date`**
Date dimension for time-based analysis.

| Column | Type | Description |
|---|---|---|
| date_id | INTEGER PK | Surrogate key |
| full_date | DATE | Full calendar date |
| year | INTEGER | Calendar year |
| month | INTEGER | Month number |
| day_of_week | INTEGER | Day of week (0=Mon) |
| week_of_season | INTEGER | NFL week number |
| season | INTEGER | NFL season year |

---

### Fact Tables

**`fact_game`**
Core game-level data including attendance and outcome.

| Column | Type | Description |
|---|---|---|
| game_id | INTEGER PK | Unique game identifier |
| date_id | INTEGER FK | Reference to dim_date |
| home_team_id | INTEGER FK | Reference to dim_team |
| away_team_id | INTEGER FK | Reference to dim_team |
| venue_id | INTEGER FK | Reference to dim_venue |
| attendance | INTEGER | Actual recorded attendance |
| home_score | INTEGER | Home team final score |
| away_score | INTEGER | Away team final score |
| season | INTEGER | NFL season year |
| week | INTEGER | NFL week number |
| game_type | TEXT | Regular season / Playoffs |

---

**`fact_team_form`**
Rolling team performance metrics leading into each game.

| Column | Type | Description |
|---|---|---|
| form_id | INTEGER PK | Unique record identifier |
| game_id | INTEGER FK | Reference to fact_game |
| team_id | INTEGER FK | Reference to dim_team |
| recent_win_rate | REAL | Win rate over last N games |
| avg_points_scored | REAL | Avg points scored recently |
| avg_points_allowed | REAL | Avg points allowed recently |
| streak | INTEGER | Current win/loss streak |

---

**`fact_weather`**
Weather conditions at game time for each venue.

| Column | Type | Description |
|---|---|---|
| weather_id | INTEGER PK | Unique record identifier |
| game_id | INTEGER FK | Reference to fact_game |
| temperature_c | REAL | Temperature in Celsius |
| precipitation_mm | REAL | Precipitation in mm |
| wind_speed_kmh | REAL | Wind speed in km/h |
| conditions | TEXT | Weather condition description |

---

## Notes

- The database is implemented as a local **SQLite** file (excluded from version control via `.gitignore`)
- All tables are built via the `scripts/build_*.py` scripts
- The schema is intentionally designed for easy expansion to additional teams and seasons
