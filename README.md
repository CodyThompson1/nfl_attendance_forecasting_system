# NFL Attendance Forecasting System

An end-to-end machine learning system for predicting NFL home game attendance using weather conditions, team performance, venue data, and scheduling context. Built as a semester-long project for **Advanced Applied Modeling (BMKT 673)**.

---

## Overview

The system answers one question: *how many fans will attend an upcoming NFL home game, and how confident are we in that estimate?*

It covers the full analytics lifecycle — raw data extraction, a PostgreSQL data warehouse, feature engineering, three trained ML models, and a live interactive dashboard.

---

## Results

| Metric | Value |
|---|---|
| Seasons covered | 2015–2025 (excl. 2020) |
| Total games | 3,028 |
| Best model | Linear Regression |
| Test MAE (271 games, 2025) | 5,290 fans |
| Avg % error per game | 7.5% |
| 95% CI capture rate | 92% |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Extraction | nflreadpy, requests, BeautifulSoup |
| Weather | Meteostat API |
| Data Processing | pandas, numpy |
| Database | PostgreSQL 18, SQLAlchemy, psycopg2 |
| Machine Learning | scikit-learn (LinearRegression, KNeighborsRegressor, MLPRegressor) |
| Dashboard | Streamlit, Plotly |

---

## Data Sources

- **Game Schedules:** nflverse ecosystem via nflreadpy
- **Attendance:** Pro Football Reference (scraped via BeautifulSoup)
- **Weather:** Meteostat historical weather API matched to venue and kickoff time

---

## Database

PostgreSQL star schema with 8 tables. `fact_game` is the central table with foreign keys to `dim_team`, `dim_venue`, and `dim_date`. `fact_weather` and `fact_team_form` extend each game with weather and team performance data. `ml_features` holds the final model-ready feature set and `model_predictions` stores all predictions written back from training.

Six data quality checks are enforced on every pipeline run including attendance validity, no duplicate game IDs, foreign key integrity, and core feature missingness under 5%.

---

## Models

Strict time-based splits were used — no random shuffling — to prevent data leakage:

- **Training:** 2015–2023 (excl. 2020)
- **Validation:** 2024
- **Test:** 2025

| Model | Val MAE | Test MAE |
|---|---|---|
| Linear Regression | 5,021 | 5,290 |
| KNN (k=11) | 5,398 | 5,737 |
| Neural Network (16,8) | 5,348 | 5,505 |

Linear Regression was selected based on lowest validation MAE and best generalization to the 2025 test set. All predictions are written back to PostgreSQL.

---

## Dashboard

Three-page Streamlit dashboard connected directly to PostgreSQL:

- **Executive Forecast** — game-by-game predictions, confidence intervals, weather risk flags, model comparison
- **Attendance Drivers** — weather and win % scatter plots, day-of-week heatmap, game context breakdowns
- **Scenario Simulator** — interactive sliders and toggles that generate live attendance predictions using the trained model

---

## Setup

**Prerequisites:** Python 3.10+, PostgreSQL with pgAdmin 4

```bash
pip install -r requirements.txt
```

Set up the database by running `setup_postgres_db.sql` in the pgAdmin Query Tool, updating the file paths inside to match your local data directory.

Update the credentials in `scripts/dashboard.py`:

```python
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "nfl_attendance"
DB_USER     = "postgres"
DB_PASSWORD = "your_password_here"
```

Run the dashboard:

```bash
streamlit run scripts/dashboard.py
```

---

## Pipeline Order

1. `extract_nfl_schedules.py`
2. `extract_sportsref_attendance.py`
3. `extract_weather_meteostat.py`
4. `clean_schedules.py` / `clean_attendance.py` / `match_game_attendance.py`
5. `run_data_quality_checks.py`
6. `build_dim_*.py` / `build_fact_*.py`
7. `build_ml_features_attendance.py`
8. `train_linear_regression.py` / `train_knn_regressor.py` / `train_neural_network.py`
9. `dashboard.py`

---

## Course Context

Built for **Advanced Applied Modeling (BMKT 673)** demonstrating end-to-end data engineering, relational database design, feature engineering for ML, comparative model evaluation with time-based validation, and interactive dashboard delivery.
