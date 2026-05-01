# NFL Attendance Forecasting System

An end-to-end data analytics and machine learning system for predicting NFL game attendance. Built as part of the Advanced Applied Modeling course, this project covers the full data pipeline — from raw data extraction through to a live interactive dashboard.

---

## Project Overview

This system forecasts NFL game attendance using historical game data, team performance metrics, venue information, and weather conditions. The initial scope covers **8 teams across two divisions** to support rapid development and validation, with an architecture designed to scale to the full league.

**Divisions covered:**
- AFC West: Denver Broncos, Los Angeles Chargers, Kansas City Chiefs, Las Vegas Raiders
- NFC South: Carolina Panthers, Tampa Bay Buccaneers, Atlanta Falcons, New Orleans Saints

---

## What Was Built

### Data Pipeline
- Extracted NFL game schedules and outcomes via the `nflverse` / `nflreadpy` ecosystem
- Scraped attendance figures from Sports Reference using `BeautifulSoup`
- Pulled historical weather data for each game using the `Meteostat` API
- Matched attendance records to game schedule data with fuzzy/exact game matching
- Applied data quality checks throughout the pipeline (`run_data_quality_checks.py`)

### Database
- Designed and implemented a relational SQLite database with a star schema
- Tables: `dim_team`, `dim_venue`, `dim_date`, `fact_game`, `fact_team_form`, `fact_weather`
- Schema supports easy expansion to additional teams, seasons, or data sources

### Feature Engineering
- Built dimensional tables for teams, venues, and dates
- Engineered team form metrics (recent win rate, scoring trends)
- Combined game, weather, and team features into an ML-ready feature set (`build_ml_features_attendance.py`)

### Machine Learning Models
Three regression models were trained and evaluated for attendance prediction:

| Model | Script |
|---|---|
| Linear Regression | `train_linear_regression.py` |
| K-Nearest Neighbors Regressor | `train_knn_regressor.py` |
| Neural Network (MLP) | `train_neural_network.py` |

Models were trained on the engineered feature set and evaluated using standard regression metrics (RMSE, MAE, R²).

### Interactive Dashboard
- Built with **Streamlit** and **Plotly/Altair** for interactive visualisation
- Displays attendance trends, model predictions vs actuals, team breakdowns, and weather impact
- Run locally with: `streamlit run scripts/dashboard.py`

---

## Repository Structure

```
nfl_attendance_forecasting_system/
│
├── data/
│   ├── raw/                  # Raw extracted datasets (schedules, attendance, weather)
│   ├── processed/            # Cleaned and merged datasets ready for modeling
│   └── modeling/             # Final ML-ready feature sets
│
├── db/                       # SQLite database and schema documentation
│
├── docs/                     # Project documentation and deliverables
│
├── scripts/                  # All Python scripts
│   ├── extract_nfl_schedules.py
│   ├── extract_sportsref_attendance.py
│   ├── extract_weather_meteostat.py
│   ├── match_game_attendance.py
│   ├── clean_attendance.py
│   ├── clean_schedules.py
│   ├── run_data_quality_checks.py
│   ├── build_dim_date.py
│   ├── build_dim_team.py
│   ├── build_dim_venue.py
│   ├── build_fact_game.py
│   ├── build_fact_team_form.py
│   ├── build_fact_weather.py
│   ├── build_ml_features_attendance.py
│   ├── train_linear_regression.py
│   ├── train_knn_regressor.py
│   ├── train_neural_network.py
│   └── dashboard.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Data Extraction | `nflreadpy`, `requests`, `BeautifulSoup`, `lxml` |
| Weather Data | `Meteostat` |
| Data Processing | `pandas`, `numpy`, `polars` |
| Database | SQLite via `SQLAlchemy` |
| Machine Learning | `scikit-learn` (LinearRegression, KNeighborsRegressor, MLPRegressor), `scipy` |
| Visualisation | `Streamlit`, `Plotly`, `Altair` |
| Utilities | `python-dotenv`, `GitPython`, `tqdm` |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run scripts/dashboard.py
```

### Running the Full Pipeline
Scripts should be run in the following order:

1. **Extract:** `extract_nfl_schedules.py` → `extract_sportsref_attendance.py` → `extract_weather_meteostat.py`
2. **Clean & Match:** `clean_schedules.py` → `clean_attendance.py` → `match_game_attendance.py`
3. **Quality Check:** `run_data_quality_checks.py`
4. **Build DB Dimensions:** `build_dim_date.py` → `build_dim_team.py` → `build_dim_venue.py`
5. **Build Facts:** `build_fact_game.py` → `build_fact_team_form.py` → `build_fact_weather.py`
6. **Feature Engineering:** `build_ml_features_attendance.py`
7. **Train Models:** `train_linear_regression.py` / `train_knn_regressor.py` / `train_neural_network.py`
8. **Dashboard:** `dashboard.py`

---

## Data Sources

- **NFL Schedules & Game Data:** [nflverse](https://github.com/nflverse) ecosystem via `nflreadpy`
- **Attendance Data:** [Pro Football Reference](https://www.pro-football-reference.com/) (scraped)
- **Weather Data:** [Meteostat](https://meteostat.net/) historical weather API

---

## Course Context

Built for **Advanced Applied Modeling** as a semester-long project demonstrating:
- End-to-end data engineering and pipeline design
- Relational database schema design (star schema)
- Feature engineering for ML
- Comparative model evaluation
- Interactive dashboard delivery
