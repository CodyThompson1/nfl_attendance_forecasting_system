# Project Summary — NFL Attendance Forecasting System

**Course:** Advanced Applied Modeling  
**Scope:** AFC West + NFC South (8 teams)  
**Status:** Completed

---

## Overview

This project built a complete end-to-end data analytics and machine learning pipeline to forecast NFL game attendance. The goal was to demonstrate the full lifecycle of a data science project — from raw data acquisition through to an interactive, user-facing dashboard — while using a controlled team subset to keep the initial data volume manageable.

The system was designed to be modular and scalable, meaning all components can be extended to cover additional divisions or the full NFL league without architectural changes.

---

## What Was Accomplished

### Phase 1 — Data Engineering

The first phase focused on building a reliable, reproducible data pipeline.

**Data sources integrated:**
- NFL game schedules and outcomes via the `nflreadpy` / nflverse ecosystem
- Historical attendance figures scraped from Pro Football Reference using `BeautifulSoup` and `lxml`
- Game-time weather conditions retrieved from the Meteostat historical weather API

**Key pipeline steps:**
1. Extracted raw schedule, attendance, and weather data independently
2. Cleaned and standardised each dataset (handling nulls, data types, naming conventions)
3. Matched attendance records to game schedule entries using date + team key matching
4. Ran automated data quality checks to flag gaps, duplicates, and outliers

---

### Phase 2 — Database Design

A relational SQLite database was designed using a **star schema** to support flexible querying and easy expansion.

**Dimension tables:** `dim_team`, `dim_venue`, `dim_date`  
**Fact tables:** `fact_game`, `fact_team_form`, `fact_weather`

The schema separates descriptive reference data (teams, venues, dates) from measurable events (game outcomes, attendance, weather conditions), which makes it straightforward to add new data sources without redesigning existing tables.

---

### Phase 3 — Feature Engineering

Before training ML models, a dedicated feature engineering step combined data from across the database into a single ML-ready dataset (`data/modeling/`).

Features included:
- Basic game context (week, season, home/away teams, venue)
- Team form metrics (recent win rate, average points scored/allowed, current streak)
- Weather conditions (temperature, precipitation, wind speed)
- Venue capacity and roof type

---

### Phase 4 — Machine Learning

Three regression models were trained and evaluated to predict attendance:

| Model | Library | Notes |
|---|---|---|
| Linear Regression | scikit-learn | Baseline model, highly interpretable |
| K-Nearest Neighbors Regressor | scikit-learn | Non-parametric, captures local patterns |
| Neural Network (MLP Regressor) | scikit-learn | Multi-layer perceptron for non-linear relationships |

All models were evaluated using RMSE, MAE, and R² on a held-out test set to compare predictive performance.

---

### Phase 5 — Dashboard

A **Streamlit** dashboard was built to present project results interactively. The dashboard includes:

- Attendance trends over time by team and season
- Model predictions vs actual attendance figures
- Weather impact visualisations
- Team-by-team breakdowns and comparisons
- Data quality summary views

The dashboard runs locally using:
```bash
streamlit run scripts/dashboard.py
```

---

## Key Decisions and Design Choices

**Why SQLite?** Lightweight, portable, and sufficient for the data volumes involved. The use of SQLAlchemy means migrating to PostgreSQL in the future would require minimal code changes.

**Why a star schema?** Enables clean separation of concerns, makes aggregation queries efficient, and aligns with industry-standard data warehouse design.

**Why three ML models?** Using Linear Regression as a baseline allowed direct comparison of complexity vs. performance. KNN and Neural Network models tested whether capturing non-linear patterns meaningfully improved predictions.

**Why Streamlit?** Rapid development of interactive dashboards in pure Python, no front-end experience required.

---

## Technologies Used

Python, pandas, numpy, polars, scikit-learn, SQLite, SQLAlchemy, Streamlit, Plotly, Altair, BeautifulSoup, nflreadpy, Meteostat, python-dotenv

---

## Limitations and Future Work

- **Scope:** The current dataset covers 8 teams across 2 divisions. Expanding to the full league would require re-running the pipeline with all 32 teams.
- **Feature depth:** Ticket pricing data, opponent strength of schedule, and local market factors were not included and could improve model accuracy.
- **Model evaluation:** Cross-validation across seasons would give more robust performance estimates than a single train/test split.
- **Deployment:** The dashboard is currently local-only. Deploying to Streamlit Cloud would make it publicly shareable.
