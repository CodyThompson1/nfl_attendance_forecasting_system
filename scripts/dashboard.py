"""
NFL Attendance Forecasting Dashboard
Install:  pip install streamlit pandas sqlalchemy psycopg2-binary plotly numpy scikit-learn
Run:      streamlit run dashboard.py
"""

import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DB_HOST     = "localhost"
DB_PORT     = 5432
DB_NAME     = "nfl_attendance"
DB_USER     = "postgres"
DB_PASSWORD = "GoGriz18318!!"

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Attendance Forecast",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Overall background */
    [data-testid="stAppViewContainer"] { background-color: #f4f6f9; }
    [data-testid="stSidebar"]          { background-color: #1a2744; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div  { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3  { color: #ffffff !important; }
    [data-testid="stSidebar"] hr  { border-color: #2d4a7a; }
    [data-testid="stSidebar"] .stRadio > div { gap: 4px; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .kpi-card.green  { border-left-color: #16a34a; }
    .kpi-card.amber  { border-left-color: #d97706; }
    .kpi-card.red    { border-left-color: #dc2626; }
    .kpi-card.purple { border-left-color: #7c3aed; }

    .kpi-label { font-size: 0.7rem; color: #6b7280; text-transform: uppercase;
                 letter-spacing: 0.06em; font-weight: 600; margin-bottom: 4px; }
    .kpi-value { font-size: 1.75rem; font-weight: 800; color: #111827; line-height: 1.1; }
    .kpi-sub   { font-size: 0.68rem; color: #9ca3af; margin-top: 3px; }
    .kpi-value.red    { color: #dc2626; }
    .kpi-value.green  { color: #16a34a; }
    .kpi-value.amber  { color: #d97706; }

    /* Section headers */
    .sec-header {
        font-size: 1rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding: 4px 0 4px 10px;
        margin: 1.4rem 0 0.8rem 0;
        background: #eff6ff;
        border-radius: 0 6px 6px 0;
    }

    /* Page title */
    h1 { color: #1e3a5f !important; }
    p, li { color: #374151; }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

PLOT_BG  = "#ffffff"
PLOT_THM = "plotly_white"
C_BLUE   = "#2563eb"
C_GREEN  = "#16a34a"
C_RED    = "#dc2626"
C_AMBER  = "#d97706"
C_PURPLE = "#7c3aed"
C_TEAL   = "#0891b2"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        res = conn.execute(text(sql), params or {})
        return pd.DataFrame(res.fetchall(), columns=res.keys())

def safe_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        return query(sql, params)
    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Check PostgreSQL is running and DB_PASSWORD is correct.")
        st.stop()

def fmt(val, decimals=0) -> str:
    if val is None or (isinstance(val, float) and np.isnan(float(val))):
        return "N/A"
    return f"{float(val):,.{decimals}f}"

def kpi_card(label, value, sub="", color="blue"):
    val_cls  = color if color in ("red","green","amber") else ""
    card_cls = color if color in ("green","amber","red","purple") else ""
    st.markdown(
        f'<div class="kpi-card {card_cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {val_cls}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

def section(title: str):
    st.markdown(f'<div class="sec-header">{title}</div>', unsafe_allow_html=True)

def cast_bool(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.lower()
                     .map({"true":True,"false":False,"1":True,"0":False})
                     .fillna(False))
    return df


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast decimal.Decimal columns (PostgreSQL NUMERIC) to float.
    Compatible with all pandas versions."""
    import decimal
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna()
            if sample.empty:
                continue
            first = sample.iloc[0]
            if isinstance(first, decimal.Decimal):
                df[col] = df[col].apply(
                    lambda x: float(x) if isinstance(x, decimal.Decimal) else x
                )
            elif isinstance(first, str):
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() >= len(df[col].dropna()) * 0.5:
                    df[col] = converted
    return df

def add_trend(fig, x_ser, y_ser, color=C_AMBER, name="Trend"):
    """Add a manual OLS trendline — no statsmodels needed."""
    mask = x_ser.notna() & y_ser.notna()
    xv, yv = x_ser[mask].astype(float).values, y_ser[mask].astype(float).values
    if len(xv) < 3:
        return fig
    m, b = np.polyfit(xv, yv, 1)
    xl = np.linspace(xv.min(), xv.max(), 100)
    fig.add_trace(go.Scatter(x=xl, y=m*xl+b, mode="lines", name=name,
        line=dict(color=color, width=2, dash="dash"), showlegend=True))
    return fig

def chart_layout(fig, height=340):
    fig.update_layout(
        height=height, template=PLOT_THM,
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAINING DATA ONCE — for simulator regression
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_training_data() -> pd.DataFrame:
    sql = """
        SELECT
            attendance,
            COALESCE(temperature, 65)           AS temperature,
            COALESCE(precipitation, 0)          AS precipitation,
            COALESCE(wind_speed, 8)             AS wind_speed,
            COALESCE(home_team_win_pct, 0.5)    AS home_team_win_pct,
            COALESCE(away_team_win_pct, 0.5)    AS away_team_win_pct,
            CASE WHEN indoor_flag       THEN 1 ELSE 0 END AS indoor_flag,
            CASE WHEN weekend_flag      THEN 1 ELSE 0 END AS weekend_flag,
            CASE WHEN primetime_flag    THEN 1 ELSE 0 END AS primetime_flag,
            CASE WHEN divisional_game_flag THEN 1 ELSE 0 END AS divisional_game_flag,
            CASE WHEN rivalry_flag      THEN 1 ELSE 0 END AS rivalry_flag,
            CASE WHEN holiday_flag      THEN 1 ELSE 0 END AS holiday_flag,
            CASE WHEN neutral_site_flag THEN 1 ELSE 0 END AS neutral_site_flag,
            CASE WHEN international_flag THEN 1 ELSE 0 END AS international_flag,
            COALESCE(home_rest_days, 7)         AS home_rest_days,
            COALESCE(away_rest_days, 7)         AS away_rest_days,
            COALESCE(week_of_season, 8)         AS week_of_season
        FROM ml_features
        WHERE season NOT IN (2020, 2025)
          AND attendance IS NOT NULL
          AND game_type = 'REG'
    """
    return cast_numeric(safe_query(sql))

@st.cache_data(ttl=600, show_spinner=False)
def fit_simulator_model():
    """Fit a real LinearRegression on the training data for the simulator."""
    df = load_training_data()
    if df.empty:
        return None, 65000.0, 7100.0

    feature_cols = [
        "temperature","precipitation","wind_speed",
        "home_team_win_pct","away_team_win_pct",
        "indoor_flag","weekend_flag","primetime_flag",
        "divisional_game_flag","rivalry_flag","holiday_flag",
        "neutral_site_flag","international_flag",
        "home_rest_days","away_rest_days","week_of_season",
    ]
    X = df[feature_cols].astype(float).values
    y = df["attendance"].astype(float).values

    model = LinearRegression()
    model.fit(X, y)

    preds    = model.predict(X)
    residuals = y - preds
    res_std  = float(np.std(residuals, ddof=1))
    avg_att  = float(y.mean())

    return model, avg_att, res_std

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏈 NFL Attendance")
    st.markdown("*End-to-End Forecasting System*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Executive Forecast", "📈 Attendance Drivers", "🎛️ Scenario Simulator"],
        label_visibility="collapsed",
    )
    st.divider()

    # Season filter
    seasons_df = cast_numeric(safe_query("SELECT DISTINCT season FROM fact_game ORDER BY season DESC"))
    season_list = seasons_df["season"].tolist() if not seasons_df.empty else [2025]
    selected_season = st.selectbox("📅 Season", season_list, index=0)

    # Model filter
    models_df = cast_numeric(safe_query("SELECT DISTINCT model_name FROM model_predictions ORDER BY model_name"))
    model_list = models_df["model_name"].tolist() if not models_df.empty else ["linear_regression"]
    selected_model = st.selectbox("🤖 Model", model_list, index=0)

    # Team filter
    teams_df = cast_numeric(safe_query("SELECT team_abbr, team_name FROM dim_team ORDER BY team_name"))
    team_opts  = ["All Teams"] + teams_df["team_name"].tolist() if not teams_df.empty else ["All Teams"]
    selected_team = st.selectbox("🏟️ Team (Home)", team_opts, index=0)

    st.divider()
    refresh_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"⏱ Refreshed: {refresh_ts}")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("BMKT 673 · NFL Attendance Forecasting")


# ─────────────────────────────────────────────────────────────────────────────
# TEAM FILTER HELPER
# ─────────────────────────────────────────────────────────────────────────────
team_filter_sql = ""
team_filter_params: dict = {"season": selected_season, "model": selected_model}

if selected_team != "All Teams":
    team_filter_sql = "AND ht.team_name = :team_name"
    team_filter_params["team_name"] = selected_team


# =============================================================================
# PAGE 1 — EXECUTIVE FORECAST VIEW
# =============================================================================
if page == "📊 Executive Forecast":

    st.title("📊 Executive Forecast View")
    team_label = f" · {selected_team}" if selected_team != "All Teams" else ""
    st.caption(
        f"Season **{selected_season}** · Model: **{selected_model}**"
        f"{team_label} · Last refresh: {refresh_ts}"
    )

    # ── KPIs ─────────────────────────────────────────────────────────────────
    kpi_sql = f"""
        SELECT
            COUNT(DISTINCT fg.game_id)                                              AS total_games,
            COUNT(DISTINCT mp.game_id)                                              AS predicted_games,
            ROUND(AVG(mp.predicted_attendance))                                     AS avg_predicted,
            ROUND(AVG(fg.attendance))                                               AS avg_actual,
            ROUND(AVG((mp.prediction_upper - mp.prediction_lower) / 2.0))          AS avg_half_ci,
            COALESCE(SUM(CASE WHEN fw.severe_weather_flag THEN 1 ELSE 0 END), 0)   AS weather_risk_count,
            ROUND(AVG(
                CASE WHEN fg.attendance IS NOT NULL AND mp.predicted_attendance IS NOT NULL
                     THEN ABS(fg.attendance - mp.predicted_attendance) END
            ))                                                                      AS avg_mae
        FROM fact_game fg
        JOIN dim_date  dd  ON dd.date_id   = fg.date_id
        JOIN dim_team  ht  ON ht.team_id   = fg.home_team_id
        JOIN dim_team  at_ ON at_.team_id  = fg.away_team_id
        JOIN dim_venue dv  ON dv.venue_id  = fg.venue_id
        LEFT JOIN model_predictions mp ON mp.game_id = fg.game_id
                                       AND mp.model_name = :model
        LEFT JOIN fact_weather fw ON fw.game_id = fg.game_id
        WHERE fg.season = :season AND fg.game_type = 'REG'
        {team_filter_sql}
    """
    k = cast_numeric(safe_query(kpi_sql, team_filter_params)).iloc[0]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi_card("Games This Season",     fmt(k["total_games"]),       "Regular season",      "blue")
    with c2: kpi_card("Games w/ Predictions",  fmt(k["predicted_games"]),   selected_model,        "blue")
    with c3: kpi_card("Avg Predicted",          fmt(k["avg_predicted"]),     "Attendance",          "blue")
    with c4: kpi_card("Avg Actual",             fmt(k["avg_actual"]),        "Attendance",          "green")
    with c5: kpi_card("Avg ± CI (95%)",         f'±{fmt(k["avg_half_ci"])}', "Half-interval width", "amber")
    risk = int(k["weather_risk_count"] or 0)
    with c6: kpi_card("⚠️ Weather Risk Games",  str(risk), "Severe conditions",
                      "red" if risk > 0 else "green")

    st.divider()

    # ── Game Table ────────────────────────────────────────────────────────────
    section("🗓️ Game-by-Game Predictions")

    games_sql = f"""
        SELECT
            fg.week,
            dd.date,
            dd.day_of_week                                                AS day,
            ht.team_name                                                  AS home_team,
            at_.team_name                                                 AS away_team,
            dv.venue_name,
            CASE WHEN dv.indoor_flag THEN '🏟 Indoor' ELSE '🌤 Outdoor' END AS roof,
            fg.attendance                                                  AS actual,
            ROUND(mp.predicted_attendance)                                 AS predicted,
            ROUND(mp.prediction_lower)                                     AS lower_95,
            ROUND(mp.prediction_upper)                                     AS upper_95,
            CASE WHEN fw.severe_weather_flag THEN '⚠️ Yes' ELSE '✅ No'   END AS risk,
            COALESCE(fw.weather_condition, 'N/A')                          AS condition,
            ROUND(fw.temperature::numeric,1)                               AS temp_f,
            ROUND(fw.precipitation::numeric,3)                             AS precip_in,
            ROUND(fw.wind_speed::numeric,1)                                AS wind_mph,
            mp.dataset_split                                               AS split
        FROM fact_game fg
        JOIN dim_date  dd  ON dd.date_id   = fg.date_id
        JOIN dim_team  ht  ON ht.team_id   = fg.home_team_id
        JOIN dim_team  at_ ON at_.team_id  = fg.away_team_id
        JOIN dim_venue dv  ON dv.venue_id  = fg.venue_id
        LEFT JOIN fact_weather      fw ON fw.game_id = fg.game_id
        LEFT JOIN model_predictions mp ON mp.game_id = fg.game_id
                                       AND mp.model_name = :model
        WHERE fg.season = :season AND fg.game_type = 'REG'
        {team_filter_sql}
        ORDER BY fg.week, dd.date
    """
    games_df = cast_numeric(safe_query(games_sql, team_filter_params))

    if games_df.empty:
        st.warning("No data found. Confirm the SQL setup and fix_views scripts have been run.")
    else:
        fa, fb = st.columns([3, 1])
        weeks = sorted(games_df["week"].dropna().unique().tolist())
        # Key includes season + team so multiselect resets whenever either changes
        week_key = f"wk_{selected_season}_{selected_team}"
        with fa:
            sel_wks = st.multiselect(
                "Filter by Week", options=weeks, default=weeks, key=week_key
            )
        with fb:
            split_opts = ["All"] + sorted(games_df["split"].dropna().unique().tolist())
            sel_split  = st.selectbox("Dataset Split", split_opts)

        view = games_df[games_df["week"].isin(sel_wks)].copy()
        if sel_split != "All":
            view = view[view["split"] == sel_split]

        st.dataframe(
            view.rename(columns={
                "week":"Wk","date":"Date","day":"Day",
                "home_team":"Home Team","away_team":"Away Team",
                "venue_name":"Venue","roof":"Roof",
                "actual":"Actual","predicted":"Predicted",
                "lower_95":"Lower 95%","upper_95":"Upper 95%",
                "risk":"⚠️ Risk","condition":"Condition",
                "temp_f":"Temp °F","precip_in":"Precip (in)","wind_mph":"Wind (mph)",
                "split":"Split",
            }),
            use_container_width=True, hide_index=True, height=380,
        )

        # Summary stats below table — force float to avoid decimal.Decimal issues
        if view["predicted"].notna().any():
            actual    = pd.to_numeric(view["actual"],    errors="coerce")
            predicted = pd.to_numeric(view["predicted"], errors="coerce")
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Rows Shown",      f"{len(view):,}")
            s2.metric("Avg Predicted",   f"{int(predicted.mean()):,}" if predicted.notna().any() else "N/A")
            s3.metric("Avg Actual",      f"{int(actual.mean()):,}"    if actual.notna().any()    else "N/A")
            mae_vals = (actual - predicted).abs().dropna()
            s4.metric("Avg Error (MAE)", f"{int(mae_vals.mean()):,}"  if not mae_vals.empty      else "N/A")

    st.divider()

    # ── Forecast Chart ────────────────────────────────────────────────────────
    section("📈 Predicted vs Actual Attendance — with 95% Confidence Intervals")

    if not games_df.empty and games_df["predicted"].notna().any():
        chart = games_df.dropna(subset=["predicted"]).copy().reset_index(drop=True)
        chart["label"] = "Wk" + chart["week"].astype(str) + " · " + chart["home_team"]
        labels = chart["label"].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=labels + list(reversed(labels)),
            y=chart["upper_95"].tolist() + list(reversed(chart["lower_95"].tolist())),
            fill="toself", fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence Interval", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=chart["predicted"], mode="lines+markers",
            name="Predicted", line=dict(color=C_BLUE, width=2), marker=dict(size=5),
        ))
        if chart["actual"].notna().any():
            fig.add_trace(go.Scatter(
                x=labels, y=chart["actual"], mode="lines+markers",
                name="Actual", line=dict(color=C_GREEN, width=2, dash="dot"),
                marker=dict(size=5),
            ))
        fig = chart_layout(fig, height=420)
        fig.update_layout(
            xaxis=dict(tickangle=-50, tickfont=dict(size=9), title=""),
            yaxis=dict(tickformat=",", title="Attendance"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Model Comparison ─────────────────────────────────────────────────────
    section("📐 Model Performance Comparison (All 3 Models)")
    st.caption("MAE = Mean Absolute Error · RMSE = Root Mean Squared Error · Lower is better for MAE/RMSE")

    perf_sql = """
        SELECT
            model_name                                                              AS "Model",
            dataset_split                                                           AS "Split",
            COUNT(*)                                                                AS "Games",
            TO_CHAR(ROUND(AVG(ABS(actual_attendance - predicted_attendance))), 'FM999,999')  AS "MAE",
            TO_CHAR(ROUND(SQRT(AVG(POWER(actual_attendance - predicted_attendance,2)))), 'FM999,999') AS "RMSE"
        FROM model_predictions
        WHERE actual_attendance IS NOT NULL AND predicted_attendance IS NOT NULL
        GROUP BY model_name, dataset_split
        ORDER BY model_name,
                 CASE dataset_split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END
    """
    perf_df = cast_numeric(safe_query(perf_sql))
    if not perf_df.empty:
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Bar chart of MAE by model + split
        bar_sql = """
            SELECT model_name AS model, dataset_split AS split,
                   ROUND(AVG(ABS(actual_attendance - predicted_attendance))) AS mae
            FROM model_predictions
            WHERE actual_attendance IS NOT NULL AND predicted_attendance IS NOT NULL
            GROUP BY model_name, dataset_split
        """
        bar_df = cast_numeric(safe_query(bar_sql))
        if not bar_df.empty:
            fig_bar = px.bar(bar_df, x="model", y="mae", color="split", barmode="group",
                labels={"model":"Model","mae":"MAE (avg error)","split":"Dataset Split"},
                color_discrete_sequence=[C_BLUE, C_AMBER, C_GREEN],
                template=PLOT_THM, text_auto=",.0f")
            fig_bar = chart_layout(fig_bar, height=320)
            fig_bar.update_layout(yaxis=dict(tickformat=",", title="Mean Absolute Error"),
                                  xaxis_title="")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# PAGE 2 — ATTENDANCE DRIVERS
# =============================================================================
elif page == "📈 Attendance Drivers":

    st.title("📈 Attendance Drivers")
    team_label = f" · {selected_team}" if selected_team != "All Teams" else " · All Teams"
    st.caption(
        f"What drives NFL home game attendance? · "
        f"Season **{selected_season}**{team_label} · Regular season only"
    )

    drv_sql = f"""
        SELECT * FROM vw_attendance_drivers
        WHERE season = :season AND attendance IS NOT NULL
        {("AND home_team = :team_name" if selected_team != "All Teams" else "")}
        ORDER BY week
    """
    drv_params = {"season": selected_season}
    if selected_team != "All Teams":
        drv_params["team_name"] = selected_team

    df = cast_numeric(safe_query(drv_sql, drv_params))

    if df.empty:
        st.warning("No data found. Try selecting a different season or team.")
        st.stop()

    bool_cols = ["severe_weather_flag","indoor_flag","weekend_flag","holiday_flag",
                 "primetime_flag","divisional_game_flag","rivalry_flag",
                 "neutral_site_flag","international_flag"]
    df = cast_bool(df, bool_cols)

    # ── Quick summary metrics ─────────────────────────────────────────────────
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: kpi_card("Games Analyzed",    f"{len(df):,}",                              "This filter",      "blue")
    with m2: kpi_card("Avg Attendance",    f"{int(df['attendance'].mean()):,}",          "Mean per game",    "blue")
    with m3: kpi_card("Max Attendance",    f"{int(df['attendance'].max()):,}",           "Sellout games",    "green")
    with m4: kpi_card("Min Attendance",    f"{int(df['attendance'].min()):,}",           "Lowest game",      "amber")
    with m5: kpi_card("Severe Wx Games",   f"{int(df['severe_weather_flag'].sum())}",    "Weather risk",     "red")

    st.divider()

    # ── WEATHER ───────────────────────────────────────────────────────────────
    section("🌦️ Weather vs Attendance")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Temperature vs Attendance**")
        tdf = df.dropna(subset=["temperature","attendance"]).copy()
        tdf["Weather"] = tdf["severe_weather_flag"].map({True:"⚠️ Severe", False:"✅ Normal"})
        if not tdf.empty:
            fig = px.scatter(tdf, x="temperature", y="attendance",
                color="Weather",
                color_discrete_map={"⚠️ Severe": C_RED, "✅ Normal": C_BLUE},
                hover_data={"home_team":True, "week":True,
                            "weather_condition":True, "Weather":False},
                labels={"temperature":"Temperature (°F)","attendance":"Attendance",
                        "home_team":"Home","week":"Week",
                        "weather_condition":"Condition"},
                template=PLOT_THM, opacity=0.65)
            fig = add_trend(fig, tdf["temperature"], tdf["attendance"], C_AMBER, "OLS Trend")
            fig = chart_layout(fig, height=350)
            fig.update_layout(yaxis=dict(tickformat=",", title="Attendance"),
                              xaxis_title="Temperature (°F)")
            st.plotly_chart(fig, use_container_width=True)
            # Insight
            corr = tdf["temperature"].corr(tdf["attendance"])
            st.markdown(
                f'<div class="info-box">📊 Correlation between temperature and attendance: '
                f'<b>r = {corr:.3f}</b>. '
                f'{"Warmer games tend to draw larger crowds." if corr > 0 else "Colder games tend to draw larger crowds."}'
                f'</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("**Precipitation vs Attendance**")
        pdf = df.dropna(subset=["precipitation","attendance"]).copy()
        pdf["Venue Type"] = pdf["indoor_flag"].map({True:"🏟 Indoor", False:"🌤 Outdoor"})
        if not pdf.empty:
            fig = px.scatter(pdf, x="precipitation", y="attendance",
                color="Venue Type",
                color_discrete_map={"🏟 Indoor": C_PURPLE, "🌤 Outdoor": C_TEAL},
                hover_data={"home_team":True, "week":True, "Venue Type":False},
                labels={"precipitation":"Precipitation (in)","attendance":"Attendance",
                        "home_team":"Home","week":"Week"},
                template=PLOT_THM, opacity=0.65)
            fig = add_trend(fig, pdf["precipitation"], pdf["attendance"], C_RED, "OLS Trend")
            fig = chart_layout(fig, height=350)
            fig.update_layout(yaxis=dict(tickformat=",", title="Attendance"),
                              xaxis_title="Precipitation (in)")
            st.plotly_chart(fig, use_container_width=True)
            corr2 = pdf["precipitation"].corr(pdf["attendance"])
            st.markdown(
                f'<div class="info-box">📊 Correlation between precipitation and attendance: '
                f'<b>r = {corr2:.3f}</b>. '
                f'Rain tends to reduce attendance for outdoor games.</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── TEAM PERFORMANCE ──────────────────────────────────────────────────────
    section("🏆 Team Performance vs Attendance")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Home Team Win % vs Attendance**")
        wdf = df.dropna(subset=["home_team_win_pct","attendance"]).copy()
        if not wdf.empty:
            fig = px.scatter(wdf, x="home_team_win_pct", y="attendance",
                color="home_team",
                hover_data={"week":True, "home_team":False, "away_team":True,
                            "home_team_win_pct":False, "attendance":False},
                labels={"home_team_win_pct":"Home Win %",
                        "attendance":"Attendance",
                        "home_team":"Team",
                        "away_team":"Away",
                        "week":"Week"},
                template=PLOT_THM, opacity=0.7)
            fig = add_trend(fig, wdf["home_team_win_pct"], wdf["attendance"], C_RED, "OLS Trend")
            fig = chart_layout(fig, height=360)
            fig.update_layout(showlegend=False,
                              yaxis=dict(tickformat=",", title="Attendance"),
                              xaxis=dict(tickformat=".0%", title="Home Win %"))
            st.plotly_chart(fig, use_container_width=True)
            corr3 = wdf["home_team_win_pct"].corr(wdf["attendance"])
            st.markdown(
                f'<div class="info-box">📊 Correlation between home win % and attendance: '
                f'<b>r = {corr3:.3f}</b>. '
                f'Better teams consistently draw more fans.</div>',
                unsafe_allow_html=True)

    with c4:
        st.markdown("**Away Team Win % vs Attendance**")
        adf = df.dropna(subset=["away_team_win_pct","attendance"]).copy()
        if not adf.empty:
            fig = px.scatter(adf, x="away_team_win_pct", y="attendance",
                color="home_team",
                hover_data={"week":True, "home_team":False, "away_team":True,
                            "away_team_win_pct":False, "attendance":False},
                labels={"away_team_win_pct":"Away Win %",
                        "attendance":"Attendance",
                        "home_team":"Home Team",
                        "away_team":"Away Team",
                        "week":"Week"},
                template=PLOT_THM, opacity=0.7)
            fig = add_trend(fig, adf["away_team_win_pct"], adf["attendance"], C_RED, "OLS Trend")
            fig = chart_layout(fig, height=360)
            fig.update_layout(showlegend=False,
                              yaxis=dict(tickformat=",", title="Attendance"),
                              xaxis=dict(tickformat=".0%", title="Away Win %"))
            st.plotly_chart(fig, use_container_width=True)
            corr4 = adf["away_team_win_pct"].corr(adf["attendance"])
            st.markdown(
                f'<div class="info-box">📊 Correlation between away win % and attendance: '
                f'<b>r = {corr4:.3f}</b>. '
                f'High-profile away teams attract larger crowds.</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── DAY OF WEEK HEATMAP ───────────────────────────────────────────────────
    section("📅 Day-of-Week Attendance Heatmap")
    st.caption("Average attendance per team per day of week — darker = higher attendance")

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    hdf = df.dropna(subset=["day_of_week","attendance"])
    if not hdf.empty:
        pivot = (
            hdf.groupby(["home_team","day_of_week"])["attendance"].mean().reset_index()
            .pivot(index="home_team", columns="day_of_week", values="attendance")
            .reindex(columns=[d for d in dow_order if d in hdf["day_of_week"].unique()])
        )
        fig_heat = px.imshow(pivot,
            labels=dict(x="Day of Week", y="Home Team", color="Avg Attendance"),
            color_continuous_scale=[[0,"#dbeafe"],[0.5,"#3b82f6"],[1,"#1e3a8a"]],
            template=PLOT_THM, aspect="auto", text_auto=",.0f")
        fig_heat.update_layout(
            height=max(400, len(pivot) * 22),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            coloraxis_colorbar=dict(tickformat=","),
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="",
            font=dict(size=10),
        )
        fig_heat.update_traces(textfont=dict(size=9))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── GAME CONTEXT BARS ────────────────────────────────────────────────────
    section("🏟️ Game Context — Average Attendance by Category")
    st.caption("How different game types affect average attendance")

    def context_bar(col, label_map, colors, title, col_widget):
        d = (df.groupby(df[col].map(label_map))["attendance"]
             .agg(["mean","count"]).reset_index())
        d.columns = ["Category","Avg Attendance","Games"]
        d = d.dropna(subset=["Category"])
        fig = px.bar(d, x="Category", y="Avg Attendance",
            color="Category", color_discrete_sequence=colors,
            template=PLOT_THM, title=title, text="Games",
            labels={"Avg Attendance":"Avg Attendance","Category":""})
        fig.update_traces(
            texttemplate="n=%{text}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg Attendance: %{y:,.0f}<br>Games: %{text}<extra></extra>",
        )
        fig.update_layout(height=290, showlegend=False,
                          yaxis=dict(tickformat=",", title=""),
                          xaxis_title="",
                          plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                          margin=dict(t=40, b=20, l=10, r=10))
        col_widget.plotly_chart(fig, use_container_width=True)

    ca, cb, cc = st.columns(3)
    context_bar("primetime_flag",
                {True:"📺 Primetime", False:"Standard"},
                [C_BLUE,"#93c5fd"], "Primetime vs Standard", ca)
    context_bar("divisional_game_flag",
                {True:"⚔️ Divisional", False:"Non-Divisional"},
                [C_GREEN,"#86efac"], "Divisional Games", cb)
    context_bar("weekend_flag",
                {True:"📅 Weekend", False:"Weekday"},
                [C_PURPLE,"#c4b5fd"], "Weekend vs Weekday", cc)

    cd, ce, cf = st.columns(3)
    context_bar("rivalry_flag",
                {True:"🔥 Rivalry", False:"Non-Rivalry"},
                [C_RED,"#fca5a5"], "Rivalry Games", cd)
    context_bar("indoor_flag",
                {True:"🏟 Indoor", False:"🌤 Outdoor"},
                [C_TEAL,"#a5f3fc"], "Indoor vs Outdoor", ce)
    context_bar("holiday_flag",
                {True:"🦃 Holiday", False:"Non-Holiday"},
                [C_AMBER,"#fde68a"], "Holiday Games", cf)

    st.divider()

    # ── TEAM ATTENDANCE RANKING ──────────────────────────────────────────────
    section("🏈 Average Attendance by Home Team")
    st.caption("Ranked from highest to lowest average home attendance")

    team_avg = (df.groupby("home_team")["attendance"]
                .agg(["mean","count","std"]).reset_index()
                .sort_values("mean", ascending=True))
    team_avg.columns = ["Team","Avg Attendance","Games","Std Dev"]

    fig_team = go.Figure()
    fig_team.add_trace(go.Bar(
        x=team_avg["Avg Attendance"], y=team_avg["Team"],
        orientation="h",
        marker=dict(
            color=team_avg["Avg Attendance"],
            colorscale=[[0,"#bfdbfe"],[0.5,"#3b82f6"],[1,"#1e3a8a"]],
            showscale=False,
        ),
        error_x=dict(type="data", array=team_avg["Std Dev"],
                     color="#9ca3af", thickness=1.5, width=4),
        text=team_avg["Avg Attendance"].apply(lambda x: f"{int(x):,}"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg Attendance: %{x:,.0f}<br>"
            "Games: %{customdata[0]}<br>"
            "Std Dev: ±%{customdata[1]:,.0f}<extra></extra>"
        ),
        customdata=team_avg[["Games","Std Dev"]].values,
    ))
    fig_team.update_layout(
        template=PLOT_THM, height=max(500, len(team_avg) * 24),
        xaxis=dict(tickformat=",", title="Average Attendance"),
        yaxis_title="",
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        margin=dict(l=10, r=80, t=10, b=10),
    )
    st.plotly_chart(fig_team, use_container_width=True)


# =============================================================================
# PAGE 3 — SCENARIO SIMULATOR
# =============================================================================
elif page == "🎛️ Scenario Simulator":

    st.title("🎛️ Scenario Simulator")
    st.caption(
        "Uses a Linear Regression model trained on your actual 2015–2024 NFL data "
        "to predict attendance based on the conditions you select."
    )

    # Fit model from real data
    with st.spinner("Loading training data from PostgreSQL..."):
        sim_model, avg_att, res_std = fit_simulator_model()

    if sim_model is None:
        st.error("Could not load training data. Check your database connection.")
        st.stop()

    st.markdown(
        f'<div class="info-box">🤖 This simulator uses a <b>Linear Regression</b> model '
        f'trained on <b>{int(avg_att):,}</b> avg attendance across 2015–2024 regular season games '
        f'(excluding 2020). Residual std: ±{int(res_std):,}. '
        f'Adjust the controls below to generate a live prediction.</div>',
        unsafe_allow_html=True)

    st.divider()

    # ── CONTROLS ─────────────────────────────────────────────────────────────
    section("⚙️ Set Game Conditions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🌦️ Weather**")
        temperature  = st.slider("Temperature (°F)",    0,   110,  65, 1)
        rain_prob    = st.slider("Rain Probability (%)", 0,   100,  10, 5)
        wind_speed_s = st.slider("Wind Speed (mph)",     0,   60,   8,  1)

        st.markdown("**🏆 Team Strength**")
        home_win_pct = st.slider("Home Team Win %",  0.0, 1.0, 0.50, 0.05, format="%.2f")
        away_win_pct = st.slider("Away Team Win %",  0.0, 1.0, 0.50, 0.05, format="%.2f")

        st.markdown("**📅 Scheduling**")
        week_num     = st.slider("Week of Season",   1,   18,   8,  1)
        home_rest    = st.slider("Home Rest Days",   4,   14,   7,  1)
        away_rest    = st.slider("Away Rest Days",   4,   14,   7,  1)

    with col2:
        st.markdown("**🎟️ Game Context**")
        promotion    = st.toggle("Promotional / Special Event Game",        value=False)
        primetime    = st.toggle("Primetime Game (SNF / MNF / TNF)",        value=False)
        divisional   = st.toggle("Divisional Rivalry Game",                 value=False)
        indoor_venue = st.toggle("Indoor / Dome Venue",                     value=False)
        weekend_game = st.toggle("Weekend Game (Sat / Sun)",                value=True)
        holiday_game = st.toggle("Holiday Game (Thanksgiving etc.)",        value=False)
        rivalry      = st.toggle("Historic Rivalry Matchup",                value=False)
        neutral_site = st.toggle("Neutral Site Game",                       value=False)
        intl_game    = st.toggle("International Game (London / Germany)",   value=False)

    st.divider()

    # ── PREDICT ──────────────────────────────────────────────────────────────
    # Build feature vector matching training columns exactly
    precip_inches = rain_prob / 100.0 * 0.5

    feature_vector = np.array([[
        float(temperature),
        float(precip_inches),
        float(wind_speed_s),
        float(home_win_pct),
        float(away_win_pct),
        1.0 if indoor_venue else 0.0,
        1.0 if weekend_game else 0.0,
        1.0 if primetime    else 0.0,
        1.0 if divisional   else 0.0,
        1.0 if rivalry      else 0.0,
        1.0 if holiday_game else 0.0,
        1.0 if neutral_site else 0.0,
        1.0 if intl_game    else 0.0,
        float(home_rest),
        float(away_rest),
        float(week_num),
    ]])

    prediction = float(sim_model.predict(feature_vector)[0])
    prediction = max(prediction, 0)
    lower      = max(prediction - 1.96 * res_std, 0)
    upper      = prediction + 1.96 * res_std

    severe = (temperature <= 20 or temperature >= 95 or
              precip_inches >= 0.5 or rain_prob >= 70 or wind_speed_s >= 35)

    # ── RESULTS ──────────────────────────────────────────────────────────────
    section("📊 Prediction Results")

    r1,r2,r3,r4,r5 = st.columns(5)
    delta = prediction - avg_att
    with r1: kpi_card("Predicted Attendance", f"{int(prediction):,}",   "Point estimate",    "blue")
    with r2: kpi_card("Lower Bound (95%)",    f"{int(lower):,}",        "Conservative",      "amber")
    with r3: kpi_card("Upper Bound (95%)",    f"{int(upper):,}",        "Optimistic",        "amber")
    with r4: kpi_card("vs League Average",    f"{int(delta):+,}",       f"Avg: {int(avg_att):,}",
                      "green" if delta >= 0 else "red")
    with r5: kpi_card("Weather Risk",
                      "⚠️ HIGH" if severe else "✅ LOW",
                      "Severe conditions flag",
                      "red" if severe else "green")

    st.divider()

    # ── GAUGE ────────────────────────────────────────────────────────────────
    capacity = 75000
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=int(prediction),
        delta={"reference": int(avg_att), "valueformat":",",
               "increasing":{"color":C_GREEN}, "decreasing":{"color":C_RED}},
        number={"valueformat":",", "font":{"size":36,"color":"#111827"}},
        gauge={
            "axis": {"range":[0,capacity], "tickformat":",",
                     "tickcolor":"#374151", "dtick":15000},
            "bar":  {"color": C_BLUE, "thickness": 0.25},
            "bgcolor": "#f9fafb",
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range":[0,             capacity*0.60], "color":"#fee2e2"},
                {"range":[capacity*0.60, capacity*0.85], "color":"#fef9c3"},
                {"range":[capacity*0.85, capacity],      "color":"#dcfce7"},
            ],
            "threshold": {"line":{"color":C_GREEN,"width":3},
                          "thickness":0.75,"value":int(avg_att)},
        },
        title={"text":"Predicted Attendance vs Typical NFL Stadium Capacity (75,000)",
               "font":{"size":13,"color":"#374151"}},
    ))
    fig_gauge.update_layout(
        template=PLOT_THM, height=320, paper_bgcolor=PLOT_BG,
        font={"color":"#374151"}, margin=dict(t=80,b=10,l=20,r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── COEFFICIENT IMPACT ───────────────────────────────────────────────────
    section("🔍 Model Feature Coefficients — What Matters Most")
    st.caption(
        "Each bar shows how much a 1-unit change in that feature "
        "shifts the predicted attendance, according to the fitted model."
    )

    feature_names = [
        "Temperature (°F)", "Precipitation (in)", "Wind Speed (mph)",
        "Home Win %", "Away Win %",
        "Indoor Venue", "Weekend Game", "Primetime", "Divisional Game",
        "Rivalry Game", "Holiday Game", "Neutral Site", "International",
        "Home Rest Days", "Away Rest Days", "Week of Season",
    ]
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": sim_model.coef_,
    }).sort_values("Coefficient")
    coef_df["Color"] = coef_df["Coefficient"].apply(lambda x: C_GREEN if x > 0 else C_RED)

    fig_coef = go.Figure(go.Bar(
        y=coef_df["Feature"], x=coef_df["Coefficient"],
        orientation="h", marker_color=coef_df["Color"],
        text=coef_df["Coefficient"].apply(lambda x: f"{x:+,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+,.1f}<extra></extra>",
    ))
    fig_coef.update_layout(
        template=PLOT_THM, height=500,
        xaxis=dict(tickformat=",", title="Attendance Impact per Unit",
                   zeroline=True, zerolinecolor="#9ca3af", zerolinewidth=2),
        yaxis_title="",
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        margin=dict(l=10, r=90, t=10, b=10),
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # ── SIMILAR HISTORICAL GAMES ─────────────────────────────────────────────
    section("🔎 Most Similar Historical Games — from PostgreSQL")
    st.caption(
        f"Games within ±8°F of {temperature}°F and closest home win % "
        f"to {home_win_pct:.0%} · excludes 2020"
    )

    sim_df = cast_numeric(safe_query("""
        SELECT
            mf.season                               AS "Season",
            mf.week                                 AS "Week",
            ht.team_name                            AS "Home Team",
            at_.team_name                           AS "Away Team",
            TO_CHAR(mf.attendance, 'FM999,999')     AS "Actual Attendance",
            ROUND(mf.temperature,1)                 AS "Temp °F",
            ROUND(mf.precipitation,3)               AS "Precip (in)",
            ROUND(mf.wind_speed,1)                  AS "Wind (mph)",
            ROUND(mf.home_team_win_pct,3)           AS "Home Win %",
            ROUND(mf.away_team_win_pct,3)           AS "Away Win %",
            CASE WHEN mf.indoor_flag       THEN '✅' ELSE '—' END AS "Indoor",
            CASE WHEN mf.primetime_flag    THEN '✅' ELSE '—' END AS "Primetime",
            CASE WHEN mf.severe_weather_flag THEN '⚠️' ELSE '✅' END AS "Wx Risk"
        FROM ml_features mf
        JOIN fact_game fg  ON fg.game_id  = mf.game_id
        JOIN dim_team  ht  ON ht.team_id  = fg.home_team_id
        JOIN dim_team  at_ ON at_.team_id = fg.away_team_id
        WHERE mf.temperature BETWEEN :tlo AND :thi
          AND mf.attendance IS NOT NULL
          AND mf.season NOT IN (2020, 2025)
        ORDER BY ABS(COALESCE(mf.home_team_win_pct,0.5) - :wp)
        LIMIT 12
    """, {"tlo": temperature-8, "thi": temperature+8, "wp": home_win_pct}))

    if not sim_df.empty:
        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        # Parse attendance back to numeric for avg
        avg_sim = (sim_df["Actual Attendance"]
                   .str.replace(",","").astype(float).mean())
        diff = prediction - avg_sim
        color_diff = "green" if diff >= 0 else "red"
        st.markdown(
            f'<div class="info-box">'
            f'📊 Average actual attendance in similar conditions: '
            f'<b>{int(avg_sim):,}</b> &nbsp;|&nbsp; '
            f'Your estimate: <b>{int(prediction):,}</b> &nbsp;|&nbsp; '
            f'Difference: <b style="color:{"#16a34a" if diff>=0 else "#dc2626"}">'
            f'{int(diff):+,}</b>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.info("No similar historical games found in this temperature range.")
