"""
File: train_linear_regression.py

Purpose:
Train a linear regression model for NFL attendance prediction using
game-level features from ml_features_attendance.csv with time-based splits.

Inputs:
- data/processed/modeling/ml_features_attendance.csv
- data/processed/features/ml_features_attendance.csv
- data/processed/ml_features_attendance.csv
- data/modeling/ml_features_attendance.csv

Outputs:
- data/modeling/linear_regression_metrics.csv
- data/modeling/linear_regression_predictions_train.csv
- data/modeling/linear_regression_predictions_validation.csv
- data/modeling/linear_regression_predictions_test.csv
- data/modeling/linear_regression_model.joblib
"""

from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_NAME = "linear_regression"
TARGET_COLUMN = "attendance"
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

INPUT_CANDIDATES = [
    Path("data/processed/modeling/ml_features_attendance.csv"),
    Path("data/processed/features/ml_features_attendance.csv"),
    Path("data/processed/ml_features_attendance.csv"),
    Path("data/modeling/ml_features_attendance.csv"),
]

OUTPUT_DIR = Path("data/modeling")

FEATURE_COLUMNS = [
    "temperature",
    "precipitation",
    "wind_speed",
    "home_team_win_pct",
    "away_team_win_pct",
    "weekend_flag",
    "holiday_flag",
    "holiday_before_flag",
    "holiday_after_flag",
    "holiday_adjacent_flag",
    "indoor_flag",
    "divisional_game_flag",
    "rivalry_flag",
    "primetime_flag",
    "week_of_season",
    "month",
    "home_rest_days",
    "away_rest_days",
    "weather_condition",
    "severe_weather_flag",
    "home_prior_season_win_pct",
    "away_prior_season_win_pct",
    "neutral_site_flag",
    "international_flag",
]

IDENTIFIER_COLUMNS = [
    "game_id",
    "season",
]

ALLOWED_GAME_TYPES = {"REG"}


def find_input_file() -> Path:
    for file_path in INPUT_CANDIDATES:
        if file_path.exists():
            return file_path
    raise FileNotFoundError(
        "Could not find ml_features_attendance.csv in the expected locations."
    )


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_columns = set(IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN, "game_type"])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")

    return df


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    working_df["season"] = pd.to_numeric(working_df["season"], errors="coerce")
    working_df[TARGET_COLUMN] = pd.to_numeric(working_df[TARGET_COLUMN], errors="coerce")

    if "game_type" in working_df.columns:
        working_df["game_type"] = (
            working_df["game_type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
        working_df = working_df[working_df["game_type"].isin(ALLOWED_GAME_TYPES)].copy()

    working_df = working_df[working_df["season"] != 2020].copy()
    working_df = working_df[working_df[TARGET_COLUMN].notna()].copy()

    numeric_like_columns = [
        "temperature",
        "precipitation",
        "wind_speed",
        "home_team_win_pct",
        "away_team_win_pct",
        "week_of_season",
        "month",
        "home_rest_days",
        "away_rest_days",
        "home_prior_season_win_pct",
        "away_prior_season_win_pct",
    ]

    categorical_like_columns = [
        "weekend_flag",
        "holiday_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
        "indoor_flag",
        "divisional_game_flag",
        "rivalry_flag",
        "primetime_flag",
        "severe_weather_flag",
        "neutral_site_flag",
        "international_flag",
        "weather_condition",
    ]

    for column in numeric_like_columns:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    for column in categorical_like_columns:
        if column in working_df.columns:
            working_df[column] = working_df[column].astype(object)

    if "weather_condition" in working_df.columns:
        working_df["weather_condition"] = working_df["weather_condition"].fillna("MISSING")

    return working_df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["season"].isin(TRAIN_YEARS)].copy()
    validation_df = df[df["season"] == VALIDATION_YEAR].copy()
    test_df = df[df["season"] == TEST_YEAR].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty.")
    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")
    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    return train_df, validation_df, test_df


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    available_features = [column for column in FEATURE_COLUMNS if column in df.columns]

    categorical_features = [
        "weekend_flag",
        "holiday_flag",
        "holiday_before_flag",
        "holiday_after_flag",
        "holiday_adjacent_flag",
        "indoor_flag",
        "divisional_game_flag",
        "rivalry_flag",
        "primetime_flag",
        "severe_weather_flag",
        "neutral_site_flag",
        "international_flag",
        "weather_condition",
    ]

    categorical_features = [
        column for column in categorical_features if column in available_features
    ]

    numeric_features = [
        column for column in available_features if column not in categorical_features
    ]

    return numeric_features, categorical_features


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    return model_pipeline


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def get_validation_residual_std(actual: pd.Series, predicted: np.ndarray) -> float:
    residuals = actual - predicted
    residual_std = float(np.std(residuals, ddof=1))

    if np.isnan(residual_std) or residual_std <= 0:
        residual_std = 0.0

    return residual_std


def build_prediction_output(
    df: pd.DataFrame,
    predictions: np.ndarray,
    residual_std: float,
    dataset_split: str,
) -> pd.DataFrame:
    interval_margin = 1.96 * residual_std

    output_df = pd.DataFrame(
        {
            "game_id": df["game_id"].values,
            "season": df["season"].values,
            "actual_attendance": df[TARGET_COLUMN].values,
            "predicted_attendance": predictions,
            "prediction_lower": predictions - interval_margin,
            "prediction_upper": predictions + interval_margin,
            "model_name": MODEL_NAME,
            "dataset_split": dataset_split,
        }
    )

    return output_df


def save_metrics(
    train_metrics: dict,
    validation_metrics: dict,
    test_metrics: dict,
    residual_std: float,
) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "dataset_split": "train",
                "mae": train_metrics["mae"],
                "rmse": train_metrics["rmse"],
                "r2": train_metrics["r2"],
                "validation_residual_std": residual_std,
            },
            {
                "model_name": MODEL_NAME,
                "dataset_split": "validation",
                "mae": validation_metrics["mae"],
                "rmse": validation_metrics["rmse"],
                "r2": validation_metrics["r2"],
                "validation_residual_std": residual_std,
            },
            {
                "model_name": MODEL_NAME,
                "dataset_split": "test",
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "validation_residual_std": residual_std,
            },
        ]
    )

    metrics_df.to_csv(OUTPUT_DIR / "linear_regression_metrics.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = find_input_file()
    raw_df = load_data(input_file)
    modeling_df = prepare_modeling_data(raw_df)

    train_df, validation_df, test_df = split_data(modeling_df)
    numeric_features, categorical_features = get_feature_types(modeling_df)

    feature_columns = numeric_features + categorical_features

    x_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    x_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()

    x_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    model_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    model_pipeline.fit(x_train, y_train)

    train_predictions = model_pipeline.predict(x_train)
    validation_predictions = model_pipeline.predict(x_validation)
    test_predictions = model_pipeline.predict(x_test)

    validation_residual_std = get_validation_residual_std(
        actual=y_validation,
        predicted=validation_predictions,
    )

    train_metrics = calculate_metrics(y_train, train_predictions)
    validation_metrics = calculate_metrics(y_validation, validation_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    train_output_df = build_prediction_output(
        df=train_df,
        predictions=train_predictions,
        residual_std=validation_residual_std,
        dataset_split="train",
    )
    validation_output_df = build_prediction_output(
        df=validation_df,
        predictions=validation_predictions,
        residual_std=validation_residual_std,
        dataset_split="validation",
    )
    test_output_df = build_prediction_output(
        df=test_df,
        predictions=test_predictions,
        residual_std=validation_residual_std,
        dataset_split="test",
    )

    train_output_df.to_csv(
        OUTPUT_DIR / "linear_regression_predictions_train.csv",
        index=False,
    )
    validation_output_df.to_csv(
        OUTPUT_DIR / "linear_regression_predictions_validation.csv",
        index=False,
    )
    test_output_df.to_csv(
        OUTPUT_DIR / "linear_regression_predictions_test.csv",
        index=False,
    )

    save_metrics(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        residual_std=validation_residual_std,
    )

    joblib.dump(
        model_pipeline,
        OUTPUT_DIR / "linear_regression_model.joblib",
    )


if __name__ == "__main__":
    main()