"""
File: train_neural_network.py

Purpose:
Train a neural network regressor for NFL home game attendance prediction using
time-based data splits and save model metrics, predictions, and an optional
model artifact to disk.

Inputs:
- data/modeling/ml_features_attendance.csv
- data/processed/ml_features_attendance.csv
- ml_features_attendance.csv

Outputs:
- data/modeling/neural_network_metrics.csv
- data/modeling/neural_network_predictions_train.csv
- data/modeling/neural_network_predictions_validation.csv
- data/modeling/neural_network_predictions_test.csv
- data/modeling/neural_network_model.joblib
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
TRAIN_YEARS = list(range(2015, 2024))
VALIDATION_YEARS = [2024]
TEST_YEARS = [2025]
EXCLUDED_YEARS = [2020]
TARGET_COLUMN = "attendance"
ID_COLUMN = "game_id"
MODEL_NAME = "neural_network"

OUTPUT_DIR = Path("data/modeling")
METRICS_OUTPUT_PATH = OUTPUT_DIR / "neural_network_metrics.csv"
TRAIN_PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "neural_network_predictions_train.csv"
VALIDATION_PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "neural_network_predictions_validation.csv"
TEST_PREDICTIONS_OUTPUT_PATH = OUTPUT_DIR / "neural_network_predictions_test.csv"
MODEL_ARTIFACT_PATH = OUTPUT_DIR / "neural_network_model.joblib"

INPUT_CANDIDATE_PATHS = [
    Path("data/processed/features/ml_features_attendance.csv"),
    Path("data/modeling/ml_features_attendance.csv"),
    Path("data/processed/ml_features_attendance.csv"),
    Path("ml_features_attendance.csv"),
]

def ensure_output_directory(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)


def find_input_file() -> Path:
    for path in INPUT_CANDIDATE_PATHS:
        if path.exists():
            return path
    candidate_text = "\n".join(str(path) for path in INPUT_CANDIDATE_PATHS)
    raise FileNotFoundError(
        f"Could not find ml_features_attendance.csv in any expected location:\n{candidate_text}"
    )


def load_modeling_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required_columns = {ID_COLUMN, "season", "game_type", TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_text}")

    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df[df["season"].between(2015, 2025, inclusive="both")]
    df = df[~df["season"].isin(EXCLUDED_YEARS)]
    df = df[df["game_type"].astype(str).str.upper() == "REG"]
    df = df[df[TARGET_COLUMN].notna()]

    if df.empty:
        raise ValueError("No modeling rows remain after filtering to regular-season rows with attendance.")

    return df.reset_index(drop=True)


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bool_columns = df.select_dtypes(include=["bool"]).columns.tolist()

    for column in bool_columns:
        df[column] = df[column].astype(int)

    return df


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded_columns = {ID_COLUMN, TARGET_COLUMN, "season"}
    feature_columns = [column for column in df.columns if column not in excluded_columns]

    feature_df = df[feature_columns].copy()

    categorical_columns = feature_df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    return numeric_columns, categorical_columns


def build_model_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.0005,
        max_iter=3000,
        shuffle=False,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        tol=0.0001,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def split_data_by_year(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["season"].isin(TRAIN_YEARS)].copy()
    validation_df = df[df["season"].isin(VALIDATION_YEARS)].copy()
    test_df = df[df["season"].isin(TEST_YEARS)].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if validation_df.empty:
        raise ValueError("Validation split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    return train_df, validation_df, test_df


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def build_predictions_output(
    df: pd.DataFrame,
    predictions: np.ndarray,
    residual_std: float,
    split_name: str,
) -> pd.DataFrame:
    z_value = 1.96

    output_df = df[[ID_COLUMN, "season", "week", "game_type", TARGET_COLUMN]].copy()
    output_df["prediction"] = predictions
    output_df["prediction_interval_lower"] = predictions - (z_value * residual_std)
    output_df["prediction_interval_upper"] = predictions + (z_value * residual_std)
    output_df["residual"] = output_df[TARGET_COLUMN] - output_df["prediction"]
    output_df["data_split"] = split_name

    return output_df


def save_metrics(
    best_params: dict[str, object],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    residual_std: float,
) -> None:
    metrics_rows = [
        {
            "model_name": MODEL_NAME,
            "data_split": "train",
            "hidden_layer_sizes": str(best_params["hidden_layer_sizes"]),
            "alpha": best_params["alpha"],
            "validation_residual_std": residual_std,
            **train_metrics,
        },
        {
            "model_name": MODEL_NAME,
            "data_split": "validation",
            "hidden_layer_sizes": str(best_params["hidden_layer_sizes"]),
            "alpha": best_params["alpha"],
            "validation_residual_std": residual_std,
            **validation_metrics,
        },
        {
            "model_name": MODEL_NAME,
            "data_split": "test",
            "hidden_layer_sizes": str(best_params["hidden_layer_sizes"]),
            "alpha": best_params["alpha"],
            "validation_residual_std": residual_std,
            **test_metrics,
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)


def main() -> None:
    ensure_output_directory(OUTPUT_DIR)

    input_path = find_input_file()
    modeling_df = load_modeling_data(input_path)
    modeling_df = convert_boolean_columns(modeling_df)

    train_df, validation_df, test_df = split_data_by_year(modeling_df)

    numeric_columns, categorical_columns = build_feature_lists(modeling_df)

    candidate_settings = [
        {"hidden_layer_sizes": (16,), "alpha": 0.0001},
        {"hidden_layer_sizes": (32,), "alpha": 0.0001},
        {"hidden_layer_sizes": (16, 8), "alpha": 0.0001},
        {"hidden_layer_sizes": (32,), "alpha": 0.001},
    ]

    train_feature_df = train_df.drop(columns=[TARGET_COLUMN])
    validation_feature_df = validation_df.drop(columns=[TARGET_COLUMN])
    test_feature_df = test_df.drop(columns=[TARGET_COLUMN])

    train_target = train_df[TARGET_COLUMN]
    validation_target = validation_df[TARGET_COLUMN]
    test_target = test_df[TARGET_COLUMN]

    best_pipeline = None
    best_params = None
    best_validation_mae = float("inf")

    for params in candidate_settings:
        pipeline = build_model_pipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
        )

        pipeline.fit(train_feature_df, train_target)
        validation_predictions = pipeline.predict(validation_feature_df)
        validation_mae = mean_absolute_error(validation_target, validation_predictions)

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_pipeline = pipeline
            best_params = params

    if best_pipeline is None or best_params is None:
        raise RuntimeError("Model training failed. No valid neural network model was selected.")

    train_predictions = best_pipeline.predict(train_feature_df)
    validation_predictions = best_pipeline.predict(validation_feature_df)
    test_predictions = best_pipeline.predict(test_feature_df)

    validation_residuals = validation_target.to_numpy() - validation_predictions
    residual_std = float(np.std(validation_residuals, ddof=1))

    train_metrics = calculate_metrics(train_target, train_predictions)
    validation_metrics = calculate_metrics(validation_target, validation_predictions)
    test_metrics = calculate_metrics(test_target, test_predictions)

    train_predictions_df = build_predictions_output(
        df=train_df,
        predictions=train_predictions,
        residual_std=residual_std,
        split_name="train",
    )
    validation_predictions_df = build_predictions_output(
        df=validation_df,
        predictions=validation_predictions,
        residual_std=residual_std,
        split_name="validation",
    )
    test_predictions_df = build_predictions_output(
        df=test_df,
        predictions=test_predictions,
        residual_std=residual_std,
        split_name="test",
    )

    save_metrics(
        best_params=best_params,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        residual_std=residual_std,
    )

    train_predictions_df.to_csv(TRAIN_PREDICTIONS_OUTPUT_PATH, index=False)
    validation_predictions_df.to_csv(VALIDATION_PREDICTIONS_OUTPUT_PATH, index=False)
    test_predictions_df.to_csv(TEST_PREDICTIONS_OUTPUT_PATH, index=False)

    joblib.dump(best_pipeline, MODEL_ARTIFACT_PATH)


if __name__ == "__main__":
    main()