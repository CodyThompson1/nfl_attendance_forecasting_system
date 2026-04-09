"""
File: train_knn_regressor.py

Purpose:
Train a K-Nearest Neighbors regressor to predict NFL home game attendance
from the model-ready feature file using time-based train, validation, and
test splits. The script selects the best k value using validation performance,
evaluates the final model, creates simple 95 percent prediction intervals,
and saves metrics, predictions, and a model artifact.

Inputs:
- data/modeling/processed/features/ml_features_attendance.csv
- /mnt/data/processed/features/ml_features_attendance.csv

Outputs:
- data/modeling/knn_metrics.csv
- data/modeling/knn_predictions_train.csv
- data/modeling/knn_predictions_validation.csv
- data/modeling/knn_predictions_test.csv
- data/modeling/knn_tuning_results.csv
- data/modeling/knn_model.joblib
"""

from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "attendance"
MODEL_NAME = "knn_regressor"
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
VALIDATION_YEAR = 2024
TEST_YEAR = 2025
EXCLUDED_YEAR = 2020
K_VALUES = [3, 5, 7, 9, 11]

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

ID_COLUMNS = ["game_id", "season"]
OPTIONAL_FILTER_COLUMNS = ["game_type"]


def find_input_file() -> Path:
    candidate_paths = [
        Path("data/processed/features/ml_features_attendance.csv"),
        Path("/mnt/data/prcessed/features/ml_features_attendance.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find ml_features_attendance.csv in expected locations."
    )


def ensure_output_directory() -> Path:
    output_dir = Path("data/modeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = ID_COLUMNS + [TARGET_COLUMN] + FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in input file: {missing_text}")


def filter_modeling_rows(df: pd.DataFrame) -> pd.DataFrame:
    modeling_df = df.copy()

    if "season" not in modeling_df.columns:
        raise ValueError("Input file must contain a season column.")

    modeling_df["season"] = pd.to_numeric(modeling_df["season"], errors="coerce")
    modeling_df = modeling_df[modeling_df["season"].notna()].copy()
    modeling_df["season"] = modeling_df["season"].astype(int)

    modeling_df = modeling_df[modeling_df["season"] != EXCLUDED_YEAR].copy()

    if "game_type" in modeling_df.columns:
        modeling_df["game_type"] = modeling_df["game_type"].astype(str).str.upper()
        modeling_df = modeling_df[modeling_df["game_type"] == "REG"].copy()

    modeling_df[TARGET_COLUMN] = pd.to_numeric(
        modeling_df[TARGET_COLUMN], errors="coerce"
    )
    modeling_df = modeling_df[modeling_df[TARGET_COLUMN].notna()].copy()
    modeling_df = modeling_df[modeling_df[TARGET_COLUMN] > 0].copy()

    return modeling_df


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted_df = df.copy()

    for column in FEATURE_COLUMNS:
        if column not in converted_df.columns:
            continue

        if pd.api.types.is_bool_dtype(converted_df[column]):
            converted_df[column] = converted_df[column].astype(int)

    return converted_df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["season"].isin(TRAIN_YEARS)].copy()
    validation_df = df[df["season"] == VALIDATION_YEAR].copy()
    test_df = df[df["season"] == TEST_YEAR].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty after filtering.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty after filtering.")

    if test_df.empty:
        raise ValueError("Test dataset is empty after filtering.")

    return train_df, validation_df, test_df


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = []
    categorical_features = []

    for column in FEATURE_COLUMNS:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    n_neighbors: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = KNeighborsRegressor(n_neighbors=n_neighbors)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def fit_and_score_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    n_neighbors: int,
) -> dict:
    feature_columns = numeric_features + categorical_features

    model_pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_neighbors=n_neighbors,
    )

    x_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    x_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()

    model_pipeline.fit(x_train, y_train)

    train_predictions = model_pipeline.predict(x_train)
    validation_predictions = model_pipeline.predict(x_validation)

    train_metrics = calculate_metrics(y_train, train_predictions)
    validation_metrics = calculate_metrics(y_validation, validation_predictions)

    return {
        "k": n_neighbors,
        "pipeline": model_pipeline,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }


def select_best_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, int, pd.DataFrame]:
    tuning_results = []

    best_pipeline = None
    best_k = None
    best_validation_rmse = None

    for k_value in K_VALUES:
        result = fit_and_score_model(
            train_df=train_df,
            validation_df=validation_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_neighbors=k_value,
        )

        tuning_results.append(
            {
                "model_name": MODEL_NAME,
                "k": k_value,
                "train_mae": result["train_metrics"]["mae"],
                "train_rmse": result["train_metrics"]["rmse"],
                "train_r2": result["train_metrics"]["r2"],
                "validation_mae": result["validation_metrics"]["mae"],
                "validation_rmse": result["validation_metrics"]["rmse"],
                "validation_r2": result["validation_metrics"]["r2"],
            }
        )

        current_validation_rmse = result["validation_metrics"]["rmse"]

        if best_validation_rmse is None or current_validation_rmse < best_validation_rmse:
            best_validation_rmse = current_validation_rmse
            best_pipeline = result["pipeline"]
            best_k = k_value

    tuning_results_df = pd.DataFrame(tuning_results).sort_values(
        by=["validation_rmse", "validation_mae", "k"]
    )

    return best_pipeline, best_k, tuning_results_df


def build_prediction_output(
    df: pd.DataFrame,
    predictions: np.ndarray,
    interval_half_width: float,
    dataset_split: str,
) -> pd.DataFrame:
    output_df = df[ID_COLUMNS].copy()
    output_df["actual_attendance"] = df[TARGET_COLUMN].values
    output_df["predicted_attendance"] = predictions
    output_df["prediction_lower"] = predictions - interval_half_width
    output_df["prediction_upper"] = predictions + interval_half_width
    output_df["prediction_lower"] = output_df["prediction_lower"].clip(lower=0)
    output_df["model_name"] = MODEL_NAME
    output_df["dataset_split"] = dataset_split

    return output_df


def evaluate_final_model(
    model_pipeline: Pipeline,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    feature_columns = numeric_features + categorical_features

    x_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    x_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()

    x_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    train_predictions = model_pipeline.predict(x_train)
    validation_predictions = model_pipeline.predict(x_validation)
    test_predictions = model_pipeline.predict(x_test)

    validation_residuals = y_validation.values - validation_predictions
    validation_residual_std = float(np.std(validation_residuals, ddof=1))
    interval_half_width = 1.96 * validation_residual_std

    train_output_df = build_prediction_output(
        df=train_df,
        predictions=train_predictions,
        interval_half_width=interval_half_width,
        dataset_split="train",
    )

    validation_output_df = build_prediction_output(
        df=validation_df,
        predictions=validation_predictions,
        interval_half_width=interval_half_width,
        dataset_split="validation",
    )

    test_output_df = build_prediction_output(
        df=test_df,
        predictions=test_predictions,
        interval_half_width=interval_half_width,
        dataset_split="test",
    )

    metrics_rows = []

    train_metrics = calculate_metrics(y_train, train_predictions)
    metrics_rows.append(
        {
            "model_name": MODEL_NAME,
            "dataset_split": "train",
            "mae": train_metrics["mae"],
            "rmse": train_metrics["rmse"],
            "r2": train_metrics["r2"],
            "prediction_interval_method": "validation_residual_std",
            "prediction_interval_z": 1.96,
            "prediction_interval_half_width": interval_half_width,
            "row_count": len(train_df),
        }
    )

    validation_metrics = calculate_metrics(y_validation, validation_predictions)
    metrics_rows.append(
        {
            "model_name": MODEL_NAME,
            "dataset_split": "validation",
            "mae": validation_metrics["mae"],
            "rmse": validation_metrics["rmse"],
            "r2": validation_metrics["r2"],
            "prediction_interval_method": "validation_residual_std",
            "prediction_interval_z": 1.96,
            "prediction_interval_half_width": interval_half_width,
            "row_count": len(validation_df),
        }
    )

    test_metrics = calculate_metrics(y_test, test_predictions)
    metrics_rows.append(
        {
            "model_name": MODEL_NAME,
            "dataset_split": "test",
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "r2": test_metrics["r2"],
            "prediction_interval_method": "validation_residual_std",
            "prediction_interval_z": 1.96,
            "prediction_interval_half_width": interval_half_width,
            "row_count": len(test_df),
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)

    return (
        metrics_df,
        train_output_df,
        validation_output_df,
        test_output_df,
        validation_residual_std,
    )


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    train_predictions_df: pd.DataFrame,
    validation_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    tuning_results_df: pd.DataFrame,
    model_pipeline: Pipeline,
    best_k: int,
    validation_residual_std: float,
) -> None:
    metrics_df.to_csv(output_dir / "knn_metrics.csv", index=False)
    train_predictions_df.to_csv(
        output_dir / "knn_predictions_train.csv", index=False
    )
    validation_predictions_df.to_csv(
        output_dir / "knn_predictions_validation.csv", index=False
    )
    test_predictions_df.to_csv(output_dir / "knn_predictions_test.csv", index=False)
    tuning_results_df.to_csv(output_dir / "knn_tuning_results.csv", index=False)

    model_artifact = {
        "model_name": MODEL_NAME,
        "best_k": best_k,
        "validation_residual_std": validation_residual_std,
        "pipeline": model_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_years": TRAIN_YEARS,
        "validation_year": VALIDATION_YEAR,
        "test_year": TEST_YEAR,
        "excluded_year": EXCLUDED_YEAR,
    }

    joblib.dump(model_artifact, output_dir / "knn_model.joblib")


def main() -> None:
    input_file = find_input_file()
    output_dir = ensure_output_directory()

    raw_df = load_data(input_file)
    validate_columns(raw_df)

    modeling_df = filter_modeling_rows(raw_df)
    modeling_df = convert_boolean_columns(modeling_df)

    train_df, validation_df, test_df = split_data(modeling_df)
    numeric_features, categorical_features = get_feature_types(modeling_df)

    best_pipeline, best_k, tuning_results_df = select_best_model(
        train_df=train_df,
        validation_df=validation_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    (
        metrics_df,
        train_predictions_df,
        validation_predictions_df,
        test_predictions_df,
        validation_residual_std,
    ) = evaluate_final_model(
        model_pipeline=best_pipeline,
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    metrics_df["best_k"] = best_k

    save_outputs(
        output_dir=output_dir,
        metrics_df=metrics_df,
        train_predictions_df=train_predictions_df,
        validation_predictions_df=validation_predictions_df,
        test_predictions_df=test_predictions_df,
        tuning_results_df=tuning_results_df,
        model_pipeline=best_pipeline,
        best_k=best_k,
        validation_residual_std=validation_residual_std,
    )


if __name__ == "__main__":
    main()