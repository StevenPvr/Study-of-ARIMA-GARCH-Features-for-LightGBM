"""Naive baseline model for benchmarking LightGBM models.

This model predicts log_volatility at J+1 using only the log_volatility at J,
providing a simple persistence baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.lightgbm.shared_utils import load_test_data
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_naive_predictions(X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """Generate naive predictions using log_volatility lag 1.

    The naive baseline predicts log_volatility_{t+1} = log_volatility_{t}.

    Args:
        X_test: Test features (must contain 'log_volatility_lag_1').
        y_test: True target values.

    Returns:
        Array of naive predictions.

    Raises:
        ValueError: If log_volatility_lag_1 feature is not found.
    """
    # Check if log_volatility_lag_1 exists in features
    if "log_volatility_lag_1" not in X_test.columns:
        raise ValueError(
            "Feature 'log_volatility_lag_1' not found in dataset. "
            "This baseline requires lag features to be present."
        )

    # Use lag1 as prediction
    y_pred_naive = cast(np.ndarray, X_test["log_volatility_lag_1"].to_numpy())

    logger.info(f"Generated naive predictions using log_volatility_lag_1: n={len(y_pred_naive)}")
    logger.info(f"  Mean prediction: {y_pred_naive.mean():.6f}")
    logger.info(f"  Std prediction:  {y_pred_naive.std():.6f}")
    logger.info(f"  Mean target:     {y_test.mean():.6f}")
    logger.info(f"  Std target:      {y_test.std():.6f}")

    return y_pred_naive


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, MSE, RMSE, R² metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def compute_naive_baseline_metrics(
    dataset_path: Path,
) -> dict[str, str | float | int | dict[str, float]]:
    """Compute metrics for naive baseline model.

    Args:
        dataset_path: Path to dataset file (.parquet or .csv).

    Returns:
        Dictionary with model_name, test_metrics, test_size, n_features.
    """
    logger.info("=" * 70)
    logger.info("NAIVE BASELINE MODEL (Persistence: J+1 = J)")
    logger.info("=" * 70)

    # Load test data
    X_test, y_test = load_test_data(dataset_path)

    # Generate naive predictions
    y_pred = _generate_naive_predictions(X_test, y_test)

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred)

    logger.info("Naive Baseline Metrics (Persistence):")
    logger.info(f"  MAE:  {metrics['mae']:.6f}")
    logger.info(f"  MSE:  {metrics['mse']:.6f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")

    return {
        "model_name": "naive_persistence_baseline",
        "test_metrics": metrics,
        "test_size": len(X_test),
        "n_features": 1,  # Only uses log_volatility_lag_1
        "feature_importances": {"log_volatility_lag_1": 1.0},
    }
