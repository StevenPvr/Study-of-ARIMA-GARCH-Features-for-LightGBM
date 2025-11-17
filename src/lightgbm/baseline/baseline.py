"""Random baseline model for benchmarking LightGBM models.

Generates random predictions from the same distribution as the test target
to establish a baseline performance level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants import DEFAULT_RANDOM_STATE
from src.lightgbm.shared_utils import load_test_data
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_random_predictions(
    y_test: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Generate random predictions from the same distribution as y_test.

    Args:
        y_test: True target values.
        random_state: Random seed for reproducibility.

    Returns:
        Array of random predictions.
    """
    rng = np.random.RandomState(random_state)
    mean = y_test.mean()
    std = y_test.std()
    n_samples = len(y_test)

    # Generate predictions from normal distribution with same mean/std as y_test
    y_pred_random = rng.normal(loc=mean, scale=std, size=n_samples)

    logger.info(f"Generated random predictions: mean={mean:.6f}, std={std:.6f}, n={n_samples}")
    return y_pred_random


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


def compute_random_baseline_metrics(
    dataset_path: Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, str | float | int | dict[str, float]]:
    """Compute metrics for random baseline model.

    Args:
        dataset_path: Path to dataset file (.parquet or .csv).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with model_name, test_metrics, test_size, n_features.
    """
    logger.info("=" * 70)
    logger.info("RANDOM BASELINE MODEL")
    logger.info("=" * 70)

    # Load test data
    X_test, y_test = load_test_data(dataset_path)

    # Generate random predictions
    y_pred = _generate_random_predictions(y_test, random_state=random_state)

    # Compute metrics
    metrics = _compute_metrics(y_test, y_pred)

    logger.info("Random Baseline Metrics:")
    logger.info(f"  MAE:  {metrics['mae']:.6f}")
    logger.info(f"  MSE:  {metrics['mse']:.6f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  R²:   {metrics['r2']:.6f}")

    return {
        "model_name": "random_baseline",
        "test_metrics": metrics,
        "test_size": len(X_test),
        "n_features": 0,  # Random baseline doesn't use features
        "feature_importances": {},  # No feature importances for random model
    }
