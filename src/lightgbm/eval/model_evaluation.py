"""Model evaluation utilities for LightGBM."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb
from src.lightgbm.eval.shap_analysis import compute_shap_values
from src.path import LIGHTGBM_SHAP_PLOTS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with evaluation metrics (MAE, MSE, RMSE, R²).
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return metrics


def _log_model_metrics(model_name: str, metrics: dict[str, float]) -> None:
    """Log model evaluation metrics.

    Args:
        model_name: Name of the model.
        metrics: Dictionary of evaluation metrics.
    """
    logger.info(f"{model_name} - Test MAE: {metrics['mae']:.6f}")
    logger.info(f"{model_name} - Test RMSE: {metrics['rmse']:.6f}")
    logger.info(f"{model_name} - Test R²: {metrics['r2']:.6f}")


def _get_sorted_feature_importances(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor], feature_names: pd.Index
) -> dict[str, float]:
    """Get feature importances sorted by importance.

    Args:
        model: Trained regression model (LightGBM or RandomForest).
        feature_names: Feature names.

    Returns:
        Dictionary mapping feature names to importances, sorted by importance.
    """
    feature_importances = dict(zip(feature_names, model.feature_importances_, strict=False))
    return dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))


def evaluate_model(
    model: Union[lgb.LGBMRegressor, RandomForestRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    *,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Evaluate model on test set with metrics and SHAP analysis.

    Args:
        model: Trained regression model (LightGBM or RandomForest).
        X_test: Test features.
        y_test: Test target.
        model_name: Name for the model.
        compute_shap: Whether to compute SHAP values for this evaluation.

    Returns:
        Dictionary with evaluation results (metrics, SHAP info).
    """
    logger.info(f"Evaluating model: {model_name}")

    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).flatten()
    metrics = compute_metrics(y_test, y_pred)
    _log_model_metrics(model_name, metrics)

    shap_plot_path: Path | None = None
    if compute_shap:
        _, shap_plot_path = compute_shap_values(
            model,
            X_test,
            model_name,
            output_dir=LIGHTGBM_SHAP_PLOTS_DIR,
        )

    feature_importances = _get_sorted_feature_importances(model, X_test.columns)

    results = {
        "model_name": model_name,
        "test_metrics": metrics,
        "test_size": int(len(X_test)),
        "n_features": int(X_test.shape[1]),
        "feature_importances": {k: float(v) for k, v in feature_importances.items()},
        "shap_plot_path": str(shap_plot_path) if shap_plot_path else None,
    }

    return results
