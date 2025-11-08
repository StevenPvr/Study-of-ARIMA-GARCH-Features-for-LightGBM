"""Module for saving GARCH dataset from SARIMA evaluation results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import GARCH_DATASET_FILE, WEIGHTED_LOG_RETURNS_SPLIT_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def _add_test_sarima_columns(results: dict[str, Any], split_df: pd.DataFrame) -> pd.DataFrame:
    """Add SARIMA test predictions and residuals (log-returns) to split dataframe.

    Args:
        results: Evaluation results with 'predictions' (log), 'actuals' (log), 'dates'
        split_df: Split dataframe with date column

    Returns:
        DataFrame with merged SARIMA test columns (log domain)
    """
    pred_log = np.asarray(results["predictions"], dtype=float)
    act_log = np.asarray(results["actuals"], dtype=float)
    dates = pd.to_datetime(results["dates"])  # type: ignore[arg-type]

    resid_log = act_log - pred_log

    sarima_df = pd.DataFrame(
        {
            "date": dates,
            # Keep column name stable for downstream code; value is in log domain
            "arima_pred_return": pred_log,
            "arima_residual_return": resid_log,
        }
    )

    return split_df.merge(sarima_df, on="date", how="left")


def _extract_train_series(merged: pd.DataFrame) -> pd.Series:
    """Extract train series from merged dataframe.

    Args:
        merged: Merged dataframe with train/test split

    Returns:
        Train series with date index
    """
    train_mask = merged["split"] == "train"
    train_df = merged.loc[train_mask, ["date", "weighted_log_return"]].copy()
    train_df = train_df.set_index("date")
    return train_df["weighted_log_return"].astype(float)


def _align_fitted_values_to_train(fitted_vals: Any, train_series: pd.Series) -> pd.Series:
    """Align fitted values to train series index.

    Args:
        fitted_vals: Fitted values from model (Series or array)
        train_series: Train series with date index

    Returns:
        Fitted series aligned to train series index
    """
    if isinstance(fitted_vals, pd.Series):
        fitted_series = fitted_vals
        if not isinstance(fitted_series.index, pd.DatetimeIndex):
            fitted_series.index = train_series.index
        return fitted_series

    return pd.Series(fitted_vals, index=train_series.index)


def _compute_train_log_returns(
    aligned: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Compute log-domain predictions and residuals for train.

    Args:
        aligned: DataFrame with 'actual_log' and 'pred_log' columns

    Returns:
        Tuple of (pred_train_log, resid_train_log) as Series
    """
    pred_train_log = pd.Series(aligned["pred_log"].astype(float), index=aligned.index)
    resid_train_log = pd.Series(
        (aligned["actual_log"].astype(float) - aligned["pred_log"].astype(float)).values,
        index=aligned.index,
    )
    return pred_train_log, resid_train_log


def _map_train_returns_to_dataframe(
    merged: pd.DataFrame, pred_train_log: pd.Series, resid_train_log: pd.Series
) -> pd.DataFrame:
    """Map train log-returns back to merged dataframe.

    Args:
        merged: Merged dataframe
        pred_train_log: Predicted train log-returns Series
        resid_train_log: Residual train log-returns Series

    Returns:
        Updated dataframe with train ARIMA columns
    """
    merged["date"] = pd.to_datetime(merged["date"])  # type: ignore[index]
    tm = merged["split"] == "train"
    merged.loc[tm, "arima_pred_return"] = merged.loc[tm, "date"].map(pred_train_log)
    merged.loc[tm, "arima_residual_return"] = merged.loc[tm, "date"].map(resid_train_log)
    return merged


def _add_train_sarima_columns(merged: pd.DataFrame, fitted_model: Any) -> pd.DataFrame:
    """Add SARIMA train predictions and residuals (log-returns) using fitted model.

    Args:
        merged: Merged dataframe with train/test split
        fitted_model: Fitted SARIMA model with fittedvalues attribute (log-domain)

    Returns:
        Updated dataframe with train ARIMA columns (log domain)
    """
    fitted_vals = getattr(fitted_model, "fittedvalues", None)
    if fitted_vals is None:
        logger.warning("Fitted model has no fittedvalues; skipping train ARIMA columns.")
        return merged

    train_series = _extract_train_series(merged)
    fitted_series = _align_fitted_values_to_train(fitted_vals, train_series)

    aligned = pd.concat(
        [train_series.rename("actual_log"), fitted_series.rename("pred_log")],
        axis=1,
    ).dropna()

    if aligned.empty:
        return merged

    pred_train_log, resid_train_log = _compute_train_log_returns(aligned)
    return _map_train_returns_to_dataframe(merged, pred_train_log, resid_train_log)


def save_garch_dataset(results: dict[str, Any], fitted_model: Any | None = None) -> None:
    """Save dataset for subsequent GARCH training using SARIMA outputs.

    Builds a dataset based on the split input CSV and adds SARIMA-based
    columns for both train and test horizons. GARCH needs train residuals
    for MLE estimation and test residuals for evaluation.

    Columns:
    - weighted_return: arithmetic return computed from weighted_log_return (if available)
    - arima_pred_return: PREDICTED LOG-RETURN (kept name for compatibility)
    - arima_residual_return: residual LOG-RETURN (actual_log - pred_log)

    Args:
        results: Evaluation results containing 'predictions', 'actuals', 'dates'.
        fitted_model: Optional fitted SARIMA model for train predictions
    """
    required = {"predictions", "actuals", "dates"}
    if not required.issubset(results):
        logger.warning("Results missing keys for GARCH dataset; skipping save.")
        return

    logger.info("Preparing dataset for GARCH training")
    split_df = pd.read_csv(WEIGHTED_LOG_RETURNS_SPLIT_FILE, parse_dates=["date"])  # type: ignore[arg-type]

    # Base arithmetic returns from log (useful for plots)
    if "weighted_log_return" in split_df.columns:
        split_df["weighted_return"] = np.expm1(split_df["weighted_log_return"])  # type: ignore[index]

    # Add test SARIMA columns
    merged = _add_test_sarima_columns(results, split_df)

    # Add train SARIMA columns if model available
    try:
        if fitted_model is None:
            from src.arima.training_arima.training_arima import load_trained_model  # type: ignore

            fitted_model, _ = load_trained_model()
        merged = _add_train_sarima_columns(merged, fitted_model)
    except (ValueError, AttributeError, FileNotFoundError) as e:
        logger.warning(f"Could not add train ARIMA columns: {e}")

    GARCH_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(GARCH_DATASET_FILE, index=False)
    logger.info(f"Saved GARCH dataset: {GARCH_DATASET_FILE}")
