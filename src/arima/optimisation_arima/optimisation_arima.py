"""SARIMA model optimization module."""

from __future__ import annotations

import itertools
from multiprocessing import cpu_count
from typing import Any, cast

import pandas as pd

from src.constants import WEIGHTED_LOG_RETURNS_SPLIT_FILE
from src.utils import get_logger

from .utils import (
    _evaluate_sarima_models,
    _log_optimization_start,
    _prepare_optimization_results,
    _save_optimization_results,
    _select_best_models,
    _validate_seasonal_period,
    _validate_split_data_columns,
    _validate_split_sets_not_empty,
    _validate_split_values,
    _validate_train_series_not_empty,
)

logger = get_logger(__name__)


def load_train_test_data() -> tuple[pd.Series, pd.Series]:
    """
    Load train and test series from split data file.

    Returns:
        Tuple of (train_series, test_series) with date as index

    Raises:
        FileNotFoundError: If split data file doesn't exist
        ValueError: If data file is empty or missing required columns
    """
    if not WEIGHTED_LOG_RETURNS_SPLIT_FILE.exists():
        msg = f"Split data file not found: {WEIGHTED_LOG_RETURNS_SPLIT_FILE}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading train/test data from {WEIGHTED_LOG_RETURNS_SPLIT_FILE}")
    split_data = pd.read_csv(WEIGHTED_LOG_RETURNS_SPLIT_FILE, parse_dates=["date"])

    _validate_split_data_columns(split_data)
    _validate_split_values(split_data)

    train_data = cast(pd.DataFrame, split_data[split_data["split"] == "train"].copy())
    test_data = cast(pd.DataFrame, split_data[split_data["split"] == "test"].copy())

    _validate_split_sets_not_empty(train_data, test_data)

    train_series = cast(pd.Series, train_data.set_index("date")["weighted_log_return"])
    test_series = cast(pd.Series, test_data.set_index("date")["weighted_log_return"])

    logger.info(f"Train set: {len(train_series)} observations")
    logger.info(f"Test set: {len(test_series)} observations")

    return train_series, test_series


def optimize_sarima_models(
    train_series: pd.Series,
    test_series: pd.Series,
    p_range: range = range(6),
    d_range: range = range(3),
    q_range: range = range(6),
    P_range: range = range(3),
    D_range: range = range(2),
    Q_range: range = range(3),
    s: int = 12,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Find optimal SARIMA parameters via exhaustive grid search.

    Tests all combinations of (p, d, q, P, D, Q, s) parameters and selects best models
    based on AIC and BIC criteria. Saves results to files.

    Args:
        train_series: Training time series data
        test_series: Test time series data (for metadata)
        p_range: Range of AR order (p) values to test
        d_range: Range of differencing order (d) values to test
        q_range: Range of MA order (q) values to test
        P_range: Range of seasonal AR order (P) values to test
        D_range: Range of seasonal differencing order (D) values to test
        Q_range: Range of seasonal MA order (Q) values to test
        s: Seasonal period
        n_jobs: Number of parallel jobs. If None, uses cpu_count()-1. If 1, sequential.

    Returns:
        Tuple containing:
            - DataFrame with all optimization results
            - Best AIC model info (dict)
            - Best BIC model info (dict)

    Raises:
        RuntimeError: If no models converged
    """
    _validate_train_series_not_empty(train_series)
    _validate_seasonal_period(s)

    param_combinations = list(
        itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range, [s])
    )

    effective_n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
    _log_optimization_start(
        param_combinations,
        p_range,
        d_range,
        q_range,
        P_range,
        D_range,
        Q_range,
        s,
        effective_n_jobs,
    )

    # Use a single, consistent interpretation for parallelism across logging and execution
    results = _evaluate_sarima_models(
        train_series, param_combinations, n_jobs=effective_n_jobs
    )
    results_df = _prepare_optimization_results(results, len(param_combinations))
    best_aic_dict, best_bic_dict = _select_best_models(results_df)

    _save_optimization_results(results_df, best_aic_dict, best_bic_dict, train_series, test_series)

    logger.info(f"Best AIC model: {best_aic_dict['params']}")
    logger.info(f"Best BIC model: {best_bic_dict['params']}")

    return results_df, best_aic_dict, best_bic_dict
