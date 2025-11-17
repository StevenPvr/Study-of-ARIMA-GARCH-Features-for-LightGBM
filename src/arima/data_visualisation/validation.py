"""Validation utilities for visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_seasonal_params(model: str, period: int | None) -> None:
    """Validate seasonal decomposition parameters.

    Args:
        model: Decomposition model ('additive' or 'multiplicative').
        period: Seasonal period (must be positive integer if not None).

    Raises:
        ValueError: If model is invalid or period is invalid.
    """
    valid_models = {"additive", "multiplicative"}
    if model not in valid_models:
        msg = f"model must be one of {valid_models}, got '{model}'"
        raise ValueError(msg)

    if period is not None and period <= 0:
        msg = f"period must be positive, got {period}"
        raise ValueError(msg)


def validate_array_lengths(
    actuals: np.ndarray,
    predictions: np.ndarray,
    test_series: pd.Series,
) -> None:
    """Validate that arrays have matching lengths.

    Args:
        actuals: Actual values array.
        predictions: Predicted values array.
        test_series: Test set time series.

    Raises:
        ValueError: If lengths don't match.
    """
    if len(actuals) != len(predictions):
        msg = (
            f"actuals and predictions must have same length, "
            f"got {len(actuals)} and {len(predictions)}"
        )
        raise ValueError(msg)

    if len(actuals) != len(test_series):
        msg = (
            f"actuals length ({len(actuals)}) must match "
            f"test_series length ({len(test_series)})"
        )
        raise ValueError(msg)


def validate_minimum_periods(series: pd.Series, period: int, min_periods: int = 2) -> None:
    """Validate that series has enough data for seasonal decomposition.

    Args:
        series: Time series to validate.
        period: Seasonal period.
        min_periods: Minimum number of complete periods required.

    Raises:
        ValueError: If series is too short for the requested period.
    """
    if len(series) < period * min_periods:
        msg = (
            f"Series length ({len(series)}) is insufficient for period {period}. "
            f"Need at least {period * min_periods} observations (got {len(series)})."
        )
        raise ValueError(msg)
