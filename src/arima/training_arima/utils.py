"""Utility functions for SARIMA model training."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _validate_non_negative_integers(
    values: tuple[int, ...], param_name: str, expected_len: int
) -> None:
    """Validate that values are non-negative integers of expected length."""
    if len(values) != expected_len:
        msg = f"Invalid {param_name}: {values}. Must be tuple of {expected_len} values"
        raise ValueError(msg)

    if any(not isinstance(x, int) or x < 0 for x in values):
        msg = f"Invalid {param_name}: {values}. All values must be non-negative integers"
        raise ValueError(msg)


def _validate_order(order: tuple[int, int, int]) -> None:
    """Validate SARIMA order parameter."""
    _validate_non_negative_integers(order, "order", 3)


def _validate_seasonal_order(seasonal_order: tuple[int, int, int, int]) -> None:
    """Validate SARIMA seasonal order parameter."""
    if len(seasonal_order) != 4:
        msg = f"Invalid seasonal_order parameter: {seasonal_order}. " "Must be tuple of 4 values"
        raise ValueError(msg)

    _validate_non_negative_integers(seasonal_order[:3], "seasonal_order (P, D, Q)", 3)

    if seasonal_order[3] <= 0:
        msg = f"Invalid seasonal period: {seasonal_order[3]}. Must be positive"
        raise ValueError(msg)


def _validate_sarima_parameters(
    train_series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> None:
    """
    Validate SARIMA model parameters.

    Args:
        train_series: Training time series data
        order: SARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)

    Raises:
        ValueError: If parameters are invalid
    """
    if train_series.empty:
        msg = "Training series cannot be empty"
        raise ValueError(msg)

    _validate_order(order)
    _validate_seasonal_order(seasonal_order)


def _extract_model_parameters(
    model_info: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """
    Extract order and seasonal_order from model_info dictionary.

    Args:
        model_info: Dictionary with model parameters

    Returns:
        Tuple of (order, seasonal_order)

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["p", "d", "q", "P", "D", "Q", "s"]
    missing_keys = [k for k in required_keys if k not in model_info]
    if missing_keys:
        msg = f"Model info missing required keys: {missing_keys}"
        raise ValueError(msg)

    order = (model_info["p"], model_info["d"], model_info["q"])
    seasonal_order = (
        int(model_info["P"]),
        int(model_info["D"]),
        int(model_info["Q"]),
        int(model_info["s"]),
    )
    return order, seasonal_order
