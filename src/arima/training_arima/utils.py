"""Utility functions for ARIMA model training."""

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
    """Validate ARIMA order parameter."""
    _validate_non_negative_integers(order, "order", 3)


def validate_arima_parameters(
    train_series: pd.Series,
    order: tuple[int, int, int],
) -> None:
    """
    Validate ARIMA model parameters.

    Args:
        train_series: Training time series data
        order: ARIMA order (p, d, q)

    Raises:
        ValueError: If parameters are invalid
    """
    if train_series.empty:
        msg = "Training series cannot be empty"
        raise ValueError(msg)

    _validate_order(order)


def extract_model_parameters(
    model_info: dict[str, Any],
) -> tuple[int, int, int]:
    """
    Extract order from model_info dictionary.

    Args:
        model_info: Dictionary with model parameters

    Returns:
        ARIMA order (p, d, q)

    Raises:
        ValueError: If required keys are missing
    """
    required_keys = ["p", "d", "q"]

    # Determine source: top-level or nested under "params"
    if all(k in model_info for k in required_keys):
        src = model_info
    elif isinstance(model_info.get("params"), dict):
        src = model_info["params"]
        missing_keys = [k for k in required_keys if k not in src]
        if missing_keys:
            msg = f"Model info missing required keys: {missing_keys}"
            raise ValueError(msg)
    else:
        msg = f"Model info missing required keys: {required_keys}"
        raise ValueError(msg)

    order = (int(src["p"]), int(src["d"]), int(src["q"]))
    return order
