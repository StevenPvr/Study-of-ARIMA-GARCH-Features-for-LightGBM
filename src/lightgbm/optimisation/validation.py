"""Data validation utilities for LightGBM optimization."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def _validate_empty_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate that data is not empty.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Raises:
        ValueError: If data is empty.
    """
    if X.empty or y.empty:
        raise ValueError("Features or target cannot be empty")


def _validate_nan_values(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate that data contains no NaN values.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Raises:
        ValueError: If NaN values are found.
    """
    nan_mask_X = X.isna()
    if nan_mask_X.values.any():
        nan_cols = X.columns[nan_mask_X.any()].tolist()
        raise ValueError(f"Features contain NaN values in columns: {nan_cols[:10]}")
    if y.isna().any():
        raise ValueError("Target contains NaN values")


def _validate_infinite_values(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate that data contains no infinite values.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Raises:
        ValueError: If infinite values are found.
    """
    # Check only numeric columns for infinite values
    X_numeric = X.select_dtypes(include=[np.number])
    if not X_numeric.empty:
        X_np = cast(np.ndarray, X_numeric.values)
        if np.isinf(X_np).any():
            inf_cols = X_numeric.columns[np.isinf(X_np).any(axis=0)].tolist()
            raise ValueError(f"Features contain infinite values in columns: {inf_cols[:10]}")

    # Check target (should be numeric)
    if pd.api.types.is_numeric_dtype(y):
        y_np = cast(np.ndarray, y.values)
        if np.isinf(y_np).any():
            raise ValueError("Target contains infinite values")


def validate_optimization_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate data for optimization.

    Args:
        X: Features DataFrame.
        y: Target Series.

    Raises:
        ValueError: If data is invalid.
    """
    _validate_empty_data(X, y)
    _validate_nan_values(X, y)
    _validate_infinite_values(X, y)
