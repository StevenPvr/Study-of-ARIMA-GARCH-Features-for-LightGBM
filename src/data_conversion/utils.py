"""Utility functions for data conversion module."""

from __future__ import annotations

import pandas as pd


def _validate_dataframe_not_empty(df: pd.DataFrame, name: str) -> None:
    """Validate that DataFrame is not empty.

    Args:
        df: DataFrame to validate.
        name: Name of the DataFrame for error messages.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        msg = f"{name} DataFrame is empty"
        raise ValueError(msg)


def _validate_columns(df: pd.DataFrame, required_columns: set[str], df_name: str) -> None:
    """Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Set of required column names.
        df_name: Name of the DataFrame for error messages.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        msg = f"Missing required columns in {df_name}: {missing_columns}"
        raise ValueError(msg)


def _validate_weight_sum(weighted_returns: pd.DataFrame) -> None:
    """Validate that weight sums are positive for all dates.

    Args:
        weighted_returns: DataFrame with weight_sum column.

    Raises:
        ValueError: If any weight sum is zero or negative.
    """
    if (weighted_returns["weight_sum"] <= 0).any():
        bad_dates = (
            weighted_returns.loc[weighted_returns["weight_sum"] <= 0, "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
        )
        raise ValueError(
            "Sum of weights is zero or negative for some dates: " + ", ".join(bad_dates[:5])
        )
