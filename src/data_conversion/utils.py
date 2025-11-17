"""Utility functions for data conversion module."""

from __future__ import annotations

import pandas as pd

from src.constants import MAX_ERROR_DATES_DISPLAY
from src.utils import validate_dataframe_not_empty, validate_required_columns

__all__ = [
    "validate_dataframe_not_empty",
    "_validate_columns",
    "_merge_weights_with_returns",
    "_compute_daily_weight_totals",
    "_validate_weight_sum",
    "_normalize_weights",
]


def _validate_columns(df: pd.DataFrame, required_columns: set[str], df_name: str) -> None:
    """Validate that DataFrame contains required columns.

    Delegates to src.utils.validate_required_columns() for consistency.

    Args:
        df: DataFrame to validate.
        required_columns: Set of required column names.
        df_name: Name of the DataFrame for error messages.

    Raises:
        ValueError: If required columns are missing.
    """
    validate_required_columns(df, required_columns, df_name)


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
            "Sum of weights is zero or negative for some dates: "
            + ", ".join(bad_dates[:MAX_ERROR_DATES_DISPLAY])
        )


def _merge_weights_with_returns(
    returns_df: pd.DataFrame, liquidity_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Merge liquidity weights with returns DataFrame.

    Supports both static weights (index=ticker) and time-varying weights
    (columns contain 'date' and 'ticker').

    Args:
        returns_df: DataFrame with returns data.
        liquidity_metrics: DataFrame with weights (static or time-varying).

    Returns:
        DataFrame with merged weights, rows without weights dropped.
    """
    if "date" in liquidity_metrics.columns and "ticker" in liquidity_metrics.columns:
        weighted_df = returns_df.merge(
            liquidity_metrics[["date", "ticker", "weight"]], on=["date", "ticker"], how="left"
        )
        weighted_df = weighted_df.dropna(subset=["weight"]).copy()
    else:
        weighted_df = returns_df.merge(
            liquidity_metrics[["weight"]], left_on="ticker", right_index=True
        )
    return weighted_df


def _compute_daily_weight_totals(weighted_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily weight totals from weighted DataFrame.

    Args:
        weighted_df: DataFrame with weight column and date column.

    Returns:
        DataFrame with date and weight_sum columns.
    """
    daily_weight_sum = weighted_df.groupby("date")["weight"].sum()
    return daily_weight_sum.to_frame(name="weight_sum").reset_index()


def _normalize_weights(weighted_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize weights by daily totals.

    Args:
        weighted_df: DataFrame with weight and weight_sum columns.

    Returns:
        DataFrame with added normalized_weight column.
    """
    weighted_df = weighted_df.copy()
    weighted_df["normalized_weight"] = weighted_df["weight"] / weighted_df["weight_sum"]
    return weighted_df
