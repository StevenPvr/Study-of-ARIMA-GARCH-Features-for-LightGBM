"""Validation functions for data preparation."""

from __future__ import annotations

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def validate_lag_value(lag: int) -> None:
    """Validate that lag value is positive.

    Args:
        lag: Lag value to validate.

    Raises:
        ValueError: If lag is not positive.
    """
    if lag <= 0:
        msg = f"Lag value must be positive, received {lag}"
        raise ValueError(msg)


def validate_data_sorted_by_date(df: pd.DataFrame, function_name: str) -> None:
    """Validate that DataFrame is sorted by date to prevent data leakage.

    For ticker-level data, validates sorting by (ticker, date).
    For aggregated data, validates sorting by date.

    Args:
        df: DataFrame to validate.
        function_name: Name of the calling function for error messages.

    Raises:
        ValueError: If DataFrame is not properly sorted by date.
    """
    if "date" not in df.columns:
        logger.warning(f"{function_name}: No 'date' column found, cannot validate sorting")
        return

    if df.empty:
        return

    # Check if data is sorted
    if "ticker" in df.columns:
        # For ticker-level data, check sorting by (ticker, date)
        sort_columns = ["ticker", "date"]
        df_sorted = df.sort_values(sort_columns).reset_index(drop=True)
        if not df[sort_columns].equals(df_sorted[sort_columns]):
            msg = (
                f"{function_name}: DataFrame must be sorted by (ticker, date) "
                "to prevent data leakage. Use sort_values(['ticker', 'date']) before calling."
            )
            raise ValueError(msg)
    else:
        # For aggregated data, check sorting by date
        df_sorted = df.sort_values("date").reset_index(drop=True)
        if not df["date"].equals(df_sorted["date"]):
            msg = (
                f"{function_name}: DataFrame must be sorted by date "
                "to prevent data leakage. Use sort_values('date') before calling."
            )
            raise ValueError(msg)


def validate_garch_columns(df: pd.DataFrame) -> None:
    """Validate that required columns are present in GARCH data.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = ["date", "weighted_closing", "log_weighted_return"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
