"""Utility functions for data preparation operations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import get_logger, log_series_summary, validate_train_ratio

logger = get_logger(__name__)

# Re-export for backward compatibility
__all__ = [
    "validate_train_ratio",
    "log_series_summary",
]


def _format_date_range(min_date: Any, max_date: Any) -> str | None:
    """Format date range for logging.

    Args:
        min_date: Minimum date object.
        max_date: Maximum date object.

    Returns:
        Formatted date range string or None if formatting fails.
    """
    try:
        min_dt = pd.to_datetime(min_date)
        max_dt = pd.to_datetime(max_date)
        min_date_obj = min_dt.date()
        max_date_obj = max_dt.date()
        return f"{min_date_obj} → {max_date_obj}"
    except (AttributeError, TypeError, ValueError):
        return None


def _log_date_range(df: pd.DataFrame, label: str) -> None:
    """Log date range for a DataFrame.

    Args:
        df: DataFrame with 'date' column.
        label: Label for the date range (e.g., "Train", "Test").
    """
    date_range = _format_date_range(
        pd.to_datetime(df["date"].min()),
        pd.to_datetime(df["date"].max()),
    )
    if date_range:
        logger.info("%s period: %s", label, date_range)


# Note: log_series_summary is now imported from src.utils for consistency
# The function is re-exported above for backward compatibility
