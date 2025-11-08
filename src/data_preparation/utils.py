"""Utility functions for data preparation operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def _validate_train_ratio(train_ratio: float) -> None:
    """Validate train ratio is between 0 and 1.

    Args:
        train_ratio: Proportion of data for training.

    Raises:
        ValueError: If train_ratio is not between 0 and 1.
    """
    if train_ratio <= 0 or train_ratio >= 1:
        msg = f"train_ratio must be between 0 and 1, got {train_ratio}"
        raise ValueError(msg)


def _validate_file_exists(file_path: Path, file_name: str) -> None:
    """Validate that a file exists.

    Args:
        file_path: Path to the file to check.
        file_name: Name of the file for error message.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not file_path.exists():
        msg = f"{file_name} not found: {file_path}"
        raise FileNotFoundError(msg)


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


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Set of required column names.

    Raises:
        KeyError: If any required column is missing.
    """
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise KeyError(msg)


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
        pass
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
        logger.info(f"  Period: {date_range}")


def _log_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_file: str,
) -> None:
    """Log summary of split operation.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        output_file: Path where split data was saved.
    """
    n_total = len(train_df) + len(test_df)
    logger.info(f"Split complete: {n_total} total observations")

    if n_total == 0:
        logger.warning("No data to split")
        return

    train_pct = len(train_df) / n_total * 100
    test_pct = len(test_df) / n_total * 100

    logger.info(f"Train set: {len(train_df)} observations ({train_pct:.1f}%)")
    _log_date_range(train_df, "Train")

    logger.info(f"Test set: {len(test_df)} observations ({test_pct:.1f}%)")
    _log_date_range(test_df, "Test")

    logger.info(f"Saved split data to {output_file}")


def _log_series_summary(train_series: pd.Series, test_series: pd.Series) -> None:
    """Log summary of loaded train/test series.

    Args:
        train_series: Training time series.
        test_series: Test time series.
    """
    logger.info(f"Train set: {len(train_series)} observations")
    train_date_range = _format_date_range(
        train_series.index.min(),
        train_series.index.max(),
    )
    if train_date_range:
        logger.info(f"  Period: {train_date_range}")

    logger.info(f"Test set: {len(test_series)} observations")
    test_date_range = _format_date_range(
        test_series.index.min(),
        test_series.index.max(),
    )
    if test_date_range:
        logger.info(f"  Period: {test_date_range}")

    train_stats = {
        "mean": train_series.mean(),
        "std": train_series.std(),
        "min": train_series.min(),
        "max": train_series.max(),
    }
    logger.info(
        f"Train statistics - Mean: {train_stats['mean']:.6f}, "
        f"Std: {train_stats['std']:.6f}, "
        f"Min: {train_stats['min']:.6f}, "
        f"Max: {train_stats['max']:.6f}"
    )


def _save_split_data(split_df: pd.DataFrame, output_file: str) -> None:
    """Save split data to CSV file.

    Args:
        split_df: DataFrame with split column.
        output_file: Path to save the CSV file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_file, index=False)
