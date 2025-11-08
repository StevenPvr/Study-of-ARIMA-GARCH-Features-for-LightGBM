"""Data preparation functions for ARIMA models."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.constants import (
    TIMESERIES_SPLIT_N_SPLITS,
    TRAIN_RATIO_DEFAULT,
    WEIGHTED_LOG_RETURNS_FILE,
    WEIGHTED_LOG_RETURNS_SPLIT_FILE,
)
from src.data_preparation.utils import (
    _log_series_summary,
    _log_split_summary,
    _save_split_data,
    _validate_dataframe_not_empty,
    _validate_file_exists,
    _validate_required_columns,
    _validate_train_ratio,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _load_and_clean_data(input_file: str) -> pd.DataFrame:
    """Load and clean weighted log returns data.

    Args:
        input_file: Path to weighted log returns CSV.

    Returns:
        Cleaned DataFrame sorted by date.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If DataFrame is empty after cleaning.
        KeyError: If required columns are missing.
    """
    input_path = Path(input_file)
    _validate_file_exists(input_path, "Input file")

    logger.info(f"Loading data from {input_file}")
    aggregated_returns = pd.read_csv(input_file, parse_dates=["date"])

    _validate_required_columns(aggregated_returns, {"date", "weighted_log_return"})

    aggregated_returns = aggregated_returns.dropna(subset=["weighted_log_return"])
    _validate_dataframe_not_empty(aggregated_returns, "Cleaned data")

    return aggregated_returns.sort_values("date").reset_index(drop=True)


def _perform_temporal_split(
    data: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform temporal split on time series data using TimeSeriesSplit.

    Uses sklearn's TimeSeriesSplit to ensure proper temporal ordering
    and prevent data leakage.

    Args:
        data: DataFrame with time series data.
        train_ratio: Proportion of data for training.

    Returns:
        Tuple of (train_df, test_df) with split column added.

    Raises:
        ValueError: If DataFrame is empty or too small for splitting.
    """
    _validate_dataframe_not_empty(data, "Input data")
    n_total = len(data)

    if n_total < 2:
        msg = f"DataFrame must have at least 2 rows for splitting, got {n_total}"
        raise ValueError(msg)

    n_test = n_total - int(n_total * train_ratio)

    # TimeSeriesSplit requires n_splits >= 2. Use n_splits=2 and take the last split
    # which gives us the desired train/test split with test_size
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLIT_N_SPLITS, test_size=n_test)

    # Get the split indices - use the last split which matches our desired ratio
    splits = list(tscv.split(data))
    train_indices, test_indices = splits[-1]

    train_df = data.iloc[train_indices].copy()
    train_df["split"] = "train"

    test_df = data.iloc[test_indices].copy()
    test_df["split"] = "test"

    return train_df, test_df


def split_train_test(
    train_ratio: float = TRAIN_RATIO_DEFAULT,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Split time series data into train and test sets.

    Performs temporal split using TimeSeriesSplit from scikit-learn
    (80% train / 20% test by default). Ensures proper temporal ordering
    and prevents data leakage.
    Saves result to CSV with 'split' column indicating train/test.

    Args:
        train_ratio: Proportion of data for training (default: TRAIN_RATIO_DEFAULT).
        input_file: Path to weighted log returns CSV. If None, uses default.
        output_file: Path to save split data CSV. If None, uses default.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If train_ratio is not between 0 and 1.
    """
    _validate_train_ratio(train_ratio)

    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_FILE)
    if output_file is None:
        output_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)

    aggregated_returns = _load_and_clean_data(input_file)
    train_df, test_df = _perform_temporal_split(aggregated_returns, train_ratio)
    split_df = pd.concat([train_df, test_df], ignore_index=True)

    _save_split_data(split_df, output_file)
    _log_split_summary(train_df, test_df, output_file)


def _load_split_dataframe(input_file: str) -> pd.DataFrame:
    """Load split data from CSV file.

    Args:
        input_file: Path to split data CSV.

    Returns:
        DataFrame with split data.

    Raises:
        FileNotFoundError: If split data file doesn't exist.
        ValueError: If DataFrame is empty.
        KeyError: If required columns are missing.
    """
    input_path = Path(input_file)
    _validate_file_exists(input_path, "Split data file")

    logger.info(f"Loading train/test data from {input_file}")
    try:
        split_data = pd.read_csv(input_file, parse_dates=["date"])
    except pd.errors.EmptyDataError:
        msg = "Split data file is empty"
        raise ValueError(msg) from None

    _validate_dataframe_not_empty(split_data, "Split data")

    return split_data


def _extract_train_test_series(
    split_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Extract train and test series from split data.

    Args:
        split_data: DataFrame with split column.

    Returns:
        Tuple of (train_series, test_series) with date as index.

    Raises:
        ValueError: If train or test data is empty.
        KeyError: If required columns are missing.
    """
    _validate_required_columns(split_data, {"date", "weighted_log_return", "split"})

    train_data = split_data[split_data["split"] == "train"].copy()
    test_data = split_data[split_data["split"] == "test"].copy()

    if train_data.empty:
        msg = "Train data is empty after splitting"
        raise ValueError(msg)
    if test_data.empty:
        msg = "Test data is empty after splitting"
        raise ValueError(msg)

    train_series = cast(pd.Series, train_data.set_index("date")["weighted_log_return"])
    test_series = cast(pd.Series, test_data.set_index("date")["weighted_log_return"])

    return train_series, test_series


def load_train_test_data(
    input_file: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load train and test series from split data file.

    Args:
        input_file: Path to split data CSV. If None, uses default.

    Returns:
        Tuple of (train_series, test_series) with date as index.

    Raises:
        FileNotFoundError: If split data file doesn't exist.
    """
    if input_file is None:
        input_file = str(WEIGHTED_LOG_RETURNS_SPLIT_FILE)

    split_data = _load_split_dataframe(input_file)
    train_series, test_series = _extract_train_test_series(split_data)
    _log_series_summary(train_series, test_series)

    return train_series, test_series
