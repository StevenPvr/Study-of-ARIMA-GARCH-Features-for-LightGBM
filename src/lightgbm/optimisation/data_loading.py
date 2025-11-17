"""Data loading utilities for LightGBM optimization."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from src.constants import LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION
from src.utils import (
    extract_features_and_target,
    get_logger,
    has_both_splits,
    log_split_dates,
    read_dataset_file,
    remove_metadata_columns,
    validate_temporal_split,
    validate_ticker_id,
)

logger = get_logger(__name__)


def _filter_to_train_split(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to train split only.

    Args:
        df: Input DataFrame.

    Returns:
        Filtered DataFrame with only train split.

    Raises:
        ValueError: If no train data found.
    """
    if "split" not in df.columns:
        return df.copy()

    df_filtered = cast(pd.DataFrame, df[df["split"] == "train"].copy())
    logger.info(f"Filtered to train split: {len(df_filtered)} rows")

    if df_filtered.empty:
        raise ValueError("No data found for split 'train'")

    return df_filtered


def _sample_recent_data(df: pd.DataFrame, sample_fraction: float) -> pd.DataFrame:
    """Sample most recent contiguous block of data.

    Args:
        df: Input DataFrame (must be sorted by date).
        sample_fraction: Fraction of data to retain (0.0 to 1.0).

    Returns:
        Sampled DataFrame.
    """
    n_samples = max(1, int(len(df) * sample_fraction))
    start_idx = len(df) - n_samples
    df_sampled = df.iloc[start_idx:].copy()
    logger.info(
        "Retained the most recent %d rows (%.0f%% of train data) for optimization",
        n_samples,
        sample_fraction * 100,
    )
    return df_sampled


def _filter_train_split(df: pd.DataFrame, sample_fraction: float = 1.0) -> pd.DataFrame:
    """Filter dataset to train split and optionally sample it.

    Args:
        df: Input DataFrame.
        sample_fraction: Fraction of train data to use (0.0 to 1.0).

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If no train data found or invalid sample_fraction.
    """
    if sample_fraction <= 0.0 or sample_fraction > 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    # Filter to train split
    df_filtered = _filter_to_train_split(df)

    # Ensure chronological order before any sampling to avoid leakage
    if "date" in df_filtered.columns:
        df_filtered = df_filtered.sort_values("date").reset_index(drop=True)

    # Sample data if fraction < 1.0 using the most recent contiguous block
    if sample_fraction < 1.0:
        df_filtered = _sample_recent_data(df_filtered, sample_fraction)

    return df_filtered


def _validate_temporal_order(df: pd.DataFrame) -> None:
    """Validate temporal order of train/test split to prevent look-ahead bias.

    Args:
        df: DataFrame with split and date columns.

    Raises:
        ValueError: If temporal order is violated.
    """
    if "split" not in df.columns or "date" not in df.columns:
        return

    if not has_both_splits(df):
        return

    if "ticker" in df.columns:
        validate_temporal_split(
            df, ticker_col="ticker", function_name="load_dataset (optimization)"
        )
    else:
        validate_temporal_split(df, function_name="load_dataset (optimization)")


def _log_split_dates(df: pd.DataFrame) -> None:
    """Log split dates if split column exists.

    Args:
        df: DataFrame with split and date columns.
    """
    # Log aggregated summary only (no per-ticker details)
    log_split_dates(df, function_name="load_dataset (optimization)")


def _log_simple_date_range(df: pd.DataFrame) -> None:
    """Log simple date range without split column.

    Args:
        df: DataFrame with date column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    logger.info(
        f"load_dataset (optimization): Data date range - "
        f"{df['date'].min().date()} to {df['date'].max().date()} "
        f"({len(df)} observations)"
    )


def _log_date_range(df: pd.DataFrame) -> None:
    """Log date range of loaded data for monitoring.

    Args:
        df: DataFrame with date column.
    """
    if "split" in df.columns and "date" in df.columns:
        _log_split_dates(df)
    elif "date" in df.columns:
        _log_simple_date_range(df)


def load_dataset(
    dataset_path: Path,
    sample_fraction: float = LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Args:
        dataset_path: Path to the dataset CSV file.
        sample_fraction: Fraction of train data to use for optimization (default: 50%).

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    df = read_dataset_file(dataset_path)

    # Validate temporal order if split column exists (before filtering)
    # This ensures train/test separation is correct in the original dataset
    _validate_temporal_order(df)

    # Filter to train period first to avoid encoding leakage
    df = _filter_train_split(df, sample_fraction=sample_fraction)

    # Log date ranges for monitoring after filtering
    _log_date_range(df)

    # Validate ticker_id exists if ticker is present
    validate_ticker_id(df)

    # Remove metadata columns and extract features/target
    df = remove_metadata_columns(df)
    X, y = extract_features_and_target(df)

    # Ensure date, split, log_volatility, volume, ticker, and tickers are not in features
    # Keep ticker_id as it's a numeric encoded feature
    columns_to_exclude = ["date", "split", "log_volatility", "volume", "ticker", "tickers"]
    X = X.drop(columns=[col for col in columns_to_exclude if col in X.columns])

    logger.info(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y
