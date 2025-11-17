"""Data loading utilities for LightGBM training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.path import LIGHTGBM_OPTIMIZATION_RESULTS_FILE
from src.utils import (
    filter_by_split,
    get_logger,
    has_both_splits,
    load_json_data,
    log_split_dates,
    read_dataset_file,
    validate_temporal_split,
)

logger = get_logger(__name__)


def _validate_temporal_order(df: pd.DataFrame, split: str) -> None:
    """Validate temporal order of train/test split to prevent look-ahead bias.

    Args:
        df: Input DataFrame with split and date columns.
        split: Current split being loaded.
    """
    if "split" not in df.columns or "date" not in df.columns:
        return

    if not has_both_splits(df):
        return

    function_name = f"load_dataset (training, split={split})"
    if "ticker" in df.columns:
        validate_temporal_split(df, ticker_col="ticker", function_name=function_name)
    else:
        validate_temporal_split(df, function_name=function_name)


def _log_date_range(df: pd.DataFrame, split: str) -> None:
    """Log date range for monitoring.

    Args:
        df: Input DataFrame with date column.
        split: Current split being loaded.
    """
    if "split" not in df.columns or "date" not in df.columns:
        return

    # Log aggregated summary only (no per-ticker details)
    log_split_dates(df, function_name=f"load_dataset (training, split={split})")


def _ensure_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column is datetime type.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with datetime date column.
    """
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_dataset(dataset_path: Path, split: str = "train") -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Args:
        dataset_path: Path to the dataset CSV file.
        split: Split to load ('train' or 'test'). Default is 'train'.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    df = read_dataset_file(dataset_path)

    # Validate temporal order if split column exists (before filtering)
    _validate_temporal_order(df, split)

    # Ensure no leakage from encoding: filter split first
    df = filter_by_split(df, split)

    # Log date ranges for monitoring after filtering
    _log_date_range(df, split)

    # Log date range if no split column but date exists
    if "split" not in df.columns and "date" in df.columns:
        df = _ensure_datetime_column(df)
        logger.info(
            f"load_dataset (training): Data date range - {df['date'].min().date()} to "
            f"{df['date'].max().date()} ({len(df)} observations)"
        )

    # Validate ticker_id requirement
    if "ticker" in df.columns and "ticker_id" not in df.columns:
        raise ValueError(
            "Dataset must contain a precomputed 'ticker_id' column. "
            "Re-run lightgbm data preparation pipeline to generate encoded datasets."
        )

    from src.lightgbm.shared_utils import prepare_features_and_target

    X, y = prepare_features_and_target(df)

    return X, y


def load_optimization_results(
    results_path: Path = LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
) -> dict[str, Any]:
    """Load optimization results from JSON file.

    Args:
        results_path: Path to optimization results JSON file.

    Returns:
        Dictionary with optimization results for both datasets.

    Raises:
        FileNotFoundError: If results file does not exist.
        ValueError: If results file is invalid or missing required keys.
    """
    logger.info(f"Loading optimization results from {results_path}")

    required_keys = ["lightgbm_dataset_complete", "lightgbm_dataset_without_insights"]
    try:
        results = load_json_data(
            results_path,
            required_keys=required_keys,
            context="optimization results",
        )
    except KeyError as err:
        raise ValueError("Missing required key in optimization results") from err

    # Validate nested best_params
    for key in required_keys:
        if "best_params" not in results[key]:
            raise ValueError(f"Missing 'best_params' in {key}")

    logger.info("Optimization results loaded successfully")
    return results
