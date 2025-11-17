"""Data loading utilities for LightGBM evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
from src.utils import (
    filter_by_split,
    get_logger,
    get_parquet_path,
    load_csv_file,
    load_parquet_file,
    log_split_dates,
    stable_ticker_id,
    validate_file_exists,
    validate_temporal_split,
)

logger = get_logger(__name__)


def _add_ticker_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `ticker_id` exists when `ticker` column is present.

    Uses stable hashing to keep ids consistent with training without
    requiring shared state. Keeps `ticker` for logging until dropped.
    """
    if "ticker" not in df.columns:
        return df
    out = df.copy()
    out["ticker_id"] = stable_ticker_id(out["ticker"])  # int32
    return out


def _validate_temporal_order(df: pd.DataFrame) -> None:
    """Validate temporal order of train/test split to prevent look-ahead bias.

    Args:
        df: DataFrame with split and date columns.

    Raises:
        ValueError: If temporal order is violated.
    """
    if "split" not in df.columns or "date" not in df.columns:
        return

    has_train = (df["split"] == "train").any()
    has_test = (df["split"] == "test").any()

    if not (has_train and has_test):
        return

    if "ticker" in df.columns:
        validate_temporal_split(df, ticker_col="ticker", function_name="load_dataset (eval)")
    else:
        validate_temporal_split(df, function_name="load_dataset (eval)")


def _log_date_range_with_split(df: pd.DataFrame, split: str) -> None:
    """Log date range when split column exists.

    Args:
        df: DataFrame with date and split columns.
        split: Split name for logging.
    """
    # For eval, log only aggregated summary (not per-ticker) to reduce verbosity
    log_split_dates(df, function_name=f"load_dataset (eval, split={split})")


def _log_date_range_without_split(df: pd.DataFrame) -> None:
    """Log date range when split column doesn't exist.

    Args:
        df: DataFrame with date column but no split column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
    logger.info(
        f"load_dataset (eval): Data date range - {df['date'].min().date()} to "
        f"{df['date'].max().date()} ({len(df)} observations)"
    )


def _log_date_range(df: pd.DataFrame, split: str) -> None:
    """Log date range for monitoring.

    Args:
        df: DataFrame with date column.
        split: Split name for logging.
    """
    if "split" not in df.columns or "date" not in df.columns:
        if "date" in df.columns:
            _log_date_range_without_split(df)
        return

    _log_date_range_with_split(df, split)


def load_dataset(dataset_path: Path, split: str = "test") -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Tries to load Parquet file first, falls back to CSV if Parquet doesn't exist.

    Args:
        dataset_path: Path to the dataset file (CSV or Parquet).
        split: Split to load ('train' or 'test'). Default is 'test'.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    # Determine Parquet path
    parquet_path = get_parquet_path(dataset_path)

    # Try Parquet first, then CSV
    df = load_parquet_file(parquet_path)
    if df is None:
        df = load_csv_file(dataset_path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    # Validate temporal order before filtering
    _validate_temporal_order(df)

    # Filter by split
    df = filter_by_split(df, split)

    # Log date ranges after filtering
    _log_date_range(df, split)

    # Prepare data
    df = _add_ticker_id_column(df)

    from src.lightgbm.shared_utils import prepare_features_and_target

    X, y = prepare_features_and_target(df)

    return X, y


def load_model(model_path: Path) -> Union[lgb.LGBMRegressor, RandomForestRegressor]:
    """Load trained regression model.

    Args:
        model_path: Path to the model joblib file.

    Returns:
        Loaded regression model (LightGBM or RandomForest).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If model type is not supported.
    """
    validate_file_exists(model_path, "Model file")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    if not isinstance(model, (lgb.LGBMRegressor, RandomForestRegressor)):
        raise ValueError(f"Expected LGBMRegressor or RandomForestRegressor, got {type(model)}")

    return model
