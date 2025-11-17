"""Shared utilities for LightGBM modules."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import cast

import pandas as pd


def setup_project_path() -> None:
    """Set up project root path for direct execution of modules.

    This function adds the project root to sys.path when modules are run directly.
    Should be called at the beginning of main functions in executable modules.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def resolve_dataset_path(dataset_path: Path | None, default_path: Path) -> Path:
    """Resolve dataset path with fallback logic.

    Args:
        dataset_path: User-provided path or None.
        default_path: Default path to use.

    Returns:
        Resolved path to existing dataset file.

    Raises:
        FileNotFoundError: If no dataset file is found.
    """
    from src.utils import get_logger

    logger = get_logger(__name__)

    if dataset_path is None:
        dataset_path = default_path.with_suffix(".parquet")

    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}, trying .csv alternative")
        dataset_path = dataset_path.with_suffix(".csv")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path} (tried .parquet and .csv)")

    return dataset_path


def load_test_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load test split from dataset.

    Args:
        dataset_path: Path to dataset file (.parquet or .csv).

    Returns:
        Tuple of (X_test, y_test).

    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If dataset format is invalid.
    """
    from src.utils import get_logger

    logger = get_logger(__name__)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df: pd.DataFrame
    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

    # Filter for test split
    df_test = cast(pd.DataFrame, df[df["split"] == "test"].copy())
    if df_test.empty:
        raise ValueError("No test data found in dataset")

    # Extract features and target using centralized function
    from src.utils.transforms import extract_features_and_target

    X_test, y_test = extract_features_and_target(df_test)

    logger.info(f"Loaded test data: {len(X_test)} samples, {len(X_test.columns)} features")
    return X_test, y_test


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target from a dataset DataFrame.

    Removes metadata columns, extracts features and target, and excludes
    non-feature columns like dates, splits, etc.

    Args:
        df: Dataset DataFrame with all columns.

    Returns:
        Tuple of (X, y) where X are features and y is target.
    """
    from src.utils import get_logger
    from src.utils.transforms import extract_features_and_target, remove_metadata_columns

    logger = get_logger(__name__)

    df = remove_metadata_columns(df)
    X, y = extract_features_and_target(df)

    # Ensure date, split, log_volatility, volume, ticker, and tickers are not in features
    # Keep ticker_id as it's a numeric encoded feature
    columns_to_exclude = ["date", "split", "log_volatility", "volume", "ticker", "tickers"]
    X = X.drop(columns=[col for col in columns_to_exclude if col in X.columns])

    logger.info(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


__all__ = [
    "setup_project_path",
    "resolve_dataset_path",
    "load_test_data",
    "prepare_features_and_target",
]
