"""Target column creation and normalization functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.lightgbm.data_preparation.validation import validate_data_sorted_by_date
from src.utils import get_logger

logger = get_logger(__name__)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistency.

    Renames log_weighted_return to weighted_log_return for aggregated data.
    For ticker-level data, keeps log_return as-is (individual ticker returns).

    Args:
        df: DataFrame with potential column naming issues.

    Returns:
        DataFrame with normalized column names.
    """
    df_normalized = df.copy()

    # For aggregated data (S&P 500), rename log_weighted_return to weighted_log_return
    if "log_weighted_return" in df_normalized.columns:
        logger.info("Renaming log_weighted_return to weighted_log_return")
        df_normalized = df_normalized.rename(columns={"log_weighted_return": "weighted_log_return"})
    # For ticker-level data, keep log_return as-is (it's already the individual ticker's log return)
    elif "log_return" in df_normalized.columns and "ticker" in df_normalized.columns:
        # Keep log_return for ticker-level data - no renaming needed
        # We'll use it directly as the target
        pass

    if "weighted_return" in df_normalized.columns:
        logger.info("Removing weighted_return column")
        df_normalized = df_normalized.drop(columns=["weighted_return"])

    # Normalize ticker-level close column name if needed
    if (
        "ticker" in df_normalized.columns
        and "closing" in df_normalized.columns
        and "close" not in df_normalized.columns
    ):
        logger.info("Renaming 'closing' to 'close' for ticker-level data")
        df_normalized = df_normalized.rename(columns={"closing": "close"})

    return df_normalized


def _compute_log_volatility_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_volatility from log_return if missing.

    Args:
        df: DataFrame that may be missing log_volatility.

    Returns:
        DataFrame with log_volatility computed if it was missing.
    """
    if "log_volatility" not in df.columns and "log_return" in df.columns:
        from src.data_preparation.computations import compute_volatility_for_tickers

        df = compute_volatility_for_tickers(df)
        logger.info("Computed log_volatility from log_return")
    return df


def _compute_log_volume_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_volume from volume if missing.

    Args:
        df: DataFrame that may be missing log_volume.

    Returns:
        DataFrame with log_volume computed if it was missing.
    """
    if "log_volume" not in df.columns and "volume" in df.columns:
        # Use log1p to handle zero volumes without infinities
        vol_series = cast(pd.Series, df["volume"])
        df["log_volume"] = np.log1p(vol_series)
        logger.info("Computed log_volume from volume using log1p")
    return df


def _select_target_column(df: pd.DataFrame) -> str:
    """Select the target column name based on priority.

    Priority order:
    1) log_volatility if present
    2) log_volume if present
    3) log_return for ticker-level data
    4) weighted_log_return for aggregated data

    Args:
        df: DataFrame to inspect.

    Returns:
        The chosen target column name.

    Raises:
        ValueError: If none of the expected target columns are present.
    """
    if "log_volatility" in df.columns:
        return "log_volatility"
    if "log_volume" in df.columns:
        return "log_volume"
    if "ticker" in df.columns and "log_return" in df.columns:
        return "log_return"
    if "weighted_log_return" in df.columns:
        return "weighted_log_return"

    msg = (
        "Missing target column: expected one of "
        "['log_volatility', 'log_volume', 'log_return', 'weighted_log_return']"
    )
    raise ValueError(msg)


def _create_temporal_split(df: pd.DataFrame, use_target_date: bool = False) -> pd.DataFrame:
    """Create temporal split column based on dates (80% train, 20% test).

    Creates split based on date column, ensuring temporal order. For ticker-level data,
    the split is created globally across all tickers based on dates to ensure consistent
    train/test periods.

    Args:
        df: DataFrame with 'date' column.
        use_target_date: If True, create split based on date + 1 day (target date).
                        If False, create split based on date (feature date).

    Returns:
        DataFrame with 'split' column added.
    """
    if "date" not in df.columns:
        logger.warning("No 'date' column found; cannot create temporal split")
        return df

    df = df.copy()

    # Determine which date to use for split creation
    if use_target_date:
        # Use target date (date + 1) for split - split will align with target after shift
        split_date_col = df["date"] + pd.Timedelta(days=1)
        logger.info("Creating temporal split based on target dates (date + 1 day)")
    else:
        # Use feature date for split - split will be shifted later to align with target
        split_date_col = df["date"]
        logger.info("Creating temporal split based on feature dates")

    # Get unique dates sorted temporally
    # For ticker-level data, we want a global temporal split across all tickers
    unique_dates = sorted(split_date_col.unique())

    # Create 80/20 temporal split based on dates
    from src.constants import LIGHTGBM_TRAIN_TEST_SPLIT_RATIO

    split_idx = int(len(unique_dates) * LIGHTGBM_TRAIN_TEST_SPLIT_RATIO)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    # Assign split based on date
    df["split"] = split_date_col.apply(lambda d: "train" if d in train_dates else "test")

    logger.info(
        f"Created temporal split: {len(train_dates)} train dates, {len(test_dates)} test dates"
    )

    return df


def _shift_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Shift target column to align features at time t with target at t+1.

    CRITICAL: This function only shifts the TARGET column using shift(-1).
    The split column is NOT shifted here. The split should be created based on
    target dates (t+1) from the beginning to ensure proper temporal alignment.

    This ensures that features at date t predict targets at t+1, preventing
    look-ahead bias.

    Args:
        df: DataFrame with target column.
        target_col: Name of target column to shift.

    Returns:
        DataFrame with shifted target column. Last row(s) with NaN target are dropped.
    """
    logger.info(f"Shifting {target_col} to align with next-day target")
    # CRITICAL: Use shift(-1) to align target with next day (t+1)
    # This ensures features at date t predict target at t+1, preventing look-ahead bias
    # Handle ticker-level data by grouping when a ticker column exists
    if "ticker" in df.columns:
        df[target_col] = df.groupby("ticker")[target_col].shift(-1)
    else:
        df[target_col] = df[target_col].shift(-1)

    # Only drop rows where target is NaN (target shift creates NaN for last row of each ticker)
    # Don't create split here - it will be created later after all NaN rows are removed
    # (after remove_missing_values in _clean_and_prepare_datasets)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    return df


# Keep backward compatibility alias
_shift_target_and_split = _shift_target


def create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create shifted target column for next-day prediction.

    Prefers ``log_volatility`` as target when present; otherwise falls back to
    ``log_volume``, then ``log_return`` (ticker-level) or ``weighted_log_return`` (aggregated).

    The function also computes ``log_volatility`` from ``log_return`` and
    ``log_volume`` from ``volume`` when missing (both with log1p for numerical
    stability), then shifts the chosen target and the ``split`` label by -1 day
    to align features at time t with the target at t+1 (no look-ahead bias).

    IMPORTANT: Call AFTER calculating technical indicators and BEFORE adding lag
    features. The DataFrame must be sorted by date (or ticker+date) to prevent leakage.

    Args:
        df: Input DataFrame with at least one of: ``log_volatility``, ``log_volume``,
            ``log_return`` (with ``ticker``), or ``weighted_log_return``.
            Must be sorted by date (or ticker+date).

    Returns:
        DataFrame with the chosen target column shifted to represent J+1.

    Raises:
        ValueError: If none of the expected target columns are present, or if
            DataFrame is not properly sorted by date.
    """
    # Validate that data is sorted to prevent data leakage
    validate_data_sorted_by_date(df, "_create_target_columns")

    df_shifted = df.copy()

    # Ensure log_volatility exists if log_return is available
    df_shifted = _compute_log_volatility_if_missing(df_shifted)

    # Ensure log_volume exists if raw volume is available
    df_shifted = _compute_log_volume_if_missing(df_shifted)

    # Decide the target column (priority):
    # log_volatility > log_volume > log_return > weighted_log_return
    target_col = _select_target_column(df_shifted)
    logger.info(f"Using {target_col} as prediction target")

    # Shift target column to align features at t with target at t+1
    df_shifted = _shift_target(df_shifted, target_col)

    return df_shifted


def get_target_column_name(df: pd.DataFrame) -> str:
    """Return the model target column name.

    Priority order:
    1) ``log_volatility`` if present
    2) ``log_volume`` if present
    3) ``log_return`` for ticker-level data
    4) ``weighted_log_return`` for aggregated data

    Args:
        df: DataFrame to inspect.

    Returns:
        The chosen target column name.
    """
    if "log_volatility" in df.columns:
        return "log_volatility"
    if "log_volume" in df.columns:
        return "log_volume"
    if "ticker" in df.columns and "log_return" in df.columns:
        return "log_return"
    if "weighted_log_return" in df.columns:
        return "weighted_log_return"
    # Conservative fallback used only for header checks
    return "log_volatility"
