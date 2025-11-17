"""Functions for ticker-level data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    REQUIRED_COLS_TICKER_DATA,
    REQUIRED_COLS_WEIGHTED_RETURNS,
    TRAIN_RATIO_DEFAULT,
)
from src.data_preparation.computations import (
    compute_log_returns_for_tickers,
    compute_log_volume,
    compute_volatility_for_tickers,
)
from src.path import (
    DATA_TICKERS_FULL_FILE,
    DATASET_FILTERED_PARQUET_FILE,
    WEIGHTED_LOG_RETURNS_FILE,
)
from src.utils import (
    get_logger,
    load_and_validate_dataframe,
    load_dataframe,
    log_split_dates,
    save_parquet_and_csv,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
    validate_temporal_split,
    validate_train_ratio,
)

logger = get_logger(__name__)


def _load_tickers_data(input_file: str) -> pd.DataFrame:
    """Load ticker data from provided path.

    Uses common loading function for consistency and reduced duplication.

    Args:
        input_file: Path to ticker data (Parquet or CSV).

    Returns:
        DataFrame with ticker data sorted by date and ticker.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If DataFrame is empty after cleaning.
        KeyError: If required columns are missing.
    """
    try:
        return load_and_validate_dataframe(
            input_file=input_file,
            date_columns=["date"],
            required_columns=list(REQUIRED_COLS_TICKER_DATA),
            sort_by=["ticker", "date"],
            validate_not_empty=True,
            column_renames={"tickers": "ticker"},  # Handle legacy column name
        )
    except ValueError as err:
        # Standardize empty-data error message for tests and callers
        raise ValueError("Ticker data DataFrame is empty") from err


def _split_single_ticker(
    ticker_data: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Split a single ticker's data temporally using a global split date.

    Args:
        ticker_data: DataFrame for a single ticker.
        split_date: Global date for splitting (train: <= split_date, test: > split_date).

    Returns:
        DataFrame with split column added, or None if too small.
    """
    ticker_data = ticker_data.sort_values(by="date").reset_index(drop=True)

    n_total = len(ticker_data)
    if n_total < 2:
        ticker = ticker_data["ticker"].iloc[0] if len(ticker_data) > 0 else "unknown"
        logger.warning("Ticker %s has less than 2 observations, skipping", ticker)
        return None

    # Split based on global date cutoff
    ticker_data["split"] = "train"
    ticker_data.loc[ticker_data["date"] > split_date, "split"] = "test"

    return ticker_data


def _split_all_tickers(
    data: pd.DataFrame,
    split_date: pd.Timestamp,
) -> list[pd.DataFrame]:
    """Split each ticker's data temporally using a global split date.

    Args:
        data: DataFrame with ticker time series data.
        split_date: Global date for splitting all tickers.

    Returns:
        List of DataFrames with split column added.
    """
    split_dfs: list[pd.DataFrame] = []

    for ticker in data["ticker"].unique():
        ticker_data = cast(pd.DataFrame, data[data["ticker"] == ticker].copy())
        split_ticker = _split_single_ticker(ticker_data, split_date)
        if split_ticker is not None:
            split_dfs.append(split_ticker)

    return split_dfs


def _validate_split_result(result_df: pd.DataFrame, function_name: str) -> None:
    """Validate temporal split result.

    Args:
        result_df: DataFrame with split column.
        function_name: Name of calling function for logging.
    """
    validate_temporal_split(
        result_df,
        ticker_col="ticker",
        function_name=function_name,
    )
    log_split_dates(
        result_df,
        ticker_col="ticker",
        function_name=function_name,
    )


def _perform_tickers_temporal_split(
    data: pd.DataFrame,
    train_ratio: float,
) -> pd.DataFrame:
    """Perform temporal split on ticker data using a global split date.

    Calculates a global split date based on train_ratio of the total temporal range,
    then applies this date to all tickers. This ensures no temporal overlap between
    train and test sets when viewed globally across all tickers.

    Args:
        data: DataFrame with ticker time series data.
        train_ratio: Proportion of temporal range for training.

    Returns:
        DataFrame with split column added ('train' or 'test').

    Raises:
        ValueError: If DataFrame is empty or too small for splitting.
    """
    validate_dataframe_not_empty(data, "Input ticker data")
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Calculate global split date based on temporal range
    min_date = data["date"].min()
    max_date = data["date"].max()
    temporal_range = max_date - min_date
    split_date = min_date + temporal_range * train_ratio

    logger.info(
        f"Global temporal split: {min_date.date()} to {max_date.date()}, "
        f"split date: {split_date.date()} (train_ratio={train_ratio:.2f})"
    )

    split_dfs = _split_all_tickers(data, split_date)

    if not split_dfs:
        msg = "No ticker data could be split"
        raise ValueError(msg)

    result_df = pd.concat(split_dfs, ignore_index=True)
    _validate_split_result(result_df, "_perform_tickers_temporal_split")

    return result_df


def _drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not required in outputs.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with unnecessary columns removed.
    """
    cols_to_drop: list[str] = []

    # Drop raw price columns not required downstream; keep only derived metrics.
    # Keep 'close' and 'volume' explicitly — downstream analyses rely on closing prices and volume.
    for col in ("open",):
        if col in df.columns:
            cols_to_drop.append(col)

    # Drop weighted_log_return (not needed once merged into higher-level dataset)
    if "weighted_log_return" in df.columns:
        cols_to_drop.append("weighted_log_return")

    # Keep volume and log_volatility for RF models
    if cols_to_drop:
        logger.info("Dropping columns not required in outputs: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    return df


def _log_split_summary_stats(df: pd.DataFrame, output_file: str) -> None:
    """Log summary statistics for split ticker dataset.

    Uses common log_split_summary for consistency.

    Args:
        df: DataFrame with split column.
        output_file: Path where data was saved.
    """
    from src.utils import log_split_summary

    # Split the dataframe by split column for consistent logging
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    log_split_summary(train_df, test_df, output_file)


def _compute_all_features(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Compute all features (log returns, volatility, volume).

    Args:
        df: DataFrame with ticker data.

    Returns:
        Tuple of (DataFrame with features, number of rows removed).
    """
    if "log_return" not in df.columns:
        msg = "Missing 'log_return' column. Ensure data_conversion computed it."
        raise ValueError(msg)
    n_removed = 0

    logger.info("Computing log volatility per ticker")
    df = compute_volatility_for_tickers(df)

    df = compute_log_volume(df)

    return df, n_removed


def _prepare_output_dataframe(split_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final output DataFrame with required columns.

    Args:
        split_df: DataFrame with all features and split column.

    Returns:
        Cleaned DataFrame with required columns only.

    Raises:
        KeyError: If required columns are missing.
    """
    split_df = _drop_unnecessary_columns(split_df)
    split_df = split_df.rename(columns={"ticker": "tickers"})
    required_output_cols = [
        "date",
        "tickers",
        "close",
        "high",
        "low",
        "volume",
        "log_return",
        "log_volatility",
        "split",
    ]
    missing = [c for c in required_output_cols if c not in split_df.columns]
    if missing:
        raise KeyError(f"Missing required columns for LightGBM pipeline: {missing}")

    split_df = cast(pd.DataFrame, split_df[required_output_cols])
    split_df = split_df.replace([np.inf, -np.inf], pd.NA)
    split_df = cast(
        pd.DataFrame,
        split_df.dropna(subset=["close", "high", "low", "volume", "log_return", "log_volatility"]),
    )
    absence_mask = (split_df["high"] == 0) & (split_df["low"] == 0)
    split_df = cast(pd.DataFrame, split_df.loc[~absence_mask].reset_index(drop=True))

    return split_df


def split_tickers_train_test(
    train_ratio: float = TRAIN_RATIO_DEFAULT,
    input_file: str | None = None,
    output_file: str | None = None,
) -> None:
    """Split ticker time series data into train and test sets.

    Performs temporal split using a global split date calculated from train_ratio
    (80% train / 20% test by default) applied to the entire temporal range.
    This ensures no temporal overlap between train and test sets across all tickers.
    Saves result to Parquet with 'split' column indicating train/test.

    Args:
        train_ratio: Proportion of temporal range for training (default: TRAIN_RATIO_DEFAULT).
        input_file: Path to ticker data CSV. If None, uses default.
        output_file: Path to save split data Parquet. If None, uses default.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If train_ratio is not between 0 and 1.
    """
    validate_train_ratio(train_ratio)

    if input_file is None:
        input_file = str(DATASET_FILTERED_PARQUET_FILE)
    if output_file is None:
        output_file = str(DATA_TICKERS_FULL_FILE)

    ticker_data = _load_tickers_data(input_file)
    split_df = _perform_tickers_temporal_split(ticker_data, train_ratio)

    # Compute features after split to preserve original split counts
    if "log_return" not in split_df.columns:
        logger.info("Computing log_return per ticker")
        split_df = compute_log_returns_for_tickers(split_df)
    logger.info("Computing log volatility per ticker")
    split_df = compute_volatility_for_tickers(split_df)
    split_df = compute_log_volume(split_df)

    _validate_split_result(split_df, "split_tickers_train_test")
    split_df = _prepare_output_dataframe(split_df)
    save_parquet_and_csv(split_df, output_file)
    _log_split_summary_stats(split_df, output_file)


def _load_weighted_returns_file(
    weighted_returns_file: Path | str,
) -> pd.DataFrame | None:
    """Load weighted returns file (Parquet or CSV).

    Delegates to src.utils.load_dataframe() with exception handling.

    Args:
        weighted_returns_file: Path to weighted log returns file.

    Returns:
        DataFrame with weighted returns or None if file doesn't exist.
    """
    weighted_returns_path = Path(weighted_returns_file)

    try:
        weighted_df = load_dataframe(
            weighted_returns_path,
            date_columns=["date"],
            required_columns=list(REQUIRED_COLS_WEIGHTED_RETURNS),
            validate_not_empty=False,
        )
        return weighted_df
    except FileNotFoundError:
        logger.warning(
            "Weighted returns file not found: %s. Skipping weighted_log_return column.",
            weighted_returns_path,
        )
        return None


def _merge_weighted_returns(df: pd.DataFrame, weighted_df: pd.DataFrame) -> pd.DataFrame:
    """Merge weighted returns into DataFrame.

    Args:
        df: DataFrame with date column.
        weighted_df: DataFrame with weighted log returns.

    Returns:
        DataFrame with weighted_log_return column added.
    """
    validate_required_columns(weighted_df, REQUIRED_COLS_WEIGHTED_RETURNS)
    df["date"] = pd.to_datetime(df["date"])

    df = df.merge(
        weighted_df[["date", "weighted_log_return"]],
        on="date",
        how="left",
    )

    n_missing_weighted = df["weighted_log_return"].isna().sum()
    if n_missing_weighted > 0:
        logger.warning(
            "%d rows have missing weighted_log_return (dates not in weighted returns file)",
            n_missing_weighted,
        )
    else:
        logger.info("Successfully added weighted_log_return column")

    return df


def _add_weighted_returns_from_file(
    df: pd.DataFrame,
    weighted_returns_file: Path | str | None,
) -> pd.DataFrame:
    """Add weighted log returns from file to DataFrame.

    Loads from Parquet if available (faster), otherwise falls back to CSV.

    Args:
        df: DataFrame with date column.
        weighted_returns_file: Path to weighted log returns file (Parquet or CSV).

    Returns:
        DataFrame with weighted_log_return column added if file exists.
    """
    if weighted_returns_file is None:
        weighted_returns_file = WEIGHTED_LOG_RETURNS_FILE

    weighted_df = _load_weighted_returns_file(weighted_returns_file)
    if weighted_df is None:
        return df

    return _merge_weighted_returns(df, weighted_df)


def add_log_returns_to_ticker_parquet(
    input_file: Path | str,
    output_file: Path | str | None = None,
    weighted_returns_file: Path | str | None = None,
) -> pd.DataFrame:
    """Add log returns and weighted log returns to ticker-level parquet file.

    Computes log returns per ticker from close prices and adds weighted log returns
    by merging with weighted log returns file. The log_return column is calculated as
    log(close_t / close_{t-1}) per ticker. The weighted_log_return column comes
    from the aggregated weighted returns file (same value for all tickers on same date).

    Args:
        input_file: Path to input parquet file with ticker data.
        output_file: Path to output parquet file. If None, overwrites input file.
        weighted_returns_file: Path to weighted log returns CSV file. If None, uses
            default WEIGHTED_LOG_RETURNS_FILE from constants.

    Returns:
        DataFrame with log_return and weighted_log_return columns added.

    Raises:
        FileNotFoundError: If input file or weighted returns file doesn't exist.
        ValueError: If required columns are missing.
    """
    input_path = Path(input_file)
    validate_file_exists(input_path, "Input file")

    logger.info("Loading ticker data from %s", input_path)
    df = pd.read_parquet(input_path)

    df, _ = _compute_all_features(df)
    df = _add_weighted_returns_from_file(df, weighted_returns_file)

    output_path = input_path if output_file is None else Path(output_file)
    save_parquet_and_csv(df, output_path)

    logger.info("Saved ticker data with log returns to %s", output_path)
    logger.info("Output: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Columns: %s", df.columns.tolist())

    return df
