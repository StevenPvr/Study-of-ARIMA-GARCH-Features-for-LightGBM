"""Data conversion functions for the S&P 500 Forecasting project.

Adds no-look-ahead aggregation support via time-varying liquidity weights
computed from trailing windows that exclude the current observation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.constants as C
from src.constants import LIQUIDITY_WEIGHTS_WINDOW_DEFAULT
from src.data_conversion.utils import (
    _compute_daily_weight_totals,
    _merge_weights_with_returns,
    _normalize_weights,
    _validate_columns,
    _validate_weight_sum,
    validate_dataframe_not_empty,
)
from src.path import DATASET_FILTERED_FILE, DATASET_FILTERED_PARQUET_FILE, WEIGHTED_LOG_RETURNS_FILE
from src.utils import (
    compute_log_returns as utils_compute_log_returns,
    compute_rolling_liquidity_weights,
    extract_date_range,
    get_logger,
    save_parquet_and_csv,
)

logger = get_logger(__name__)


def _load_dataset_file(input_path: Path) -> pd.DataFrame:
    """Load dataset from file based on explicit extension.

    Tests expect explicit, non-fallback behavior with clear errors. We do not
    silently fall back across formats here to avoid ambiguous I/O paths.

    Args:
        input_path: Path to dataset file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If file extension is unsupported or loaded data is empty.
    """
    logger.info("Loading dataset from: %s", input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        # Keep message stable for tests and avoid implicit fallbacks
        raise ValueError("Unsupported file extension")

    if df.empty:
        # Standardize error across formats for easier handling in callers/tests
        raise ValueError("Loaded dataset is empty")
    return df


def _standardize_column_names(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to internal convention.

    Args:
        raw_df: DataFrame with potentially non-standard column names.

    Returns:
        DataFrame with standardized column names.
    """
    raw_df = raw_df.rename(
        columns={
            "data": "date",
            "tickers": "ticker",
            "close": "closing",
        }
    )
    _validate_columns(
        raw_df,
        {"date", "ticker", "closing", "volume"},
        "loaded dataset",
    )
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    return raw_df


def load_filtered_dataset(input_file: str | Path) -> pd.DataFrame:
    """Load and prepare filtered dataset.

    Loads from Parquet file if available (faster), otherwise falls back to CSV.

    Args:
        input_file: Path to filtered dataset (CSV or Parquet).

    Returns:
        DataFrame with date parsed and sorted by ticker and date.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If DataFrame is empty after loading.
    """
    logger.info("Loading filtered dataset")
    input_path = Path(input_file)
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)

    raw_df = _load_dataset_file(input_path)

    if raw_df.empty:
        msg = "Loaded dataset is empty"
        raise ValueError(msg)

    raw_df = _standardize_column_names(raw_df)
    return raw_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_liquidity_weights(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute static liquidity weights from full sample.

    Uses average volume * average closing price per ticker over the entire
    sample period. Convenient for descriptive analysis but may introduce
    look-ahead bias when used for modeling.

    Args:
        raw_df: DataFrame with ticker, volume, closing columns.

    Returns:
        DataFrame with weight column indexed by ticker.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if raw_df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    required_columns = {"ticker", "volume", "closing"}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)

    logger.info("Computing static liquidity weights")

    # Filter valid liquidity data (positive volume and price)
    df_valid = raw_df.dropna(subset=["volume", "closing"]).copy()
    mask = (df_valid["volume"] > 0) & (df_valid["closing"] > 0)
    df_valid = df_valid.loc[mask].copy()

    # Compute liquidity scores (volume * closing price)
    liquidity_scores = (
        df_valid.groupby("ticker")["volume"].mean() * df_valid.groupby("ticker")["closing"].mean()
    )
    liquidity_metrics = liquidity_scores.to_frame(name="liquidity_score")

    # Normalize to weights
    total_liquidity = liquidity_metrics["liquidity_score"].sum()
    if total_liquidity <= 0:
        msg = "Sum of liquidity scores is zero or negative"
        raise ValueError(msg)

    liquidity_metrics["weight"] = liquidity_metrics["liquidity_score"] / total_liquidity
    return liquidity_metrics


def save_liquidity_weights(liquidity_metrics: pd.DataFrame, output_file: str | Path) -> None:
    """Save liquidity weights to disk.

    Args:
        liquidity_metrics: DataFrame with weights to save.
        output_file: Path where to save the weights.

    Raises:
        KeyError: If required columns are missing.
    """
    required_columns = {"liquidity_score", "weight"}
    missing_columns = required_columns - set(liquidity_metrics.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)

    output_path = Path(output_file)
    from src.utils.io import ensure_output_dir

    ensure_output_dir(output_path)
    liquidity_metrics.to_csv(output_path)
    logger.info("Saved liquidity weights to %s", output_path)


def compute_log_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns per ticker.

    Delegates to src.utils.compute_log_returns() for consistency.
    Uses 'closing' column name for backward compatibility with this module's convention.

    Args:
        raw_df: DataFrame with ticker and closing price columns.

    Returns:
        DataFrame with log_return column, rows with NaN removed.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if raw_df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    required_columns = {"ticker", "closing"}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)

    logger.info("Computing log returns per ticker")
    raw_df = raw_df.copy()
    # Guard against zero/negative prices (from cleaning completion); exclude from return calc
    raw_df.loc[raw_df["closing"] <= 0, "closing"] = pd.NA

    # Delegate to src.utils.compute_log_returns with 'closing' as price column
    return utils_compute_log_returns(
        raw_df,
        price_col="closing",
        group_by="ticker",
        output_col="log_return",
        remove_first=True,
    )


def _compute_weighted_log_returns_per_ticker(weighted_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted log returns per ticker-date.

    Args:
        weighted_returns: DataFrame with log_return, weight, and weight_sum columns.

    Returns:
        DataFrame with added weighted_log_return column (log_return * weight / weight_sum).
    """
    weighted_returns = weighted_returns.copy()
    weighted_returns["weighted_log_return"] = (
        weighted_returns["log_return"] * weighted_returns["weight"] / weighted_returns["weight_sum"]
    )
    return weighted_returns


def _aggregate_weighted_returns(weighted_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted log returns by date.

    Args:
        weighted_returns: DataFrame with weighted_log_return and date columns.

    Returns:
        DataFrame with date and weighted_log_return columns (sum of weighted log returns per date).
    """
    aggregated_df = weighted_returns.groupby("date", as_index=False).agg(
        weighted_log_return=("weighted_log_return", "sum")
    )
    return pd.DataFrame(aggregated_df)


def _create_empty_aggregated_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create empty aggregated returns and daily totals DataFrames.

    Returns:
        Tuple of empty aggregated DataFrame and empty daily totals DataFrame.
    """
    empty_aggregated = pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "weighted_log_return": pd.Series(dtype="float64"),
        }
    )
    empty_daily = pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "weight_sum": pd.Series(dtype="float64"),
        }
    )
    return empty_aggregated, empty_daily


def compute_weighted_aggregated_returns(
    returns_df: pd.DataFrame, liquidity_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weighted aggregated log returns by date.

    Args:
        returns_df: DataFrame with log returns per ticker and date.
        liquidity_metrics: DataFrame with weights per ticker or (date, ticker).

    Returns:
        Tuple of (aggregated DataFrame with weighted_log_return,
        daily_weight_totals DataFrame).

    Raises:
        ValueError: If required columns are missing, DataFrames are empty,
            or sum of weights is zero or negative for any date.
    """
    validate_dataframe_not_empty(returns_df, "Returns")
    validate_dataframe_not_empty(liquidity_metrics, "Liquidity metrics")
    _validate_columns(returns_df, {"ticker", "date", "log_return"}, "returns_df")
    _validate_columns(liquidity_metrics, {"weight"}, "liquidity_metrics")

    logger.info("Computing weighted log returns")
    weighted_returns = _merge_weights_with_returns(returns_df, liquidity_metrics)
    daily_weight_totals_df = _compute_daily_weight_totals(weighted_returns)
    _validate_weight_sum(daily_weight_totals_df)
    weighted_returns = weighted_returns.merge(daily_weight_totals_df, on="date", how="left")

    # Ignore dates with zero or negative total weights (tickers not in index)
    weighted_returns = weighted_returns[weighted_returns["weight_sum"] > 0].copy()
    if weighted_returns.empty:
        empty_aggregated, _ = _create_empty_aggregated_returns()
        filtered_daily = daily_weight_totals_df.loc[
            daily_weight_totals_df["weight_sum"] > 0
        ].reset_index(drop=True)
        return empty_aggregated, filtered_daily

    # Compute weighted log returns directly: log_return * weight / weight_sum
    weighted_returns = _compute_weighted_log_returns_per_ticker(pd.DataFrame(weighted_returns))
    aggregated = _aggregate_weighted_returns(weighted_returns)
    return aggregated, daily_weight_totals_df


def _compute_weighted_avg(group: pd.DataFrame, price_col: str) -> float:
    """Compute weighted average for a group.

    Args:
        group: DataFrame group with price and normalized_weight columns.
        price_col: Name of price column to average.

    Returns:
        Weighted average price.
    """
    return float(np.average(group[price_col], weights=group["normalized_weight"]))


def _aggregate_weighted_prices_by_date(weighted_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted prices by date.

    Args:
        weighted_df: DataFrame with normalized_weight, closing, and date columns.

    Returns:
        DataFrame with date and weighted_closing columns.
    """

    def compute_weighted_averages(group: pd.DataFrame) -> pd.Series:
        """Compute weighted averages for a date group."""
        return pd.Series(
            {"weighted_closing": (group["closing"] * group["normalized_weight"]).sum()}
        )

    result = (
        weighted_df.groupby("date")
        .apply(compute_weighted_averages, include_groups=False)  # type: ignore[arg-type]
        .reset_index()
    )
    return result


def compute_weighted_prices(
    returns_df: pd.DataFrame,
    liquidity_metrics: pd.DataFrame,
    daily_weight_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted prices (open/close) for backtesting.

    Args:
        returns_df: DataFrame with log returns per ticker and date.
        liquidity_metrics: DataFrame with weights per ticker or (date, ticker).
        daily_weight_totals: DataFrame with daily weight sums (date, weight_sum).

    Returns:
        DataFrame with weighted_closing by date.

    Raises:
        ValueError: If required columns are missing or DataFrames are empty.
    """
    validate_dataframe_not_empty(returns_df, "Returns")
    validate_dataframe_not_empty(liquidity_metrics, "Liquidity metrics")
    validate_dataframe_not_empty(daily_weight_totals, "Daily weight totals")
    _validate_columns(returns_df, {"ticker", "date", "closing"}, "returns_df")
    _validate_columns(liquidity_metrics, {"weight"}, "liquidity_metrics")
    _validate_columns(daily_weight_totals, {"date", "weight_sum"}, "daily_weight_totals")
    _validate_weight_sum(daily_weight_totals)

    logger.info("Computing weighted prices (for backtesting)")
    raw_weighted = _merge_weights_with_returns(returns_df, liquidity_metrics)
    raw_weighted = raw_weighted.merge(daily_weight_totals, on="date", how="left")
    # Ignore dates with zero or negative total weights
    raw_weighted = raw_weighted[raw_weighted["weight_sum"] > 0].copy()
    if raw_weighted.empty:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "weighted_closing": pd.Series(dtype="float64"),
            }
        )
    raw_weighted = _normalize_weights(pd.DataFrame(raw_weighted))
    return _aggregate_weighted_prices_by_date(raw_weighted)


def save_weighted_returns(aggregated: pd.DataFrame, output_file: str | Path) -> None:
    """Save weighted log returns to Parquet and CSV files.

    Args:
        aggregated: DataFrame with weighted log returns and prices.
        output_file: Path to save the file (Parquet and CSV will be created).

    Raises:
        ValueError: If aggregated DataFrame is empty or missing required columns.
    """
    if aggregated.empty:
        msg = "Aggregated DataFrame is empty"
        raise ValueError(msg)

    if "date" not in aggregated.columns:
        msg = "Missing 'date' column in aggregated DataFrame"
        raise ValueError(msg)

    output_path = Path(output_file)
    # If output_file is CSV, convert to Parquet path (save_parquet_and_csv will create both)
    if output_path.suffix.lower() == ".csv":
        output_path = output_path.with_suffix(".parquet")

    logger.info("Saving weighted log returns to %s", output_path)
    save_parquet_and_csv(aggregated, output_path)

    start_date, end_date = extract_date_range(aggregated, date_col="date", as_string=True)
    logger.info(
        "Conversion complete: %d dates, period %s → %s",
        len(aggregated),
        start_date,
        end_date,
    )


def _run_weighted_conversion_pipeline(
    raw_df: pd.DataFrame,
    liquidity_metrics: pd.DataFrame,
    returns_output_file: str | Path,
) -> None:
    """Run the common pipeline to compute and save weighted returns and prices.

    This helper centralizes the orchestration shared by both static-weight
    and no-look-ahead time-varying pipelines to respect DRY and keep the
    high-level functions focused on their specific weighting scheme.

    Args:
        raw_df: Loaded filtered dataset with at least [date, ticker, open,
            closing] columns.
        liquidity_metrics: Liquidity weights, either static per ticker or
            time-varying per (date, ticker). Must contain a ``weight`` column.
        returns_output_file: Path to the CSV file where the aggregated
            weighted returns and prices will be saved.
    """
    validate_dataframe_not_empty(raw_df, "Input")
    validate_dataframe_not_empty(liquidity_metrics, "Liquidity metrics")

    returns_df = compute_log_returns(raw_df)
    aggregated, daily_weight_totals = compute_weighted_aggregated_returns(
        returns_df, liquidity_metrics
    )
    raw_aggregated = compute_weighted_prices(returns_df, liquidity_metrics, daily_weight_totals)
    aggregated = aggregated.merge(raw_aggregated, on="date")
    save_weighted_returns(aggregated, returns_output_file)


def compute_weighted_log_returns(
    input_file: str | Path | None = None,
    returns_output_file: str | Path | None = None,
) -> None:
    """Compute liquidity-weighted log returns from filtered dataset (static).

    This version uses *static* liquidity weights based on average
    ``volume * price`` per ticker over the full sample. It is convenient
    for descriptive analysis but may introduce look-ahead if used directly
    to generate features for forecasting models.

    Args:
        input_file: Path to filtered dataset CSV. If None, uses default.
        returns_output_file: Path to save weighted log returns CSV. If None,
            uses default path.
    """
    logger.warning(
        "Static weights mode: this may introduce look-ahead if used to produce "
        "training data. Prefer compute_weighted_log_returns_no_lookahead() "
        "for any modeling pipeline."
    )

    if input_file is None:
        # Respect monkeypatched defaults in this module first.
        # If neither was overridden, default to Parquet for performance.
        if DATASET_FILTERED_PARQUET_FILE != C.DATASET_FILTERED_PARQUET_FILE:
            input_file = DATASET_FILTERED_PARQUET_FILE
        elif DATASET_FILTERED_FILE != C.DATASET_FILTERED_FILE:
            input_file = DATASET_FILTERED_FILE
        else:
            input_file = DATASET_FILTERED_PARQUET_FILE
    if returns_output_file is None:
        returns_output_file = WEIGHTED_LOG_RETURNS_FILE

    raw_df = load_filtered_dataset(input_file)
    liquidity_metrics = compute_liquidity_weights(raw_df)

    _run_weighted_conversion_pipeline(
        raw_df=raw_df,
        liquidity_metrics=liquidity_metrics,
        returns_output_file=returns_output_file,
    )


def compute_weighted_log_returns_no_lookahead(
    input_file: str | Path | None = None,
    returns_output_file: str | Path | None = None,
    window: int = LIQUIDITY_WEIGHTS_WINDOW_DEFAULT,
    min_periods: int | None = None,
) -> None:
    """Compute liquidity-weighted log returns without look-ahead.

    Uses trailing-window liquidity scores per (date, ticker), shifted by one
    day. Aggregation normalizes these scores by date and computes weighted
    log returns and prices that respect causal ordering.

    Args:
        input_file: Path to filtered dataset CSV. Defaults to constant.
        returns_output_file: Path to save weighted log returns CSV. Defaults
            to constant.
        window: Trailing window for liquidity scores (in days).
        min_periods: Minimum observations required in the rolling window.
    """
    if input_file is None:
        # Respect monkeypatched defaults in this module first.
        # If neither was overridden, default to Parquet for performance.
        if DATASET_FILTERED_PARQUET_FILE != C.DATASET_FILTERED_PARQUET_FILE:
            input_file = DATASET_FILTERED_PARQUET_FILE
        elif DATASET_FILTERED_FILE != C.DATASET_FILTERED_FILE:
            input_file = DATASET_FILTERED_FILE
        else:
            input_file = DATASET_FILTERED_PARQUET_FILE
    if returns_output_file is None:
        returns_output_file = WEIGHTED_LOG_RETURNS_FILE

    raw_df = load_filtered_dataset(input_file)
    # Time-varying (date, ticker) liquidity scores using only past info
    tv_weights = compute_rolling_liquidity_weights(
        raw_df, window=window, min_periods=min_periods, price_col="closing"
    )

    _run_weighted_conversion_pipeline(
        raw_df=raw_df,
        liquidity_metrics=tv_weights,
        returns_output_file=returns_output_file,
    )
