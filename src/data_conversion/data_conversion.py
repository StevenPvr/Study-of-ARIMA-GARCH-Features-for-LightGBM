"""Data conversion functions for the S&P 500 Forecasting project.

Adds no-look-ahead aggregation support via time-varying liquidity weights
computed from trailing windows that exclude the current observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.constants import (
    DATASET_FILTERED_FILE,
    LIQUIDITY_WEIGHTS_FILE,
    LIQUIDITY_WEIGHTS_WINDOW_DEFAULT,
    WEIGHTED_LOG_RETURNS_FILE,
)
from src.utils import get_logger

from src.data_conversion.utils import (
    _validate_columns,
    _validate_dataframe_not_empty,
    _validate_weight_sum,
)

logger = get_logger(__name__)


def compute_liquidity_weights_timevarying(
    raw_df: pd.DataFrame,
    window: int = LIQUIDITY_WEIGHTS_WINDOW_DEFAULT,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Compute time-varying liquidity weights without look-ahead.

    Uses trailing rolling mean of ``volume * closing`` per ticker, shifted by 1 day
    to avoid using same-day information. Weights are returned as unnormalized
    liquidity scores for each (date, ticker); normalization is done at aggregation.

    Args:
        raw_df: DataFrame with columns [date, ticker, volume, closing].
        window: Trailing window size in days for rolling mean.
        min_periods: Minimum observations in window. Defaults to ``window`` if None.

    Returns:
        DataFrame with columns [date, ticker, weight] representing trailing liquidity.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    _validate_dataframe_not_empty(raw_df, "Input")
    _validate_columns(raw_df, {"date", "ticker", "volume", "closing"}, "raw_df")

    logger.info(f"Computing time-varying liquidity scores (window={window}, no look-ahead)")
    df = raw_df.sort_values(["ticker", "date"]).copy()
    df["liquidity"] = df["volume"] * df["closing"]
    mp = window if min_periods is None else min_periods
    df["weight"] = df.groupby("ticker")["liquidity"].transform(
        lambda s: s.rolling(window, min_periods=mp).mean()
    )
    # Shift by 1 to ensure no look-ahead (use only past info)
    df["weight"] = df.groupby("ticker")["weight"].shift(1)
    weights = (
        df.dropna(subset=["weight"])[["date", "ticker", "weight"]].reset_index(drop=True).copy()
    )
    return pd.DataFrame(weights)


def load_filtered_dataset(input_file: str | Path) -> pd.DataFrame:
    """Load and prepare filtered dataset.

    Args:
        input_file: Path to filtered dataset CSV.

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

    raw_df = pd.read_csv(input_path)
    if raw_df.empty:
        msg = "Loaded dataset is empty"
        raise ValueError(msg)

    raw_df["date"] = pd.to_datetime(raw_df["date"])
    return raw_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_liquidity_weights(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity weights per ticker.

    Calculates mean volume and price per ticker, then computes liquidity
    score (volume * price) and normalizes weights.

    Args:
        raw_df: DataFrame with ticker, volume, and closing price columns.

    Returns:
        DataFrame with liquidity metrics and normalized weights per ticker.

    Raises:
        ValueError: If required columns are missing, DataFrame is empty,
            or sum of liquidity scores is zero.
    """
    if raw_df.empty:
        msg = "Input DataFrame is empty"
        raise ValueError(msg)

    required_columns = {"ticker", "volume", "closing"}
    missing_columns = required_columns - set(raw_df.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    logger.info("Computing liquidity metrics per ticker")
    grouped = raw_df.groupby("ticker")[["volume", "closing"]].mean()
    # Convert to DataFrame explicitly to satisfy type checker
    liquidity_metrics = pd.DataFrame(grouped).rename(
        columns={"volume": "mean_volume", "closing": "mean_price"}
    )
    liquidity_metrics["liquidity_score"] = (
        liquidity_metrics["mean_volume"] * liquidity_metrics["mean_price"]
    )

    total_liquidity = liquidity_metrics["liquidity_score"].sum()
    if total_liquidity == 0:
        msg = "Sum of liquidity scores is zero: check data."
        raise ValueError(msg)

    liquidity_metrics["weight"] = liquidity_metrics["liquidity_score"] / total_liquidity
    return liquidity_metrics


def save_liquidity_weights(liquidity_metrics: pd.DataFrame, output_file: str | Path) -> None:
    """Save liquidity weights to CSV file.

    Args:
        liquidity_metrics: DataFrame with liquidity metrics and weights.
        output_file: Path to save the CSV file.

    Raises:
        ValueError: If required columns are missing in liquidity_metrics.
    """
    required_columns = {"mean_volume", "mean_price", "liquidity_score", "weight"}
    missing_columns = required_columns - set(liquidity_metrics.columns)
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise ValueError(msg)

    output_path = Path(output_file)
    logger.info(f"Saving liquidity weights to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    liquidity_metrics[["mean_volume", "mean_price", "liquidity_score", "weight"]].to_csv(
        output_path, index=True
    )


def compute_log_returns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns per ticker.

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
        raise ValueError(msg)

    logger.info("Computing log returns per ticker")
    raw_df["log_return"] = raw_df.groupby("ticker")["closing"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    return raw_df.dropna(subset=["log_return"])


def compute_weighted_aggregated_returns(
    returns_df: pd.DataFrame, liquidity_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute weighted aggregated log returns by date.

    Args:
        returns_df: DataFrame with log returns per ticker and date.
        liquidity_metrics: DataFrame with weights per ticker.

    Returns:
        Tuple of (aggregated DataFrame with weighted_log_return, daily_weight_totals DataFrame).

    Raises:
        ValueError: If required columns are missing, DataFrames are empty,
            or sum of weights is zero or negative for any date.
    """
    _validate_dataframe_not_empty(returns_df, "Returns")
    _validate_dataframe_not_empty(liquidity_metrics, "Liquidity metrics")
    _validate_columns(returns_df, {"ticker", "date", "log_return"}, "returns_df")
    _validate_columns(liquidity_metrics, {"weight"}, "liquidity_metrics")

    logger.info("Computing weighted log returns")
    # Support both static weights (index=ticker) and time-varying weights (columns contain 'date')
    if "date" in liquidity_metrics.columns and "ticker" in liquidity_metrics.columns:
        weighted_returns = returns_df.merge(
            liquidity_metrics[["date", "ticker", "weight"]], on=["date", "ticker"], how="left"
        )
        # Drop rows lacking weights (insufficient history at period start)
        weighted_returns = weighted_returns.dropna(subset=["weight"]).copy()
    else:
        weighted_returns = returns_df.merge(
            liquidity_metrics[["weight"]], left_on="ticker", right_index=True
        )
    daily_weight_sum = weighted_returns.groupby("date")["weight"].sum()
    daily_weight_totals_df = daily_weight_sum.to_frame(name="weight_sum").reset_index()
    weighted_returns = weighted_returns.merge(daily_weight_totals_df, on="date", how="left")

    _validate_weight_sum(weighted_returns)

    weighted_returns["normalized_weight"] = (
        weighted_returns["weight"] / weighted_returns["weight_sum"]
    )
    weighted_returns["weighted_contribution"] = (
        weighted_returns["log_return"] * weighted_returns["normalized_weight"]
    )
    aggregated_df = weighted_returns.groupby("date", as_index=False).agg(
        weighted_log_return=("weighted_contribution", "sum")
    )
    aggregated = pd.DataFrame(aggregated_df)
    return aggregated, daily_weight_totals_df


def compute_weighted_prices(
    returns_df: pd.DataFrame,
    liquidity_metrics: pd.DataFrame,
    daily_weight_totals: pd.DataFrame,
) -> pd.DataFrame:
    """Compute weighted prices (open/close) for backtesting.

    Args:
        returns_df: DataFrame with log returns per ticker and date.
        liquidity_metrics: DataFrame with weights per ticker.
        daily_weight_totals: DataFrame with daily weight sums (date, weight_sum columns).

    Returns:
        DataFrame with weighted_open and weighted_closing by date.

    Raises:
        ValueError: If required columns are missing or DataFrames are empty.
    """
    _validate_dataframe_not_empty(returns_df, "Returns")
    _validate_dataframe_not_empty(liquidity_metrics, "Liquidity metrics")
    _validate_dataframe_not_empty(daily_weight_totals, "Daily weight totals")
    _validate_columns(returns_df, {"ticker", "date", "open", "closing"}, "returns_df")
    _validate_columns(liquidity_metrics, {"weight"}, "liquidity_metrics")
    _validate_columns(daily_weight_totals, {"date", "weight_sum"}, "daily_weight_totals")

    logger.info("Computing weighted prices (for backtesting)")
    if "date" in liquidity_metrics.columns and "ticker" in liquidity_metrics.columns:
        raw_weighted = returns_df.merge(
            liquidity_metrics[["date", "ticker", "weight"]], on=["date", "ticker"], how="left"
        )
        raw_weighted = raw_weighted.dropna(subset=["weight"]).copy()
    else:
        raw_weighted = returns_df.merge(
            liquidity_metrics[["weight"]], left_on="ticker", right_index=True
        )
    raw_weighted = raw_weighted.merge(daily_weight_totals, on="date", how="left")
    raw_weighted["normalized_weight"] = raw_weighted["weight"] / raw_weighted["weight_sum"]

    def _compute_weighted_avg(group: pd.DataFrame, price_col: str) -> float:
        """Compute weighted average for a group.

        Args:
            group: DataFrame group with price and normalized_weight columns.
            price_col: Name of price column to average.

        Returns:
            Weighted average price.
        """
        return float(np.average(group[price_col], weights=group["normalized_weight"]))

    def _aggregate_group(group: pd.DataFrame) -> pd.Series:
        """Aggregate a group to compute weighted prices.

        Args:
            group: DataFrame group with open, closing, and normalized_weight columns.

        Returns:
            Series with weighted_open and weighted_closing.
        """
        return pd.Series(
            {
                "weighted_open": _compute_weighted_avg(group, "open"),
                "weighted_closing": _compute_weighted_avg(group, "closing"),
            }
        )

    raw_aggregated_result = (
        raw_weighted.groupby("date")
        .apply(_aggregate_group, include_groups=False)  # type: ignore[arg-type]
        .reset_index()
    )
    return pd.DataFrame(raw_aggregated_result)


def save_weighted_returns(aggregated: pd.DataFrame, output_file: str | Path) -> None:
    """Save weighted log returns to CSV file.

    Args:
        aggregated: DataFrame with weighted log returns and prices.
        output_file: Path to save the CSV file.

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
    logger.info(f"Saving weighted log returns to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)

    logger.info(
        f"Conversion complete: {len(aggregated)} dates, "
        f"period {aggregated['date'].min().date()} → {aggregated['date'].max().date()}"
    )


def compute_weighted_log_returns(
    input_file: str | Path | None = None,
    weights_output_file: str | Path | None = None,
    returns_output_file: str | Path | None = None,
) -> None:
    """Compute liquidity-weighted log returns from filtered dataset.

    Calculates liquidity weights based on average volume * price per ticker,
    then computes weighted log returns aggregated by date.

    Args:
        input_file: Path to filtered dataset CSV. If None, uses default.
        weights_output_file: Path to save liquidity weights CSV. If None, uses default.
        returns_output_file: Path to save weighted log returns CSV. If None, uses default.
    """
    if input_file is None:
        input_file = DATASET_FILTERED_FILE
    if weights_output_file is None:
        weights_output_file = LIQUIDITY_WEIGHTS_FILE
    if returns_output_file is None:
        returns_output_file = WEIGHTED_LOG_RETURNS_FILE

    raw_df = load_filtered_dataset(input_file)
    liquidity_metrics = compute_liquidity_weights(raw_df)
    save_liquidity_weights(liquidity_metrics, weights_output_file)
    returns_df = compute_log_returns(raw_df)
    aggregated, daily_weight_totals = compute_weighted_aggregated_returns(
        returns_df, liquidity_metrics
    )
    raw_aggregated = compute_weighted_prices(returns_df, liquidity_metrics, daily_weight_totals)
    aggregated = aggregated.merge(raw_aggregated, on="date")
    save_weighted_returns(aggregated, returns_output_file)


def compute_weighted_log_returns_no_lookahead(
    input_file: str | Path | None = None,
    returns_output_file: str | Path | None = None,
    window: int = LIQUIDITY_WEIGHTS_WINDOW_DEFAULT,
    min_periods: Optional[int] = None,
) -> None:
    """Compute liquidity-weighted log returns without look-ahead.

    Uses trailing-window liquidity scores per (date, ticker), shifted by one day.
    Aggregation normalizes weights by date and computes weighted log returns.

    Args:
        input_file: Path to filtered dataset CSV. Defaults to constant.
        returns_output_file: Path to save weighted log returns CSV. Defaults to constant.
        window: Trailing window for liquidity scores (days).
        min_periods: Minimum observations for rolling window.
    """
    if input_file is None:
        input_file = DATASET_FILTERED_FILE
    if returns_output_file is None:
        returns_output_file = WEIGHTED_LOG_RETURNS_FILE

    raw_df = load_filtered_dataset(input_file)
    # Time-varying (date, ticker) liquidity scores using only past info
    tv_weights = compute_liquidity_weights_timevarying(
        raw_df, window=window, min_periods=min_periods
    )
    returns_df = compute_log_returns(raw_df)
    aggregated, daily_weight_totals = compute_weighted_aggregated_returns(returns_df, tv_weights)
    raw_aggregated = compute_weighted_prices(returns_df, tv_weights, daily_weight_totals)
    aggregated = aggregated.merge(raw_aggregated, on="date")
    save_weighted_returns(aggregated, returns_output_file)
