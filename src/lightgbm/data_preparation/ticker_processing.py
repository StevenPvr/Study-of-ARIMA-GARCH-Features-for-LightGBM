"""Ticker file processing functions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from src.constants import (
    DATA_TICKERS_FULL_FILE,
    DATA_TICKERS_FULL_INDICATORS_FILE,
    DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE,
    LIGHTGBM_LAG_WINDOWS,
    LIGHTGBM_TURNOVER_MA_WINDOW,
    LIGHTGBM_VOL_MA_LONG_WINDOW,
    LIGHTGBM_VOL_MA_SHORT_WINDOW,
)
from src.lightgbm.data_preparation.data_loading import ensure_ticker_id_column
from src.lightgbm.data_preparation.features import (
    add_custom_ml_indicators_per_ticker,
    add_lag_features,
)
from src.utils import get_logger, save_parquet_and_csv

logger = get_logger(__name__)


def _remove_warmup_period(df: pd.DataFrame, warmup_window: int) -> pd.DataFrame:
    """Remove warm-up period for all tickers.

    Args:
        df: DataFrame with indicators.
        warmup_window: Number of days to remove from start of each ticker.

    Returns:
        DataFrame with warm-up period removed.
    """
    df_sorted = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    result_dfs: list[pd.DataFrame] = []
    for ticker in df_sorted["ticker"].unique():
        ticker_df = df_sorted[df_sorted["ticker"] == ticker].copy()

        # Forward fill volume for missing values
        if "volume" in ticker_df.columns:
            volume_series = cast(pd.Series, ticker_df["volume"])
            ticker_df["volume"] = volume_series.ffill()

        # Skip first warmup_window rows
        ticker_df_trimmed = ticker_df.iloc[warmup_window:].copy()
        result_dfs.append(ticker_df_trimmed)

    df_clean = pd.concat(result_dfs, ignore_index=True)

    # Ensure all tickers start at the same date
    first_valid_date = cast(pd.Timestamp, df_clean.groupby("ticker")["date"].min().max())
    df_clean = cast(pd.DataFrame, df_clean[df_clean["date"] >= first_valid_date].copy())

    return df_clean


def _compute_log_returns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns if missing.

    Args:
        df: DataFrame that may be missing log_return.

    Returns:
        DataFrame with log_return computed if needed.
    """
    if "close" in df.columns and "log_return" not in df.columns:
        logger.info("Computing log returns per ticker")
        from src.data_preparation.computations import compute_log_returns_for_tickers

        df = compute_log_returns_for_tickers(cast(pd.DataFrame, df))
    return df


def _drop_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw OHLC price columns if present."""
    columns_to_drop = [
        col for col in ["open", "high", "low", "close", "adj_close"] if col in df.columns
    ]
    if columns_to_drop:
        logger.info(f"Dropping price columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    return df


def _drop_na_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in technical indicators.

    Args:
        df: DataFrame with potential NaN indicators.

    Returns:
        DataFrame without NaN indicators.
    """
    indicator_cols = [
        "sma_20",
        "ema_20",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "roc_10",
    ]
    existing_indicator_cols = [col for col in indicator_cols if col in df.columns]

    if existing_indicator_cols:
        for col in existing_indicator_cols:
            col_series = cast(pd.Series, df[col])
            df = cast(pd.DataFrame, df[col_series.notna()])
        df = cast(pd.DataFrame, df).reset_index(drop=True)

    return df


def _add_lag_features_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features per ticker based on LIGHTGBM_LAG_WINDOWS.

    Args:
        df: DataFrame with indicators.

    Returns:
        DataFrame with lag features added.
    """
    df_sorted = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    lag_feature_columns = [
        "log_return",
        "abs_ret",
        "ret_sq",
        "log_volatility",
        "log_volume",
        "log_volume_rel_ma_5",
        "log_volume_zscore_20",
        "log_turnover",
        "turnover_rel_ma_5",
        "obv",
        "atr",
        "sigma2_garch",
        "sigma_garch",
        "std_resid_garch",
        "arima_residual_return",
        "arima_pred_return",
    ]

    feature_columns = [col for col in lag_feature_columns if col in df_sorted.columns]

    if not feature_columns:
        logger.warning("No lag feature columns found in dataframe")
        return df_sorted

    df_with_lags = add_lag_features(
        df_sorted,
        feature_columns=feature_columns,
        lag_windows=LIGHTGBM_LAG_WINDOWS,
    )

    return df_with_lags


def process_ticker_file(input_path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    """Process a single ticker file and save processed datasets.

    Args:
        input_path: Path to the input parquet file for a single ticker.
        output_dir: Directory to save output files. If None, uses data directory.

    Returns:
        Processed DataFrame with indicators and lags.
    """
    logger.info(f"Processing ticker file: {input_path}")
    df = pd.read_parquet(input_path)

    n_rows_before = len(df)

    # Normalize ticker_id if missing
    df = ensure_ticker_id_column(df)

    # Add custom ML indicators per ticker
    df_with_indicators = add_custom_ml_indicators_per_ticker(df)

    # Remove warm-up period
    warmup_window = max(
        int(LIGHTGBM_VOL_MA_SHORT_WINDOW),
        int(LIGHTGBM_VOL_MA_LONG_WINDOW),
        int(LIGHTGBM_TURNOVER_MA_WINDOW),
        int(max(LIGHTGBM_LAG_WINDOWS)),
    )
    logger.info("Removing unified warm-up period of %d days for each ticker", warmup_window)
    df_clean = _remove_warmup_period(df_with_indicators, warmup_window)

    # Compute log returns if needed
    df_clean = _compute_log_returns_if_needed(df_clean)

    # Drop price columns
    df_clean = _drop_price_columns(df_clean)

    # Legacy technical indicators (SMA/EMA/MACD/ROC) may contain NaN,
    # especially in the warm-up region. We no longer drop rows based on
    # these columns here to avoid excessive data loss. Any remaining NaN
    # will be handled later after column selection or by the downstream
    # model (which can handle missing values).
    n_dropped_na = 0

    n_rows_after = len(df_clean)
    n_removed = n_rows_before - n_rows_after

    logger.info(
        f"Removed {n_removed:,} rows total "
        f"({warmup_window} warmup + {n_dropped_na:,} with NaN indicators)"
    )
    first_valid_date = cast(pd.Timestamp, df_clean.groupby("ticker")["date"].min().max())
    logger.info(f"All tickers now start at {first_valid_date.date()}")

    # Save processed ticker-level dataset with indicators
    if output_dir is None:
        output_dir = DATA_TICKERS_FULL_FILE.parent

    output_path = output_dir / DATA_TICKERS_FULL_INDICATORS_FILE.name
    save_parquet_and_csv(df_clean, output_path)
    logger.info(f"Saved processed ticker-level data with indicators: {output_path}")

    return df_clean


def process_all_tickers(input_file: Path | None = None, output_dir: Path | None = None) -> None:
    """Process all tickers from a single parquet file.

    Args:
        input_file: Path to input parquet file. If None, uses default file.
        output_dir: Directory to save processed datasets. If None, uses data directory.
    """
    if input_file is None:
        input_file = DATA_TICKERS_FULL_FILE

    if output_dir is None:
        output_dir = input_file.parent

    logger.info(f"Loading ticker-level data from {input_file}")

    df_clean = process_ticker_file(input_file, output_dir=output_dir)

    # Save with insights indicators if available
    insights_output_path = output_dir / DATA_TICKERS_FULL_INSIGHTS_INDICATORS_FILE.name
    save_parquet_and_csv(df_clean, insights_output_path)
    logger.info(f"Saved ticker-level data with insights indicators: {insights_output_path}")


# Backward compatibility aliases
add_indicators_to_ticker_parquet = process_ticker_file
process_all_ticker_parquets = process_all_tickers
