"""Computation functions for data preparation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.constants import (
    LIGHTGBM_REALIZED_VOL_WINDOW,
    REQUIRED_COLS_CLOSE_PRICE,
    REQUIRED_COLS_LOG_RETURN,
)
from src.utils import get_logger, validate_required_columns

logger = get_logger(__name__)


def compute_log_returns_for_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns per ticker from close prices.

    Args:
        df: DataFrame with ticker, date, and close columns.

    Returns:
        DataFrame with log_return column added, rows with NaN removed.

    Raises:
        ValueError: If required columns are missing.
    """
    validate_required_columns(df, REQUIRED_COLS_CLOSE_PRICE)

    # Compute log returns per ticker
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Return at date t uses close price at t and t-1 (no future info)
    def _safe_log_return(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        prev = x.shift(1)
        ratio = x / prev
        # Clean problematic ratios before log
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio = ratio.mask(ratio <= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return cast(pd.Series, np.log(ratio))

    df["log_return"] = df.groupby("ticker")["close"].transform(_safe_log_return)

    # Remove first observation per ticker (NaN log_return)
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    logger.info("Computed log_return per ticker")
    return df


def compute_volatility_for_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log volatility per ticker from log returns.

    Calculates log_volatility as log1p of sqrt of LIGHTGBM_REALIZED_VOL_WINDOW-day
    sum of squared returns. This column is the RF target.

    Args:
        df: DataFrame with ticker, date, and log_return columns.
            Must be sorted by (ticker, date).

    Returns:
        DataFrame with log_volatility column added.

    Raises:
        ValueError: If required columns are missing.
    """
    validate_required_columns(df, REQUIRED_COLS_LOG_RETURN)

    # Ensure data is sorted by ticker and date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def _compute_volatility_per_group(log_ret: pd.Series) -> pd.Series:
        """Compute volatility for a group of log returns."""
        ret_sq = cast(pd.Series, log_ret**2)

        # Trailing rolling window (no look-ahead)
        realized_vol = cast(
            pd.Series,
            np.sqrt(
                ret_sq.rolling(
                    LIGHTGBM_REALIZED_VOL_WINDOW,
                    min_periods=LIGHTGBM_REALIZED_VOL_WINDOW,
                ).sum(),
            ),
        )
        return cast(pd.Series, np.log1p(realized_vol.astype(float)))

    # Apply per ticker without accessing grouping columns
    df = df.assign(
        log_volatility=df.groupby("ticker")["log_return"].transform(_compute_volatility_per_group)
    )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info("Computed log_volatility per ticker")
    return df


def compute_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_volume from volume column if present.

    Args:
        df: DataFrame with volume column.

    Returns:
        DataFrame with log_volume column added if volume exists.
    """
    if "volume" in df.columns and "log_volume" not in df.columns:
        logger.info("Computing log_volume from volume using log1p")
        df["log_volume"] = np.log1p(df["volume"])
    return df
