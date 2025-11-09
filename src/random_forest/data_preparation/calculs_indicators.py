"""Technical indicators calculation for Random Forest data preparation."""

from __future__ import annotations

from typing import cast

import pandas as pd

from src.constants import (
    RF_TECHNICAL_BB_STD,
    RF_TECHNICAL_BB_WINDOW,
    RF_TECHNICAL_EMA_SPAN,
    RF_TECHNICAL_MACD_FAST,
    RF_TECHNICAL_MACD_SIGNAL,
    RF_TECHNICAL_MACD_SLOW,
    RF_TECHNICAL_ROC_PERIOD,
    RF_TECHNICAL_SMA_WINDOW,
)
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices.
        period: Period for RSI calculation (default: 14).

    Returns:
        Series with RSI values.
    """
    delta = prices.diff()
    gain = delta.where(delta > 0.0, 0.0)  # type: ignore[operator]
    loss = -delta.where(delta < 0.0, 0.0)  # type: ignore[operator]

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = cast(pd.Series, 100 - (100 / (1 + rs)))

    return rsi


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return cast(pd.Series, prices.rolling(window=window, min_periods=window).mean())


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return cast(pd.Series, prices.ewm(span=span, adjust=False).mean())


def calculate_macd(
    prices: pd.Series,
    *,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = calculate_ema(prices, span=fast)
    ema_slow = calculate_ema(prices, span=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema(macd_line, span=signal)
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    )


def calculate_bollinger_bands(
    prices: pd.Series,
    *,
    window: int,
    num_std: float,
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle = cast(pd.Series, prices.rolling(window=window, min_periods=window).mean())
    rolling_std = prices.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    bandwidth = (upper - lower) / middle.replace(0.0, pd.NA)

    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
        }
    )


def calculate_rate_of_change(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Rate of Change (ROC) indicator."""
    previous = cast(pd.Series, prices.shift(period))
    roc = (prices - previous) / previous.replace(0.0, pd.NA)
    return cast(pd.Series, roc)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to DataFrame."""
    df = df.copy()
    close = cast(pd.Series, df["weighted_closing"])

    df["rsi_14"] = calculate_rsi(close, period=14)
    df["sma_20"] = calculate_sma(close, window=RF_TECHNICAL_SMA_WINDOW)
    df["ema_20"] = calculate_ema(close, span=RF_TECHNICAL_EMA_SPAN)

    macd_df = calculate_macd(
        close,
        fast=RF_TECHNICAL_MACD_FAST,
        slow=RF_TECHNICAL_MACD_SLOW,
        signal=RF_TECHNICAL_MACD_SIGNAL,
    )
    df = df.join(macd_df)

    bb_df = calculate_bollinger_bands(
        close,
        window=RF_TECHNICAL_BB_WINDOW,
        num_std=RF_TECHNICAL_BB_STD,
    )
    df = df.join(bb_df)

    df["roc_10"] = calculate_rate_of_change(close, period=RF_TECHNICAL_ROC_PERIOD)

    logger.info("Technical indicators added successfully")
    return df
