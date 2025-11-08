"""Technical indicators calculation for Random Forest data preparation."""

from __future__ import annotations

from typing import cast

import pandas as pd

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


def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        prices: Series of prices.
        period: Period for SMA calculation (default: 20).

    Returns:
        Series with SMA values.
    """
    sma = cast(pd.Series, prices.rolling(window=period, min_periods=period).mean())
    return sma


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Args:
        prices: Series of prices.
        period: Period for EMA calculation (default: 20).

    Returns:
        Series with EMA values.
    """
    ema = cast(pd.Series, prices.ewm(span=period, adjust=False, min_periods=period).mean())
    return ema


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        Tuple of (MACD line, signal line, histogram).
    """
    ema_fast = calculate_ema(prices, period=fast)
    ema_slow = calculate_ema(prices, period=slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, period=signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        prices: Series of closing prices.
        period: Period for moving average (default: 20).
        num_std: Number of standard deviations (default: 2.0).

    Returns:
        Tuple of (upper band, middle band, lower band).
    """
    sma = calculate_sma(prices, period=period)
    std = prices.rolling(window=period, min_periods=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return upper_band, sma, lower_band


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to DataFrame.

    Calculates RSI, SMA, EMA, and MACD using closing prices.

    Args:
        df: DataFrame with columns including 'weighted_closing'.

    Returns:
        DataFrame with added technical indicator columns.
    """
    df = df.copy()
    close = cast(pd.Series, df["weighted_closing"])

    # RSI (14 days)
    df["rsi_14"] = calculate_rsi(close, period=14)

    # Simple Moving Average (20 days)
    df["sma_20"] = calculate_sma(close, period=20)

    # Exponential Moving Average (20 days)
    df["ema_20"] = calculate_ema(close, period=20)

    # MACD (12, 26, 9)
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram

    logger.info("Technical indicators added successfully")
    return df
