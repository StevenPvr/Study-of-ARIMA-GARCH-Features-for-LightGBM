"""Feature engineering utilities for LightGBM data preparation.

This module consolidates technical indicators and lag feature creation
previously split across multiple files to reduce duplication and keep
the folder simpler (KISS/DRY). All operations avoid look-ahead bias.
"""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np
import pandas as pd

from src.constants import (
    LIGHTGBM_LAG_WINDOWS,
    LIGHTGBM_TURNOVER_MA_WINDOW,
    LIGHTGBM_VOL_MA_LONG_WINDOW,
    LIGHTGBM_VOL_MA_SHORT_WINDOW,
)
from src.lightgbm.data_preparation.validation import (
    validate_data_sorted_by_date,
    validate_lag_value,
)
from src.utils import get_logger

logger = get_logger(__name__)


# =============================
# Technical indicator features
# =============================


def add_volume_features(tdf: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Add log-volume and normalized variants.

    Why: Normalize volume dynamics to help tree-based models leverage scale- and
    regime-aware information without leaking future values.
    """
    tdf["log_volume"] = cast(pd.Series, np.log1p(vol.astype(float)))
    log_vol = cast(pd.Series, tdf["log_volume"])
    log_vol_ma_5 = cast(
        pd.Series,
        log_vol.rolling(
            LIGHTGBM_VOL_MA_SHORT_WINDOW, min_periods=LIGHTGBM_VOL_MA_SHORT_WINDOW
        ).mean(),
    )
    log_vol_ma_20 = cast(
        pd.Series,
        log_vol.rolling(
            LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
        ).mean(),
    )
    log_vol_std_20 = cast(
        pd.Series,
        log_vol.rolling(LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW).std(),
    )
    tdf["log_volume_rel_ma_5"] = log_vol - log_vol_ma_5
    tdf["log_volume_zscore_20"] = (log_vol - log_vol_ma_20) / log_vol_std_20.replace(0.0, pd.NA)
    return tdf


def add_return_features(tdf: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Add log-returns and simple transforms (absolute/squared)."""
    tdf["log_return"] = cast(pd.Series, np.log(close / close.shift(1)))
    tdf["abs_ret"] = cast(pd.Series, tdf["log_return"].abs())
    tdf["ret_sq"] = cast(pd.Series, tdf["log_return"] ** 2)
    return tdf


def add_calendar_features(tdf: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonality features from the ``date`` column."""
    dt = cast(pd.Series, pd.to_datetime(tdf["date"]))
    tdf["day_of_week"] = dt.dt.dayofweek
    tdf["month"] = dt.dt.month
    tdf["is_month_end"] = dt.dt.is_month_end.astype(int)
    tdf["day_in_month_norm"] = dt.dt.day.astype(float) / 31.0
    return tdf


def add_turnover_features(tdf: pd.DataFrame, vol: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Add log-turnover and deviation from a short moving average."""
    turnover = cast(pd.Series, vol.astype(float) * close.astype(float))
    log_turnover = cast(pd.Series, np.log1p(turnover))
    tdf["log_turnover"] = log_turnover
    tdf["turnover_rel_ma_5"] = log_turnover - cast(
        pd.Series,
        log_turnover.rolling(
            LIGHTGBM_TURNOVER_MA_WINDOW, min_periods=LIGHTGBM_TURNOVER_MA_WINDOW
        ).mean(),
    )
    return tdf


def add_obv_feature(tdf: pd.DataFrame, vol: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Add On-Balance Volume (cumulative volume signed by price direction)."""
    price_diff = cast(pd.Series, close.diff())
    direction = cast(pd.Series, np.sign(price_diff)).fillna(0.0)
    obv = cast(pd.Series, (direction * vol.astype(float)).cumsum())
    tdf["obv"] = obv
    return tdf


def add_atr_feature(
    tdf: pd.DataFrame,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Add Average True Range (rolling mean of True Range) to ``tdf``.

    Why: ATR proxies daily range volatility without peeking into the future.
    """
    high_float = cast(pd.Series, high.astype(float))
    low_float = cast(pd.Series, low.astype(float))
    close_float = cast(pd.Series, close.astype(float))

    tr1 = high_float - low_float
    tr2 = cast(pd.Series, (high_float - close_float.shift(1)).abs())
    tr3 = cast(pd.Series, (low_float - close_float.shift(1)).abs())
    true_range = cast(pd.Series, pd.concat([tr1, tr2, tr3], axis=1).max(axis=1))

    atr_period = 14
    atr = cast(
        pd.Series,
        true_range.rolling(window=atr_period, min_periods=atr_period).mean(),
    )
    tdf["atr"] = atr
    return tdf


def _add_volume_features_single_group(df: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Add per-ticker volume features (internal helper)."""
    df["log_volume"] = np.log1p(vol.astype(float))
    log_vol = df["log_volume"]

    log_vol_ma_5 = log_vol.rolling(
        LIGHTGBM_VOL_MA_SHORT_WINDOW, min_periods=LIGHTGBM_VOL_MA_SHORT_WINDOW
    ).mean()
    log_vol_ma_20 = log_vol.rolling(
        LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
    ).mean()
    log_vol_std_20 = log_vol.rolling(
        LIGHTGBM_VOL_MA_LONG_WINDOW, min_periods=LIGHTGBM_VOL_MA_LONG_WINDOW
    ).std()

    df["log_volume_rel_ma_5"] = log_vol - log_vol_ma_5
    df["log_volume_zscore_20"] = (log_vol - log_vol_ma_20) / log_vol_std_20.replace(0.0, pd.NA)
    return df


def _compute_ticker_features_vectorized(df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
    """Compute all ticker features in one grouped pass (performance)."""
    out = df.copy()

    def _compute_all_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        close = cast(pd.Series, group["close"])
        vol = cast(pd.Series, group["volume"])

        group = _add_volume_features_single_group(group, vol)

        group["log_return"] = np.log(close / close.shift(1))
        group["abs_ret"] = group["log_return"].abs()
        group["ret_sq"] = group["log_return"] ** 2

        turnover = vol.astype(float) * close.astype(float)
        log_turnover = np.log1p(turnover)
        group["log_turnover"] = log_turnover
        turnover_ma_5 = log_turnover.rolling(
            LIGHTGBM_TURNOVER_MA_WINDOW, min_periods=LIGHTGBM_TURNOVER_MA_WINDOW
        ).mean()
        group["turnover_rel_ma_5"] = log_turnover - turnover_ma_5

        close_diff = close.diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        group["obv"] = (direction * vol).cumsum()

        if "high" in group.columns and "low" in group.columns:
            high = group["high"]
            low = group["low"]
            high_low = high - low
            high_close = (high - close.shift(1)).abs()
            low_close = (low - close.shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            group["atr"] = tr.rolling(14, min_periods=1).mean()

        return group

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrameGroupBy.apply operated on the grouping columns",
            category=FutureWarning,
        )
        grouped = out.groupby(ticker_col, group_keys=False)
        out = grouped.apply(_compute_all_features, include_groups=True)  # type: ignore

    out = add_calendar_features(out)
    out = out.sort_values([ticker_col, "date"]).reset_index(drop=True)
    out = out.rename(columns={ticker_col: "ticker"})
    return out


def add_custom_ml_indicators_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ML indicators per ticker without leakage (vectorized)."""
    if "ticker" in df.columns:
        ticker_col_name = "ticker"
    elif "tickers" in df.columns:
        ticker_col_name = "tickers"
    else:
        raise ValueError("Neither 'ticker' nor 'tickers' column found in DataFrame")

    required = {"date", ticker_col_name, "close", "volume"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out = out.sort_values([ticker_col_name, "date"]).reset_index(drop=True)
    out = _compute_ticker_features_vectorized(out, ticker_col_name)
    logger.info("Custom ML indicators computed for ticker-level data (optimized)")
    return out


# =============================
# Lag feature creation
# =============================


def add_single_lag_feature(df: pd.DataFrame, column: str, lag: int) -> pd.DataFrame:
    """Add one lag feature to ``df`` (per-ticker if available)."""
    lag_column_name = f"{column}_lag_{lag}"
    if "ticker" in df.columns:
        df[lag_column_name] = df.groupby("ticker")[column].shift(lag)
    else:
        df[lag_column_name] = df[column].shift(lag)
    return df


def log_missing_columns(feature_columns: Sequence[str], df: pd.DataFrame) -> None:
    """Log columns requested for lagging that are missing in ``df``."""
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        logger.debug("Columns skipped for lagging because they are missing: %s", missing_columns)


def _add_lags_for_column_vectorized(
    df: pd.DataFrame, column: str, lag_windows: Sequence[int]
) -> pd.DataFrame:
    """Add lag features for ``column`` using efficient groupby.shift."""
    if column not in df.columns:
        return df

    df_copy = df.copy()
    if "ticker" in df_copy.columns:
        for lag in lag_windows:
            lag_column_name = f"{column}_lag_{lag}"
            df_copy[lag_column_name] = df_copy.groupby("ticker")[column].shift(lag)
    else:
        for lag in lag_windows:
            lag_column_name = f"{column}_lag_{lag}"
            df_copy[lag_column_name] = df_copy[column].shift(lag)
    return df_copy


def add_lags_for_column(df: pd.DataFrame, column: str, lag_windows: Sequence[int]) -> pd.DataFrame:
    """Add lag features for one column after validating ``lag_windows``."""
    for lag in lag_windows:
        validate_lag_value(lag)
    return _add_lags_for_column_vectorized(df, column, lag_windows)


def add_lag_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    lag_windows: Sequence[int],
) -> pd.DataFrame:
    """Return a DataFrame with lagged versions of the requested columns.

    Order and sorting are validated to prevent leakage, and per-ticker grouping
    avoids cross-entity leakage.
    """
    logger.info(
        "Adding lag features for %d columns with windows %s",
        len(feature_columns),
        lag_windows,
    )
    validate_data_sorted_by_date(df, "add_lag_features")
    log_missing_columns(feature_columns, df)

    df_with_lags = df.copy()
    for column in feature_columns:
        if column not in df_with_lags.columns:
            continue
        df_with_lags = _add_lags_for_column_vectorized(df_with_lags, column, lag_windows)
    return df_with_lags


def get_base_columns(include_lags: bool) -> list[str]:
    """Return base feature columns (e.g., ``log_volatility``) with optional lags."""
    from src.constants import LIGHTGBM_BASE_FEATURE_COLUMNS

    base_cols = list(LIGHTGBM_BASE_FEATURE_COLUMNS)
    if not include_lags:
        return base_cols
    lagged = [f"{col}_lag_{lag}" for col in base_cols for lag in LIGHTGBM_LAG_WINDOWS]
    return base_cols + lagged


__all__ = [
    # Indicators
    "add_volume_features",
    "add_return_features",
    "add_calendar_features",
    "add_turnover_features",
    "add_obv_feature",
    "add_atr_feature",
    "add_custom_ml_indicators_per_ticker",
    # Lags
    "add_single_lag_feature",
    "add_lags_for_column",
    "add_lag_features",
    "get_base_columns",
]
