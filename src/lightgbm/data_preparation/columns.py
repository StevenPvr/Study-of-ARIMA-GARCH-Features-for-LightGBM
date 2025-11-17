"""Column utilities and selectors for dataset creation.

This module merges column selection (drop/keep) and reusable selectors to
reduce fragmentation and duplication in the data-prep code.
"""

from __future__ import annotations

import pandas as pd

from src.constants import LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS, LIGHTGBM_LAG_WINDOWS
from src.utils import get_logger

logger = get_logger(__name__)


# =============================
# Cleaning / filtering helpers
# =============================


def remove_missing_values(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove rows with missing values on selected columns (ignore meta columns).

    Meta columns (``date``, ``ticker``, ``ticker_id``, ``split``) never trigger
    row removal to avoid excessive loss.
    """
    initial_rows = len(df)
    meta_cols: set[str] = {"date", "ticker", "ticker_id", "split"}

    if subset is None:
        subset = [c for c in df.columns if c not in meta_cols]
    else:
        subset = [c for c in subset if c not in meta_cols]

    if not subset:
        logger.info("No subset columns provided for dropna; returning original dataframe")
        return df.reset_index(drop=True).copy()

    logger.info(
        "Removing rows with missing values from %d columns (ignoring meta columns)",
        len(subset),
    )
    df_clean = df.dropna(subset=subset).reset_index(drop=True).copy()
    removed_rows = initial_rows - len(df_clean)
    logger.info(
        "Removed %d rows with missing values (from %d to %d)",
        removed_rows,
        initial_rows,
        len(df_clean),
    )
    return df_clean


def get_base_insight_columns(df: pd.DataFrame) -> list[str]:
    """List ARIMA-GARCH insight base columns present in ``df``."""
    return [col for col in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS if col in df.columns]


def get_lagged_insight_columns(df: pd.DataFrame) -> list[str]:
    """List lagged ARIMA-GARCH insight columns present in ``df``."""
    lagged_cols: list[str] = []
    for col in LIGHTGBM_ARIMA_GARCH_INSIGHT_COLUMNS:
        lagged_cols.extend(
            [f"{col}_lag_{lag}" for lag in LIGHTGBM_LAG_WINDOWS if f"{col}_lag_{lag}" in df.columns]
        )
    return lagged_cols


def get_insight_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Columns to drop when removing all insights (base + lags)."""
    base_cols = get_base_insight_columns(df)
    lagged_cols = get_lagged_insight_columns(df)
    return base_cols + lagged_cols


def get_sigma2_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Columns to drop when removing only ``sigma2_garch`` and its lags."""
    columns_to_drop: list[str] = []
    if "sigma2_garch" in df.columns:
        columns_to_drop.append("sigma2_garch")
    for lag in LIGHTGBM_LAG_WINDOWS:
        col_name = f"sigma2_garch_lag_{lag}"
        if col_name in df.columns:
            columns_to_drop.append(col_name)
    return columns_to_drop


def get_non_observable_columns_to_drop(df: pd.DataFrame) -> list[str]:
    """Non-observable columns at prediction time to remove (prevent leakage)."""
    non_observable_columns = [
        "sigma2_garch_true",
        "sigma_garch_true",
        "std_resid_garch_true",
        "sigma2_egarch_raw",
        "target_date",
    ]
    return [col for col in non_observable_columns if col in df.columns]


def get_technical_indicator_columns(include_lags: bool) -> list[str]:
    """Return list of technical indicator columns (with optional lags)."""
    from src.constants import LIGHTGBM_TECHNICAL_FEATURE_COLUMNS

    base_cols = list(LIGHTGBM_TECHNICAL_FEATURE_COLUMNS)
    if not include_lags:
        return base_cols
    lagged_cols = [
        f"{indicator}_lag_{lag}" for indicator in base_cols for lag in LIGHTGBM_LAG_WINDOWS
    ]
    return base_cols + lagged_cols


# =============================
# Reusable selectors (composable)
# =============================


def select_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Select metadata columns present in ``df`` (date/split/ticker/id)."""
    base_cols = ["date", "split"]
    keep_cols: list[str] = [col for col in base_cols if col in df.columns]
    if "ticker" in df.columns:
        keep_cols.append("ticker")
    if "tickers" in df.columns:
        keep_cols.append("tickers")
    if "ticker_id" in df.columns:
        keep_cols.append("ticker_id")
    return keep_cols


def select_target_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select target column with optional lags."""
    from src.lightgbm.data_preparation.target_creation import get_target_column_name

    keep_cols: list[str] = []
    return_col = get_target_column_name(df)
    if return_col in df.columns:
        keep_cols.append(return_col)
    if include_lags:
        lag_cols = [col for col in df.columns if col.startswith(f"{return_col}_lag_")]
        keep_cols.extend(lag_cols)
    return keep_cols


def select_garch_insight_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select ARIMA-GARCH insight columns with optional lags."""
    keep_cols: list[str] = []
    insight_cols = [
        "sarima_pred",
        "sarima_resid",
        "sigma2_garch",
        "sigma_garch",
        "std_resid_garch",
    ]

    for insight_col in insight_cols:
        if insight_col in df.columns:
            keep_cols.append(insight_col)
        if include_lags:
            lag_cols = [
                f"{insight_col}_lag_{lag}"
                for lag in LIGHTGBM_LAG_WINDOWS
                if f"{insight_col}_lag_{lag}" in df.columns
            ]
            keep_cols.extend(lag_cols)
    return keep_cols


def select_base_feature_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select base feature columns (e.g., ``log_volatility``) with optional lags."""
    from src.lightgbm.data_preparation.features import get_base_columns

    base_cols = get_base_columns(include_lags)
    return [col for col in base_cols if col in df.columns]


def select_technical_indicator_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select technical indicators (optionally with lags) that exist in ``df``."""
    tech_cols = get_technical_indicator_columns(include_lags)
    return [col for col in tech_cols if col in df.columns]


def select_calendar_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select non-lagged calendar feature columns present in ``df``."""
    from src.constants import LIGHTGBM_CALENDAR_FEATURE_COLUMNS

    return [col for col in LIGHTGBM_CALENDAR_FEATURE_COLUMNS if col in df.columns]


def select_close_column(df: pd.DataFrame) -> list[str]:
    """Return ["close"] if present, otherwise an empty list."""
    return ["close"] if "close" in df.columns else []


def select_sigma_plus_base_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for sigma-plus-base dataset (metadata + target + insights + base + close)."""
    keep_cols: list[str] = []
    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_garch_insight_columns(df, include_lags))
    keep_cols.extend(select_base_feature_columns(df, include_lags))
    keep_cols.extend(select_close_column(df))
    return list(dict.fromkeys(keep_cols))


def select_log_volatility_only_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for log-volatility-only dataset (metadata + target + base + close)."""
    keep_cols: list[str] = []
    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_base_feature_columns(df, include_lags))
    keep_cols.extend(select_close_column(df))
    return keep_cols


def select_technical_indicators_dataset_columns(df: pd.DataFrame, include_lags: bool) -> list[str]:
    """Select columns for technical indicators dataset
    (metadata + target + tech + calendar + close)."""
    keep_cols: list[str] = []
    keep_cols.extend(select_metadata_columns(df))
    keep_cols.extend(select_target_columns(df, include_lags))
    keep_cols.extend(select_technical_indicator_columns(df, include_lags))
    keep_cols.extend(select_calendar_feature_columns(df))
    keep_cols.extend(select_close_column(df))
    return list(dict.fromkeys(keep_cols))


## Removed get_sigma_plus_base_feature_columns_for_lag (unused).


__all__ = [
    # cleaning
    "remove_missing_values",
    "get_insight_columns_to_drop",
    "get_sigma2_columns_to_drop",
    "get_non_observable_columns_to_drop",
    "get_technical_indicator_columns",
    # selectors
    "select_metadata_columns",
    "select_target_columns",
    "select_garch_insight_columns",
    "select_base_feature_columns",
    "select_technical_indicator_columns",
    "select_calendar_feature_columns",
    "select_close_column",
    "select_sigma_plus_base_columns",
    "select_log_volatility_only_columns",
    "select_technical_indicators_dataset_columns",
]
