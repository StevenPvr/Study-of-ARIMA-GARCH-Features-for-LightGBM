"""Dataset preparation pipeline functions.

This module provides reusable functions for dataset preparation to avoid duplication.
"""

from __future__ import annotations

import pandas as pd

from src.lightgbm.data_preparation.data_loading import load_ticker_data_with_fallback
from src.lightgbm.data_preparation.target_creation import normalize_column_names


def prepare_base_dataframe(df: pd.DataFrame | None, prefer_insights: bool = True) -> pd.DataFrame:
    """Prepare base dataframe with normalization and sorting.

    Args:
        df: Input DataFrame or None to load from file.
        prefer_insights: If True, prefer data with GARCH insights.

    Returns:
        Prepared DataFrame with normalized columns and proper sorting.
    """
    if df is None:
        df = load_ticker_data_with_fallback(prefer_insights=prefer_insights)

    df_normalized = normalize_column_names(df)

    # Sort by ticker and date for proper lag calculation
    if "ticker" in df_normalized.columns:
        df_normalized = df_normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    elif "date" in df_normalized.columns:
        df_normalized = df_normalized.sort_values("date").reset_index(drop=True)

    return df_normalized


def drop_rows_with_missing_features(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    """Drop rows with NaN on feature/target columns only.

    Meta columns (date, ticker, ticker_id, split) are preserved even if they
    contain NaN to avoid excessive row loss from auxiliary metadata.

    Args:
        df: DataFrame to clean.
        keep_cols: Columns to keep in the final dataset.

    Returns:
        Cleaned DataFrame without missing values.
    """
    meta_cols: set[str] = {"date", "ticker", "ticker_id", "split"}
    subset = [c for c in keep_cols if c not in meta_cols]

    # Use .loc to ensure DataFrame return type (not Series)
    df_filtered = df.loc[:, keep_cols].copy()

    if not subset:
        return df_filtered.reset_index(drop=True)

    return df_filtered.dropna(subset=subset).reset_index(drop=True)
