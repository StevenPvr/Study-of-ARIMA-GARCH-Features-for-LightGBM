"""Data loading functions for LightGBM data preparation."""

from __future__ import annotations

import pandas as pd

from src.path import DATA_TICKERS_FULL_FILE, DATA_TICKERS_FULL_INSIGHTS_FILE
from src.utils import get_logger, load_dataframe

logger = get_logger(__name__)


def ensure_ticker_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ticker_id column exists for ticker-level data."""
    if "ticker" not in df.columns:
        return df

    df_with_id = df.copy()
    if "ticker_id" not in df_with_id.columns:
        ticker_codes = df_with_id["ticker"].astype("category").cat.codes.astype("int32")
        df_with_id["ticker_id"] = ticker_codes
    return df_with_id


## Removed add_split_column (unused); split creation is handled in target pipeline.


def _merge_volume_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Merge volume column from filtered dataset if missing.

    Args:
        df: DataFrame that may be missing volume column.

    Returns:
        DataFrame with volume column added if it was missing.
    """
    if "volume" in df.columns:
        return df

    try:
        from src.path import DATASET_FILTERED_FILE

        if DATASET_FILTERED_FILE.exists():
            base = load_dataframe(
                DATASET_FILTERED_FILE, date_columns=["date"], validate_not_empty=False
            )
            # Handle both 'ticker' and 'tickers' column names
            ticker_col = "tickers" if "tickers" in base.columns else "ticker"
            merge_cols = ["date", ticker_col, "volume"]

            # Normalize ticker column name in base if needed
            if ticker_col == "tickers" and "ticker" in df.columns:
                base = base.rename(columns={"tickers": "ticker"})
                ticker_col = "ticker"

            if all(col in base.columns for col in merge_cols):
                df = df.merge(base[merge_cols], on=["date", ticker_col], how="left")
                logger.info("Merged 'volume' into dataframe from DATASET_FILTERED_FILE")
    except Exception as e:  # pragma: no cover - best-effort enrichment
        logger.warning(f"Could not merge volume into data: {e}")

    return df


def load_ticker_data_with_fallback(prefer_insights: bool = True) -> pd.DataFrame:
    """Load ticker-level data (individual company tickers only, no aggregated data).

    Loads raw ticker data without pre-computed technical indicators, so indicators can be
    calculated on-the-fly by the calling functions. If prefer_insights=True, loads data
    with GARCH insights (sigma forecasts).

    Args:
        prefer_insights: If True, prefer data with GARCH insights over data without insights.

    Returns:
        DataFrame with ticker-level data (with 'ticker' column).

    Raises:
        FileNotFoundError: If no ticker-level data files are found.
    """
    # Try to load data with GARCH insights first (no pre-computed technical indicators)
    if prefer_insights and DATA_TICKERS_FULL_INSIGHTS_FILE.exists():
        logger.info(
            f"Loading ticker-level data with GARCH insights from {DATA_TICKERS_FULL_INSIGHTS_FILE}"
        )
        df = load_dataframe(
            DATA_TICKERS_FULL_INSIGHTS_FILE, date_columns=["date"], validate_not_empty=False
        )
        # Normalize ticker column name: accept 'tickers' and rename to 'ticker'
        if "ticker" not in df.columns and "tickers" in df.columns:
            df = df.rename(columns={"tickers": "ticker"})
        df = _merge_volume_if_missing(df)
        return ensure_ticker_id_column(df)

    # Load raw ticker data (without insights or pre-computed indicators)
    if DATA_TICKERS_FULL_FILE.exists():
        logger.info(f"Loading raw ticker-level data from {DATA_TICKERS_FULL_FILE}")
        df = load_dataframe(DATA_TICKERS_FULL_FILE, date_columns=["date"], validate_not_empty=False)
        if "ticker" not in df.columns and "tickers" in df.columns:
            df = df.rename(columns={"tickers": "ticker"})
        df = _merge_volume_if_missing(df)
        return ensure_ticker_id_column(df)

    msg = (
        "No ticker-level data found. Please run data preparation pipeline first to generate "
        f"{DATA_TICKERS_FULL_FILE}"
    )
    raise FileNotFoundError(msg)
