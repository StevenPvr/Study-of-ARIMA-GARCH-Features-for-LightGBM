"""Data quality analysis functions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.constants import REQUIRED_OHLCV_COLUMNS, TOP_N_TICKERS_REPORT
from src.data_cleaning.validation import assert_not_empty, assert_required_columns
from src.utils import get_logger
from src.utils.temporal import is_datetime_series_monotonic_increasing

logger = get_logger(__name__)


def _ensure_basic_structure(raw_df: pd.DataFrame) -> None:
    """Validate minimal structure for analysis functions.

    Raises a clear error early if the DataFrame is empty or structurally
    invalid. This keeps individual analysis functions simple (KISS).
    """
    assert_not_empty(raw_df, name="raw dataset")
    assert_required_columns(raw_df, REQUIRED_OHLCV_COLUMNS)


def analyze_general_statistics(raw_df: pd.DataFrame) -> None:
    """Analyze and log general statistics about the dataset.

    Args:
        raw_df: Raw dataset DataFrame.
    """
    _ensure_basic_structure(raw_df)

    logger.info("\n1. GENERAL STATISTICS")
    logger.info("   Total observations: %s", f"{len(raw_df):,}")
    logger.info("   Unique tickers: %s", raw_df["tickers"].nunique())
    logger.info(
        "   Period: %s → %s",
        raw_df["date"].min().date(),
        raw_df["date"].max().date(),
    )


def analyze_missing_values(raw_df: pd.DataFrame) -> None:
    """Analyze and log missing values per column.

    Args:
        raw_df: Raw dataset DataFrame.
    """
    _ensure_basic_structure(raw_df)

    logger.info("\n2. MISSING VALUES")
    na_counts = raw_df.isna().sum()
    total_rows = len(raw_df)

    for col, count in na_counts.items():
        if count == 0:
            continue
        pct = 100.0 * count / total_rows
        logger.info("   %s: %d (%.2f%%)", col, int(count), pct)


def analyze_outliers(raw_df: pd.DataFrame) -> None:
    """Analyze and log simple outlier statistics on key numeric columns.

    Args:
        raw_df: Raw dataset DataFrame.
    """
    _ensure_basic_structure(raw_df)

    logger.info("\n3. OUTLIERS (simple quantiles)")
    numeric_columns: Iterable[str] = ["close", "volume"]

    for col in numeric_columns:
        if col not in raw_df.columns:
            continue

        series = raw_df[col].dropna()
        if series.empty:
            continue

        q01 = series.quantile(0.01)
        q50 = series.quantile(0.50)
        q99 = series.quantile(0.99)
        logger.info("   %s: q1%%=%.4f, median=%.4f, q99%%=%.4f", col, q01, q50, q99)


def analyze_ticker_distribution(raw_df: pd.DataFrame) -> None:
    """Analyze and log distribution of observations by ticker.

    Args:
        raw_df: Raw dataset DataFrame.
    """
    _ensure_basic_structure(raw_df)

    logger.info("\n4. TICKER DISTRIBUTION")
    obs_per_ticker = raw_df.groupby("tickers").size()  # type: ignore[return-value]

    logger.info("   Min observations per ticker: %d", int(obs_per_ticker.min()))
    logger.info("   Median observations per ticker: %d", int(obs_per_ticker.median()))
    logger.info("   Max observations per ticker: %d", int(obs_per_ticker.max()))


def report_least_observations(raw_df: pd.DataFrame, top_n: int = TOP_N_TICKERS_REPORT) -> None:
    """Report tickers with the fewest observations.

    Args:
        raw_df: Raw dataset DataFrame.
        top_n: Number of tickers to display.
    """
    _ensure_basic_structure(raw_df)

    logger.info("\n5. TICKERS WITH FEWEST OBSERVATIONS")
    obs_per_ticker: pd.Series = raw_df.groupby("tickers").size()  # type: ignore[assignment]
    least_obs = obs_per_ticker.sort_values(ascending=True).head(top_n)

    for ticker, count in least_obs.items():
        logger.info("   %s: %d observations", ticker, int(count))


# ---- Monotonicity checks used by metrics.py ---------------------------------


def has_required_columns_for_monotonicity(raw_df: pd.DataFrame) -> bool:
    """Return True if DataFrame has the minimal columns for monotonicity checks."""
    return {"tickers", "date"}.issubset(raw_df.columns) and not raw_df.empty


# Use the generic monotonicity check from utils
is_ticker_monotonic = is_datetime_series_monotonic_increasing


def compute_monotonicity_violations(raw_df: pd.DataFrame) -> int:
    """Count tickers where dates are not strictly increasing.

    Args:
        raw_df: DataFrame with ``ticker`` and ``date`` columns.

    Returns:
        Number of tickers with non-monotonic dates.
    """
    if not has_required_columns_for_monotonicity(raw_df):
        return 0

    non_mono = 0
    for _, grp in raw_df.groupby("tickers"):
        if not is_ticker_monotonic(grp["date"]):  # type: ignore[arg-type]
            non_mono += 1

    return int(non_mono)
