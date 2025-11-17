"""Data quality metrics computation functions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.constants import TOP_N_TICKERS_REPORT
from src.data_cleaning.analysis import compute_monotonicity_violations
from src.utils import get_logger

logger = get_logger(__name__)


def compute_empty_quality_metrics() -> dict[str, Any]:
    """Return empty quality metrics for an empty DataFrame.

    Returns:
        Dictionary with empty quality metrics.
    """
    return {
        "na_by_column": {},
        "duplicate_rows_on_date_ticker": 0,
        "rows_with_nonpositive_volume": 0,
        "non_monotonic_ticker_dates": 0,
        "top_missing_business_days": [],
    }


def compute_missing_business_days(raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute per-ticker missing business days between first and last date.

    For each ticker, the function builds the expected business-day calendar
    between its minimum and maximum date, compares it to the actual dates
    present in the data, and counts missing business days.

    Args:
        raw_df: Raw dataset DataFrame.

    Returns:
        A list of dictionaries sorted by number of missing days in descending
        order, truncated to ``TOP_N_TICKERS_REPORT`` elements. Each dict
        contains keys: ``ticker`` and ``missing_business_days``.
    """
    if raw_df.empty or "tickers" not in raw_df.columns or "date" not in raw_df.columns:
        return []

    results: list[dict[str, Any]] = []

    for ticker, grp in raw_df.groupby("tickers"):
        dates: pd.Series = grp["date"].dropna().sort_values(ascending=True)  # type: ignore[assignment]
        if dates.empty:
            continue

        # Normalize dates to naive (timezone-unaware) to avoid timezone mismatch errors
        # This handles cases where dates might have timezone info from parsing
        try:
            if hasattr(dates, "dt") and dates.dt.tz is not None:
                dates = dates.dt.tz_convert(None)  # type: ignore[assignment]
        except (AttributeError, TypeError):
            # Handle case where dates is a mock or doesn't support .dt accessor
            pass

        min_date = dates.min()
        max_date = dates.max()
        expected = pd.date_range(min_date, max_date, freq="B")
        missing = expected.difference(pd.DatetimeIndex(dates.unique()))
        if len(missing) == 0:
            continue

        results.append(
            {
                "ticker": str(ticker),
                "missing_business_days": int(len(missing)),
            }
        )

    # Sort by number of missing days, descending, and keep the top N
    results.sort(key=lambda d: d["missing_business_days"], reverse=True)
    return results[:TOP_N_TICKERS_REPORT]


def compute_quality_metrics(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Compute a concise set of data quality metrics for JSON reporting.

    Args:
        raw_df: DataFrame to analyze.

    Returns:
        Dictionary with quality metrics.
    """
    if raw_df.empty:
        return compute_empty_quality_metrics()

    metrics: dict[str, Any] = {}

    # Missing values per column
    metrics["na_by_column"] = {col: int(v) for col, v in raw_df.isna().sum().items()}

    # Duplicate rows on (date, tickers)
    if {"date", "tickers"}.issubset(raw_df.columns):
        dup_mask = raw_df.duplicated(subset=["date", "tickers"])
        metrics["duplicate_rows_on_date_ticker"] = int(dup_mask.sum())
    else:
        metrics["duplicate_rows_on_date_ticker"] = 0

    # Rows with non-positive volume (if volume column exists)
    if "volume" in raw_df.columns:
        metrics["rows_with_nonpositive_volume"] = int((raw_df["volume"] <= 0).sum())
    else:
        metrics["rows_with_nonpositive_volume"] = 0

    # Monotonicity violations by ticker
    metrics["non_monotonic_ticker_dates"] = compute_monotonicity_violations(raw_df)

    # Missing business days per ticker
    metrics["top_missing_business_days"] = compute_missing_business_days(raw_df)

    return metrics
