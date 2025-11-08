"""Utility functions for data cleaning operations."""

from __future__ import annotations

import json
from typing import Any, cast

import pandas as pd

from src.constants import (
    DATA_QUALITY_REPORT_FILE,
    DATASET_FILE,
    DATASET_FILTERED_FILE,
    MIN_VOLUME_THRESHOLD,
    MONOTONICITY_CHECK_DEFAULT_DIFF,
    TOP_N_TICKERS_REPORT,
)
from src.utils import get_logger

logger = get_logger(__name__)

_REQUIRED_COLUMNS = ["date", "ticker", "open", "closing", "volume"]


def validate_file_exists() -> None:
    """Validate that the dataset file exists.

    Raises:
        FileNotFoundError: If dataset file does not exist.
    """
    if not DATASET_FILE.exists():
        msg = f"Dataset file not found: {DATASET_FILE}"
        raise FileNotFoundError(msg)


def validate_columns(raw_df: pd.DataFrame) -> None:
    """Validate that required columns are present in the DataFrame.

    Args:
        raw_df: DataFrame to validate.

    Raises:
        KeyError: If required columns are missing.
    """
    missing_columns = [col for col in _REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)


def convert_and_validate_dates(raw_df: pd.DataFrame) -> None:
    """Convert date column to datetime and log invalid dates.

    Args:
        raw_df: DataFrame with date column to convert. Modified in-place.

    Raises:
        KeyError: If 'date' column is missing.
    """
    if "date" not in raw_df.columns:
        msg = "Missing required column: 'date'"
        raise KeyError(msg)

    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    invalid_dates = raw_df["date"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Found {invalid_dates} invalid dates that were set to NaT")


def load_dataset() -> pd.DataFrame:
    """Load and prepare the raw dataset.

    Returns:
        DataFrame with date column converted to datetime.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        KeyError: If required columns are missing.
        ValueError: If dataset is empty.
    """
    validate_file_exists()

    raw_df = pd.read_csv(DATASET_FILE)
    validate_columns(raw_df)

    if raw_df.empty:
        msg = "Dataset is empty"
        raise ValueError(msg)

    convert_and_validate_dates(raw_df)

    return raw_df


def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on (date, ticker) combination.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, number of duplicates removed).
    """
    before = len(df)
    df_cleaned = df.drop_duplicates(subset=["date", "ticker"], keep="first")
    duplicates_removed = before - len(df_cleaned)
    return df_cleaned, duplicates_removed


def _replace_invalid_volumes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace non-positive volume values with NaN.

    Preserves temporal continuity for return calculations.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, number of volumes replaced).
    """
    if "volume" not in df.columns:
        return df, 0

    invalid_volume_mask = df["volume"] <= MIN_VOLUME_THRESHOLD
    volumes_replaced = int(invalid_volume_mask.sum())

    if volumes_replaced > 0:
        df = df.copy()
        df.loc[invalid_volume_mask, "volume"] = pd.NA

    return df, volumes_replaced


def _complete_ticker_dates(ticker_df: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Complete date range for a ticker by adding missing dates with NaN.

    Args:
        ticker_df: DataFrame with data for a single ticker.
        all_dates: Full date range to complete.

    Returns:
        DataFrame with all dates, missing ones filled with NaN.
    """
    full_dates = pd.DataFrame({"date": all_dates, "ticker": ticker_df["ticker"].iloc[0]})
    ticker_complete = full_dates.merge(ticker_df, on=["date", "ticker"], how="left")

    # Fill missing numeric columns with NaN
    numeric_columns = ["open", "closing", "volume"]
    for col in numeric_columns:
        if col in ticker_complete.columns:
            ticker_complete[col] = ticker_complete[col].fillna(pd.NA)

    return ticker_complete


def _get_incomplete_tickers(obs_per_ticker: pd.Series) -> list[str]:
    """Identify tickers with incomplete observations.

    Args:
        obs_per_ticker: Series with observation counts per ticker.

    Returns:
        List of ticker symbols with incomplete observations.
    """
    if len(obs_per_ticker) == 0:
        return []

    max_obs = obs_per_ticker.max()
    return [
        str(ticker) for ticker, count in obs_per_ticker.items() if count < max_obs
    ]


def _process_incomplete_ticker(
    df: pd.DataFrame, ticker: str, all_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Process a single incomplete ticker by filling missing dates.

    Args:
        df: DataFrame to update.
        ticker: Ticker symbol to process.
        all_dates: Full date range to complete.

    Returns:
        Updated DataFrame with completed dates for the ticker.
    """
    ticker_mask = df["ticker"] == ticker
    ticker_df = cast(pd.DataFrame, df[ticker_mask]).copy()
    ticker_complete = _complete_ticker_dates(ticker_df, all_dates)
    df = cast(pd.DataFrame, df[~ticker_mask]).copy()
    return pd.concat([df, ticker_complete], ignore_index=True, sort=False)


def _has_required_columns_for_filling(df: pd.DataFrame) -> bool:
    """Check if DataFrame has required columns for filling missing dates.

    Args:
        df: DataFrame to check.

    Returns:
        True if required columns are present and DataFrame is not empty.
    """
    return (
        "ticker" in df.columns
        and "date" in df.columns
        and len(df) > 0
    )


def _fill_missing_dates_for_incomplete_tickers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing dates for incomplete tickers with NaN.

    Tickers contribute to calculations only for dates where they exist.
    Preserves temporal continuity while excluding missing periods.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, number of incomplete tickers processed).
    """
    if not _has_required_columns_for_filling(df):
        return df, 0

    obs_per_ticker_series = cast(pd.Series, df.groupby("ticker").size())
    incomplete_tickers = _get_incomplete_tickers(obs_per_ticker_series)
    incomplete_count = len(incomplete_tickers)

    if incomplete_count == 0:
        return df, 0

    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="B")

    for ticker in incomplete_tickers:
        df = _process_incomplete_ticker(df, ticker, all_dates)

    return df, incomplete_count


def apply_basic_integrity_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply simple integrity fixes and return counts of fixes.

    Args:
        df: DataFrame to clean.

    Returns:
        Tuple of (cleaned DataFrame, dictionary with fix counts).

    - Drop duplicate (date,ticker) rows
    - Replace non-positive volume with NaN (preserves time series continuity)
    - Fill missing dates for incomplete tickers with NaN (preserves temporal continuity)
    - Sort by ticker, date

    Note:
        Volume values <= MIN_VOLUME_THRESHOLD are replaced with NaN instead of
        removing rows to preserve temporal continuity for return calculations.
        Tickers with incomplete observations have missing dates filled with NaN,
        allowing them to contribute to calculations only for dates where they exist.
        ARIMA/GARCH models and liquidity weight calculations handle NaN values correctly.
    """
    if df.empty:
        return df.copy(), {
            "duplicates_removed": 0,
            "nonpositive_volume_replaced": 0,
            "incomplete_tickers_nullified": 0,
        }

    counters: dict[str, int] = {}

    # Step 1: Remove duplicates
    df, counters["duplicates_removed"] = _remove_duplicates(df)

    # Step 2: Replace invalid volumes with NaN
    df, counters["nonpositive_volume_replaced"] = _replace_invalid_volumes(df)

    # Step 3: Fill missing dates for incomplete tickers
    df, counters["incomplete_tickers_nullified"] = _fill_missing_dates_for_incomplete_tickers(df)

    # Final sort
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df, counters


def analyze_general_statistics(raw_df: pd.DataFrame) -> None:
    """Analyze and log general statistics about the dataset.

    Args:
        raw_df: Raw dataset DataFrame.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If DataFrame is empty.
    """
    if raw_df.empty:
        msg = "DataFrame is empty"
        raise ValueError(msg)

    required_columns = ["date", "ticker"]
    missing_columns = [col for col in required_columns if col not in raw_df.columns]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)

    logger.info("\n1. GENERAL STATISTICS")
    logger.info(f"   Total observations: {len(raw_df):,}")
    logger.info(f"   Unique tickers: {raw_df['ticker'].nunique()}")
    logger.info(f"   Period: {raw_df['date'].min().date()} → {raw_df['date'].max().date()}")
    logger.info(f"   Unique dates: {raw_df['date'].nunique()}")


def log_missing_value_for_column(col: str, missing: pd.Series, total_rows: int) -> None:
    """Log missing value statistics for a single column.

    Args:
        col: Column name.
        missing: Series with missing value counts per column.
        total_rows: Total number of rows in the dataset.

    Raises:
        KeyError: If column is not in missing Series.
    """
    if col not in missing.index:
        logger.warning(f"Column '{col}' not found in missing values Series")
        return

    count = int(missing[col])
    pct = (count / total_rows * 100) if total_rows > 0 else 0.0
    logger.info(f"   {col:<8}: {count} ({pct:.2f}%)")


def analyze_missing_values(raw_df: pd.DataFrame) -> None:
    """Analyze and log missing values in the dataset.

    Args:
        raw_df: Raw dataset DataFrame.

    Raises:
        KeyError: If required columns are missing.
    """
    logger.info("\n2. MISSING VALUES")
    validate_columns(raw_df)

    missing = raw_df.isna().sum()
    total_rows = len(raw_df)
    for col in _REQUIRED_COLUMNS:
        log_missing_value_for_column(col, missing, total_rows)


def analyze_outliers(raw_df: pd.DataFrame) -> None:
    """Analyze and log outliers (zero or negative values).

    Args:
        raw_df: Raw dataset DataFrame.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If DataFrame is empty.
    """
    logger.info("\n3. OUTLIERS")
    if raw_df.empty:
        logger.warning("DataFrame is empty, skipping outlier analysis")
        return

    required_columns = ["open", "closing", "volume"]
    missing_columns = [col for col in required_columns if col not in raw_df.columns]
    if missing_columns:
        msg = f"Missing required columns: {missing_columns}"
        raise KeyError(msg)

    zero_open = int((raw_df["open"] <= MIN_VOLUME_THRESHOLD).sum())
    zero_closing = int((raw_df["closing"] <= MIN_VOLUME_THRESHOLD).sum())
    zero_volume = int((raw_df["volume"] <= MIN_VOLUME_THRESHOLD).sum())
    logger.info(f"   Opening price <= {MIN_VOLUME_THRESHOLD}: {zero_open}")
    logger.info(f"   Closing price <= {MIN_VOLUME_THRESHOLD}: {zero_closing}")
    logger.info(f"   Volume <= {MIN_VOLUME_THRESHOLD}: {zero_volume}")


def is_ticker_monotonic(dates: pd.Series) -> bool:
    """Check if dates are strictly increasing for a single ticker.

    Args:
        dates: Series of dates for a ticker.

    Returns:
        True if dates are strictly increasing, False otherwise.
    """
    dates_converted = pd.to_datetime(dates, errors="coerce")
    if dates_converted.isna().any():
        return True  # Skip invalid dates, consider as monotonic

    diffs = dates_converted.diff().dt.total_seconds().fillna(MONOTONICITY_CHECK_DEFAULT_DIFF)
    return (diffs <= 0).sum() == 0


def has_required_columns_for_monotonicity(df: pd.DataFrame) -> bool:
    """Check if DataFrame has required columns for monotonicity check.

    Args:
        df: DataFrame to check.

    Returns:
        True if required columns are present, False otherwise.
    """
    return "ticker" in df.columns and "date" in df.columns and not df.empty


def compute_monotonicity_violations(raw_df: pd.DataFrame) -> int:
    """Count tickers where dates are not strictly increasing.

    Args:
        raw_df: DataFrame with 'ticker' and 'date' columns.

    Returns:
        Number of tickers with non-monotonic dates.
    """
    if not has_required_columns_for_monotonicity(raw_df):
        return 0

    non_mono = sum(
        1
        for _, grp in raw_df.groupby("ticker")
        if not is_ticker_monotonic(grp["date"])  # type: ignore[arg-type]
    )
    return int(non_mono)


def analyze_ticker_distribution(raw_df: pd.DataFrame) -> pd.Series:
    """Analyze distribution of observations per ticker.

    Args:
        raw_df: Raw dataset DataFrame.

    Returns:
        Series with observations count per ticker.

    Raises:
        KeyError: If 'ticker' column is missing.
        ValueError: If DataFrame is empty.
    """
    if "ticker" not in raw_df.columns:
        msg = "Missing required column: 'ticker'"
        raise KeyError(msg)

    if raw_df.empty:
        msg = "DataFrame is empty"
        raise ValueError(msg)

    logger.info("\n4. DISTRIBUTION BY TICKER")
    obs_per_ticker: pd.Series = raw_df.groupby("ticker").size()  # type: ignore[assignment]

    if len(obs_per_ticker) == 0:
        logger.warning("No tickers found in dataset")
        return obs_per_ticker

    max_obs = int(obs_per_ticker.max())
    min_obs = int(obs_per_ticker.min())
    logger.info(f"   Min observations: {min_obs}")
    logger.info(f"   Max observations: {max_obs}")
    logger.info(f"   Mean observations: {obs_per_ticker.mean():.0f}")
    logger.info(f"   Median observations: {obs_per_ticker.median():.0f}")

    tickers_incomplete = int((obs_per_ticker < max_obs).sum())
    pct_incomplete = (
        (tickers_incomplete / len(obs_per_ticker) * 100) if len(obs_per_ticker) > 0 else 0.0
    )
    logger.info(f"\n   ⚠ {tickers_incomplete} tickers have less than {max_obs} observations")
    logger.info(f"     ({pct_incomplete:.1f}% of tickers)")

    return obs_per_ticker


def report_least_observations(obs_per_ticker: pd.Series) -> None:
    """Report tickers with the least observations.

    Args:
        obs_per_ticker: Series with observations count per ticker.

    Raises:
        ValueError: If Series is empty.
    """
    if len(obs_per_ticker) == 0:
        logger.warning("No tickers to report")
        return

    logger.info(f"\n5. TICKERS WITH LEAST OBSERVATIONS (top {TOP_N_TICKERS_REPORT})")
    max_obs = float(obs_per_ticker.max())
    n_to_show = min(TOP_N_TICKERS_REPORT, len(obs_per_ticker))
    least_obs_series = obs_per_ticker.nsmallest(n_to_show)  # type: ignore[arg-type]
    for ticker, count in least_obs_series.items():
        pct_complete = (float(count) / max_obs * 100) if max_obs > 0 else 0.0
        logger.info(f"   {ticker:6s}: {int(count):4d} observations ({pct_complete:.1f}% complete)")


def report_low_volume_tickers(raw_df: pd.DataFrame) -> None:
    """Report tickers with low average volume.

    Args:
        raw_df: Raw dataset DataFrame.

    Raises:
        KeyError: If 'ticker' or 'volume' columns are missing.
    """
    if "ticker" not in raw_df.columns or "volume" not in raw_df.columns:
        msg = "Missing required columns: 'ticker' and/or 'volume'"
        raise KeyError(msg)

    if raw_df.empty:
        logger.warning("No data to analyze for low volume tickers")
        return

    logger.info(f"\n6. TICKERS WITH LOW AVERAGE VOLUME (top {TOP_N_TICKERS_REPORT})")
    avg_volume_series = (
        raw_df.groupby("ticker")["volume"].mean().nsmallest(TOP_N_TICKERS_REPORT)  # type: ignore[arg-type]
    )
    for ticker, vol in avg_volume_series.items():
        logger.info(f"   {ticker:6s}: {float(vol):,.0f} average volume")


def compute_empty_quality_metrics() -> dict[str, Any]:
    """Return empty quality metrics for empty DataFrame.

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


def compute_basic_quality_metrics(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Compute basic quality metrics (NA, duplicates, volume).

    Args:
        raw_df: DataFrame to analyze.

    Returns:
        Dictionary with basic quality metrics.
    """
    na_by_col = {c: int(raw_df[c].isna().sum()) for c in _REQUIRED_COLUMNS if c in raw_df.columns}
    dup_count = int(raw_df.duplicated(subset=["date", "ticker"]).sum())

    if "volume" in raw_df.columns:
        invalid_volume = int((raw_df["volume"] <= MIN_VOLUME_THRESHOLD).sum())
    else:
        invalid_volume = 0

    return {
        "na_by_column": na_by_col,
        "duplicate_rows_on_date_ticker": dup_count,
        "rows_with_nonpositive_volume": invalid_volume,
    }


def compute_missing_days_for_ticker(ticker: str, dates: pd.Series) -> dict[str, Any] | None:
    """Compute missing business days for a single ticker.

    Args:
        ticker: Ticker symbol.
        dates: Series of dates for the ticker.

    Returns:
        Dictionary with ticker and missing business days, or None if no missing days.
    """
    dates_converted = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if dates_converted.empty:
        return None

    expected = pd.date_range(dates_converted.min(), dates_converted.max(), freq="B")
    dates_index = pd.DatetimeIndex(dates_converted)
    missing = int(len(expected.difference(dates_index)))
    if missing > 0:
        return {"ticker": str(ticker), "missing_business_days": missing}
    return None


def has_required_columns_for_missing_days(df: pd.DataFrame) -> bool:
    """Check if DataFrame has required columns for missing days computation.

    Args:
        df: DataFrame to check.

    Returns:
        True if required columns are present, False otherwise.
    """
    return "ticker" in df.columns and "date" in df.columns


def collect_missing_days_results(raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Collect missing business days results for all tickers.

    Args:
        raw_df: DataFrame with 'ticker' and 'date' columns.

    Returns:
        List of dictionaries with ticker and missing business days.
    """
    missing_days_top: list[dict[str, Any]] = []
    for ticker, grp in raw_df.groupby("ticker"):
        result = compute_missing_days_for_ticker(
            str(ticker),
            grp["date"],  # type: ignore[arg-type]
        )
        if result is not None:
            missing_days_top.append(result)
    return missing_days_top


def compute_missing_business_days(raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute missing business days per ticker.

    Args:
        raw_df: DataFrame with 'ticker' and 'date' columns.

    Returns:
        List of dictionaries with ticker and missing business days.
    """
    if not has_required_columns_for_missing_days(raw_df):
        return []

    try:
        missing_days_top = collect_missing_days_results(raw_df)
        missing_days_top = sorted(
            missing_days_top, key=lambda x: x["missing_business_days"], reverse=True
        )[:TOP_N_TICKERS_REPORT]
        return missing_days_top
    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Failed to compute missing business days: {e}")
        return []


def compute_quality_metrics(raw_df: pd.DataFrame) -> dict[str, Any]:
    """Compute a concise set of data quality metrics for JSON reporting.

    Args:
        raw_df: DataFrame to analyze.

    Returns:
        Dictionary with quality metrics.
    """
    if raw_df.empty:
        return compute_empty_quality_metrics()

    basic_metrics = compute_basic_quality_metrics(raw_df)
    non_mono_tickers = compute_monotonicity_violations(raw_df)
    missing_days_top = compute_missing_business_days(raw_df)

    return {
        **basic_metrics,
        "non_monotonic_ticker_dates": non_mono_tickers,
        "top_missing_business_days": missing_days_top,
    }


def save_quality_report(metrics: dict[str, Any]) -> None:
    """Persist data quality metrics to JSON under results/eval.

    Args:
        metrics: Dictionary containing quality metrics to save.

    Raises:
        OSError: If file cannot be written (logged as warning, not raised).
    """
    try:
        DATA_QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Quality report saved: {DATA_QUALITY_REPORT_FILE}")
    except OSError as e:
        logger.warning(f"Failed to write data quality report: {e}")


def calculate_observations_per_ticker(raw_df: pd.DataFrame) -> pd.Series:
    """Calculate number of observations per ticker.

    Args:
        raw_df: Raw dataset DataFrame.

    Returns:
        Series with observations count per ticker.

    Raises:
        KeyError: If 'ticker' column is missing.
        ValueError: If DataFrame is empty.
    """
    if raw_df.empty:
        msg = "DataFrame is empty"
        raise ValueError(msg)

    if "ticker" not in raw_df.columns:
        msg = "Missing required column: 'ticker'"
        raise KeyError(msg)

    return raw_df.groupby("ticker").size()  # type: ignore[return-value]


def validate_filtering_inputs(raw_df: pd.DataFrame, valid_tickers: list[str]) -> None:
    """Validate inputs for dataset filtering.

    Args:
        raw_df: Raw dataset DataFrame.
        valid_tickers: List of valid ticker symbols.

    Raises:
        KeyError: If 'ticker' column is missing.
        ValueError: If valid_tickers is empty or raw_df is empty.
    """
    if raw_df.empty:
        msg = "Raw dataset is empty"
        raise ValueError(msg)

    if "ticker" not in raw_df.columns:
        msg = "Missing required column: 'ticker'"
        raise KeyError(msg)

    if not valid_tickers:
        msg = "No valid tickers to filter. Dataset would be empty."
        raise ValueError(msg)


def write_filtered_dataset(filtered_df: pd.DataFrame) -> None:
    """Write filtered dataset to CSV file.

    Args:
        filtered_df: Filtered DataFrame to save.

    Raises:
        OSError: If file cannot be written.
    """
    try:
        DATASET_FILTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(DATASET_FILTERED_FILE, index=False)
    except OSError as e:
        msg = f"Failed to save filtered dataset to {DATASET_FILTERED_FILE}: {e}"
        raise OSError(msg) from e


def save_filtered_dataset(raw_df: pd.DataFrame, valid_tickers: list[str]) -> pd.DataFrame:
    """Filter dataset and save to CSV.

    Args:
        raw_df: Raw dataset DataFrame.
        valid_tickers: List of valid ticker symbols.

    Returns:
        Filtered DataFrame.

    Raises:
        KeyError: If 'ticker' column is missing.
        ValueError: If valid_tickers is empty or raw_df is empty.
        OSError: If file cannot be written.
    """
    validate_filtering_inputs(raw_df, valid_tickers)

    # Use bracket-based boolean indexing (not .loc) to ease testing with mocks
    filtered_df: pd.DataFrame = raw_df[raw_df["ticker"].isin(valid_tickers)].reset_index(drop=True)  # type: ignore[assignment]

    if filtered_df.empty:
        logger.warning("Filtered dataset is empty")

    write_filtered_dataset(filtered_df)

    logger.info("\nAfter filtering:")
    num_tickers = int(filtered_df["ticker"].nunique())  # type: ignore[attr-defined]
    logger.info(f"  Number of tickers: {num_tickers}")
    logger.info(f"  Total observations: {len(filtered_df):,}")
    logger.info(f"  Saved to: {DATASET_FILTERED_FILE}")

    return filtered_df
