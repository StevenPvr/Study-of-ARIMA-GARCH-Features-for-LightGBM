"""Data cleaning functions for S&P 500 dataset.

Enhancements:
- Basic integrity fixes (drop duplicates, enforce volume > 0)
- JSON data quality reporting in results/eval
"""

from __future__ import annotations

from src.constants import DATASET_FILTERED_FILE
from src.data_cleaning.utils import (
    analyze_general_statistics,
    analyze_missing_values,
    analyze_outliers,
    analyze_ticker_distribution,
    apply_basic_integrity_fixes,
    compute_quality_metrics,
    load_dataset,
    report_least_observations,
    report_low_volume_tickers,
    save_quality_report,
)
from src.utils import get_logger

logger = get_logger(__name__)


def data_quality_analysis() -> None:
    """Analyze data quality and display summary statistics.

    Analyzes the raw dataset for missing values, outliers, and distribution
    statistics per ticker. Logs comprehensive quality metrics.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        KeyError: If required columns are missing.
        ValueError: If dataset is empty or invalid.
    """
    logger.info("Starting data quality analysis")

    try:
        raw_df = load_dataset()

        logger.info("=" * 60)
        logger.info("DATA QUALITY ANALYSIS")
        logger.info("=" * 60)

        analyze_general_statistics(raw_df)
        analyze_missing_values(raw_df)
        analyze_outliers(raw_df)
        obs_per_ticker = analyze_ticker_distribution(raw_df)
        report_least_observations(obs_per_ticker)
        report_low_volume_tickers(raw_df)

        # Build and persist a compact JSON quality report
        metrics = compute_quality_metrics(raw_df)
        save_quality_report(metrics)

        logger.info("Data quality analysis completed")
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Data quality analysis failed: {e}")
        raise


def filter_by_membership() -> None:
    """Apply basic integrity fixes to dataset and save output.

    Applies basic integrity fixes (duplicate removal, volume > 0) and saves
    the cleaned dataset.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        KeyError: If required columns are missing in dataset.
        OSError: If filtered dataset cannot be saved.
    """
    logger.info("Starting dataset filtering and integrity fixes")

    raw_df = load_dataset()

    # Apply basic integrity fixes and persist
    fixed_df, counters = apply_basic_integrity_fixes(raw_df)
    dup_removed = counters.get("duplicates_removed", 0)
    vol_replaced = counters.get("nonpositive_volume_replaced", 0)
    incomplete_nullified = counters.get("incomplete_tickers_nullified", 0)
    if dup_removed > 0 or vol_replaced > 0 or incomplete_nullified > 0:
        logger.info(
            f"Integrity fixes: duplicates_removed={dup_removed}, "
            f"nonpositive_volume_replaced={vol_replaced}, "
            f"incomplete_tickers_nullified={incomplete_nullified}"
        )

    DATASET_FILTERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    fixed_df.to_csv(DATASET_FILTERED_FILE, index=False)
    logger.info(f"Saved filtered dataset to: {DATASET_FILTERED_FILE}")
