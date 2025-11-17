"""CLI entry point for technical indicators calculation and dataset preparation."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd

from src.constants import (
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
    LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
)
from src.lightgbm.data_preparation.dataset_variants import (
    create_dataset_insights_only_from_file,
    ensure_technical_only_no_target_lags_dataset,
    ensure_technical_plus_insights_no_target_lags_dataset,
)
from src.lightgbm.data_preparation.utils import prepare_datasets
from src.lightgbm.data_preparation.visualization import plot_log_volatility_distribution
from src.utils import get_logger

logger = get_logger(__name__)


def _load_variant_datasets() -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """Load all variant datasets created from the complete dataset.

    Returns:
        Tuple of (
            sigma_plus_base,
            log_volatility_only,
            insights_only,
            technical_only_no_target_lags,
            technical_plus_insights_no_target_lags,
        ).
    """
    sigma_path = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE.with_suffix(".parquet")
    log_volatility_path = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE.with_suffix(".parquet")
    insights_only_path = LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE.with_suffix(".parquet")
    tech_only_path = LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE.with_suffix(".parquet")
    tech_plus_insights_path = (
        LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE.with_suffix(".parquet")
    )

    df_sigma_plus_base = pd.read_parquet(sigma_path) if sigma_path.exists() else None
    df_log_volatility_only = (
        pd.read_parquet(log_volatility_path) if log_volatility_path.exists() else None
    )

    # Create insights-only dataset from sigma-plus-base if needed
    if sigma_path.exists() and not insights_only_path.exists():
        logger.info("Creating insights-only dataset from sigma-plus-base")
        create_dataset_insights_only_from_file()

    df_insights_only = pd.read_parquet(insights_only_path) if insights_only_path.exists() else None

    # Ensure and load the two dedicated technical datasets (no target lags)
    tech_only_path = ensure_technical_only_no_target_lags_dataset()
    tech_plus_insights_path = ensure_technical_plus_insights_no_target_lags_dataset()

    df_tech_only = pd.read_parquet(tech_only_path) if tech_only_path.exists() else None
    df_tech_plus_insights = (
        pd.read_parquet(tech_plus_insights_path) if tech_plus_insights_path.exists() else None
    )

    return (
        df_sigma_plus_base,
        df_log_volatility_only,
        df_insights_only,
        df_tech_only,
        df_tech_plus_insights,
    )


def _log_dataset_info(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    df_sigma_plus_base: pd.DataFrame | None,
    df_log_volatility_only: pd.DataFrame | None,
    df_insights_only: pd.DataFrame | None,
    df_tech_only: pd.DataFrame | None,
    df_tech_plus_insights: pd.DataFrame | None,
) -> None:
    """Log information about created datasets."""
    logger.info("Complete dataset: %d rows, %d columns", len(df_complete), len(df_complete.columns))
    logger.info(
        "Dataset without insights: %d rows, %d columns",
        len(df_without_insights),
        len(df_without_insights.columns),
    )

    if df_sigma_plus_base is not None:
        logger.info(
            "Sigma-plus-base dataset: %d rows, %d columns",
            len(df_sigma_plus_base),
            len(df_sigma_plus_base.columns),
        )

    if df_log_volatility_only is not None:
        logger.info(
            "Log volatility only dataset: %d rows, %d columns",
            len(df_log_volatility_only),
            len(df_log_volatility_only.columns),
        )

    if df_insights_only is not None:
        logger.info(
            "Insights-only dataset: %d rows, %d columns",
            len(df_insights_only),
            len(df_insights_only.columns),
        )

    if df_tech_only is not None:
        logger.info(
            "Technical-only (no target lags, no insights): %d rows, %d columns",
            len(df_tech_only),
            len(df_tech_only.columns),
        )

    if df_tech_plus_insights is not None:
        logger.info(
            "Technical + insights (no target lags): %d rows, %d columns",
            len(df_tech_plus_insights),
            len(df_tech_plus_insights.columns),
        )


def _verify_dataset_consistency(
    df_complete: pd.DataFrame,
    df_without_insights: pd.DataFrame,
    df_sigma_plus_base: pd.DataFrame | None,
    df_log_volatility_only: pd.DataFrame | None,
    df_insights_only: pd.DataFrame | None,
    df_tech_only: pd.DataFrame | None,
    df_tech_plus_insights: pd.DataFrame | None,
) -> None:
    """Verify that all datasets have the same number of rows."""
    dataset_sizes = {
        "complete": len(df_complete),
        "without_insights": len(df_without_insights),
    }

    if df_sigma_plus_base is not None:
        dataset_sizes["sigma_plus_base"] = len(df_sigma_plus_base)
    if df_log_volatility_only is not None:
        dataset_sizes["log_volatility_only"] = len(df_log_volatility_only)
    if df_insights_only is not None:
        dataset_sizes["insights_only"] = len(df_insights_only)
    if df_tech_only is not None:
        dataset_sizes["technical_only_no_target_lags"] = len(df_tech_only)
    if df_tech_plus_insights is not None:
        dataset_sizes["technical_plus_insights_no_target_lags"] = len(df_tech_plus_insights)

    unique_sizes = set(dataset_sizes.values())
    if len(unique_sizes) == 1:
        logger.info("All datasets have the same number of rows: %d", unique_sizes.pop())
    else:
        logger.warning(
            "Datasets have different numbers of rows: %s. "
            "This should not happen - all datasets should use the same observations.",
            dataset_sizes,
        )


def main() -> None:
    """Main CLI function to prepare LightGBM datasets."""
    try:
        logger.info("Starting LightGBM data preparation")

        df_complete, df_without_insights = prepare_datasets()
        (
            df_sigma_plus_base,
            df_log_volatility_only,
            df_insights_only,
            df_tech_only,
            df_tech_plus_insights,
        ) = _load_variant_datasets()

        logger.info("Data preparation completed successfully")
        _log_dataset_info(
            df_complete,
            df_without_insights,
            df_sigma_plus_base,
            df_log_volatility_only,
            df_insights_only,
            df_tech_only,
            df_tech_plus_insights,
        )
        _verify_dataset_consistency(
            df_complete,
            df_without_insights,
            df_sigma_plus_base,
            df_log_volatility_only,
            df_insights_only,
            df_tech_only,
            df_tech_plus_insights,
        )

        # Generate data preparation plots
        logger.info("Generating data preparation visualization plots")
        plot_log_volatility_distribution(df_complete)

    except Exception as e:
        logger.error("Failed to prepare datasets: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
