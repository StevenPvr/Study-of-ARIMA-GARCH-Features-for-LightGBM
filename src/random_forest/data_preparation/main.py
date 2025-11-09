"""CLI entry point for technical indicators calculation and dataset preparation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.random_forest.data_preparation.utils import (
    create_dataset_technical_indicators,
    prepare_datasets,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function to prepare Random Forest datasets."""
    try:
        logger.info("Starting Random Forest data preparation")
        df_complete, df_without_insights = prepare_datasets()

        logger.info("Creating technical indicators dataset")
        df_technical = create_dataset_technical_indicators(include_lags=True)

        logger.info("Data preparation completed successfully")
        logger.info(
            f"Complete dataset: {len(df_complete)} rows, {len(df_complete.columns)} columns"
        )
        logger.info(
            f"Dataset without insights: {len(df_without_insights)} rows, "
            f"{len(df_without_insights.columns)} columns"
        )
        logger.info(
            f"Technical indicators dataset: {len(df_technical)} rows, "
            f"{len(df_technical.columns)} columns"
        )

    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
