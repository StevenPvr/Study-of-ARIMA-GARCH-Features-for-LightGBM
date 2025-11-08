"""CLI entry point for Spearman correlation calculation and visualization."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.random_forest.correlation.correlation import compute_correlations
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function to compute Spearman correlations and create visualizations."""
    try:
        logger.info("Starting Spearman correlation calculation and visualization")
        corr_complete, corr_without = compute_correlations()

        logger.info("Correlation calculation and visualization completed successfully")
        logger.info(f"Complete dataset correlation matrix: {corr_complete.shape}")
        logger.info(f"Without insights correlation matrix: {corr_without.shape}")

    except Exception as e:
        logger.error(f"Failed to compute correlations: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
