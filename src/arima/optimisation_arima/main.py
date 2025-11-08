"""CLI entry point for optimisation_arima module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.optimisation_arima.optimisation_arima import (
    load_train_test_data,
    optimize_sarima_models,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for SARIMA optimization."""
    logger.info("=" * 60)
    logger.info("SARIMA OPTIMIZATION MODULE")
    logger.info("=" * 60)

    # Load train/test data
    logger.info("Loading train/test data...")
    train_series, test_series = load_train_test_data()

    # Optimize SARIMA models
    logger.info("Starting SARIMA optimization...")
    results_df, best_aic, best_bic = optimize_sarima_models(train_series, test_series)

    logger.info(f"Best AIC model: {best_aic['params']}")
    logger.info(f"Best BIC model: {best_bic['params']}")

    logger.info("=" * 60)
    logger.info("SARIMA optimization complete!")
    logger.info("Next steps:")
    logger.info("  1. Run training_arima to train the best model")
    logger.info("  2. Run evaluation_arima to evaluate the trained model")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
