"""CLI entry point for training_arima module."""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.optimisation_arima.optimisation_arima import load_train_data
from src.arima.training_arima.training_arima import save_trained_model, train_best_model
from src.constants import WEIGHTED_LOG_RETURNS_SPLIT_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function for ARIMA training."""
    logger.info("=" * 60)
    logger.info("ARIMA TRAINING MODULE")
    logger.info("=" * 60)

    # Load train data using defaults from constants (no CLI required)
    logger.info("Loading train data...")
    train_series = load_train_data(
        csv_path=WEIGHTED_LOG_RETURNS_SPLIT_FILE,
        value_col="weighted_log_return",
        date_col=None,
    )

    # Train best AIC model
    logger.info("Training best AIC model...")
    fitted_model_aic, model_info_aic = train_best_model(train_series, prefer="aic")
    save_trained_model(fitted_model_aic, model_info_aic)

    logger.info(f"Trained model: {model_info_aic['params']}")
    logger.info(f"Model AIC: {fitted_model_aic.aic:.2f}")

    logger.info("=" * 60)
    logger.info("ARIMA training complete!")
    logger.info("Next step: Run evaluation_arima to evaluate the trained model")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
