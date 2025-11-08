"""CLI to train EGARCH(1,1) on ARIMA residuals dataset."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import GARCH_DATASET_FILE
from src.garch.structure_garch.utils import load_garch_dataset
from src.garch.training_garch.training import train_egarch_from_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Load dataset and train GARCH, saving artifacts and outputs."""
    logger.info("=" * 60)
    logger.info("GARCH TRAINING")
    logger.info("=" * 60)

    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except FileNotFoundError as ex:
        logger.error("Failed to load GARCH dataset: %s", ex)
        raise
    except ValueError as ex:
        logger.error("Invalid GARCH dataset: %s", ex)
        raise

    try:
        info = train_egarch_from_dataset(df)
    except (FileNotFoundError, ValueError) as ex:
        logger.error("Failed to train GARCH model: %s", ex)
        raise

    logger.info("Best distribution: %s", info["dist"])
    logger.info("Params: %s", info["params"])
    logger.info("Outputs shape: %s", info["outputs_shape"])
    logger.info("Standardized residuals diagnostics: %s", info["std_resid_diagnostics"])


if __name__ == "__main__":
    main()
