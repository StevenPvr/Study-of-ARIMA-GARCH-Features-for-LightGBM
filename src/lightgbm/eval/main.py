"""Main script for LightGBM model evaluation."""

from __future__ import annotations

import gc
from pathlib import Path
import sys

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.lightgbm.eval.eval import run_evaluation
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run LightGBM model evaluation on test set."""
    logger.info("Starting LightGBM evaluation (test set)")

    try:
        results = run_evaluation()

        logger.info("\n✓ Evaluation completed successfully")
        logger.info("Results saved and SHAP plots generated")

        # Final cleanup before exit
        del results
        gc.collect()

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        # Cleanup even on error
        gc.collect()
        raise


if __name__ == "__main__":
    main()
