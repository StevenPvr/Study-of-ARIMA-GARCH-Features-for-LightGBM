"""Main script for Random Forest model evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.random_forest.eval.eval import run_evaluation
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run Random Forest model evaluation on test set."""
    logger.info("Starting Random Forest evaluation (test set)")

    try:
        run_evaluation()

        logger.info("\n✓ Evaluation completed successfully")
        logger.info(f"Results saved and SHAP plots generated")

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
