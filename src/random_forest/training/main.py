"""Main script for Random Forest training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.random_forest.training.training import run_training
from src.utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Random Forest models using optimized hyperparameters on train split"
    )
    parser.add_argument(
        "--optimization-results",
        type=Path,
        default=None,
        help="Path to optimization results JSON file (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Run Random Forest training on train split only."""
    args = parse_args()

    logger.info("Starting Random Forest training (train split only)")

    try:
        if args.optimization_results:
            results = run_training(
                optimization_results_path=args.optimization_results,
            )
        else:
            results = run_training()

        logger.info("\n✓ Training completed successfully")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

