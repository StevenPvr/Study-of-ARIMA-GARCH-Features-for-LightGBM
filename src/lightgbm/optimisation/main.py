"""Main script for LightGBM hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.lightgbm.optimisation.execution import run_optimization
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run LightGBM hyperparameter optimization for both datasets."""
    try:
        run_optimization()
        logger.info("Optimization completed successfully")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
