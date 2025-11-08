"""Main script for running ablation studies."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.random_forest.ablation.ablation_sigma2 import run_ablation_study
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run ablation study to test causal effect of removing sigma2_garch."""
    try:
        results = run_ablation_study()
        logger.info("Ablation study completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ablation study failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

