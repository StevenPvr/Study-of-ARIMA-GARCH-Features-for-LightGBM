"""Main entry point for data leakage detection tests.

This script runs the target shuffle test to detect potential data leakage
in LightGBM models.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse

from src.constants import (
    DEFAULT_RANDOM_STATE,
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE,
)
from src.lightgbm.data_leakage_checkup.leakage_test import run_target_shuffle_test
from src.lightgbm.shared_utils import resolve_dataset_path
from src.utils import ensure_output_dir, get_logger, save_json_pretty

logger = get_logger(__name__)


def _save_test_results(
    results: dict,
    output_path: Path = LIGHTGBM_LEAKAGE_TEST_RESULTS_FILE,
) -> None:
    """Save leakage test results to JSON.

    Args:
        results: Test results dictionary.
        output_path: Path to save results.
    """
    ensure_output_dir(output_path)
    save_json_pretty(results, output_path)
    logger.info(f"Leakage test results saved to {output_path}")


def main(
    dataset_path: Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Run data leakage detection tests.

    Args:
        dataset_path: Path to dataset file. Defaults to complete dataset.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with test results.
    """
    dataset_path = resolve_dataset_path(dataset_path, LIGHTGBM_DATASET_COMPLETE_FILE)

    logger.info(f"Using dataset: {dataset_path}")

    # Run target shuffle test
    results = run_target_shuffle_test(
        dataset_path=dataset_path,
        random_state=random_state,
    )

    # Save results
    _save_test_results(results)

    # Exit with error code if leakage detected
    if results["leakage_detected"]:
        logger.error("Leakage test FAILED: Potential data leakage detected!")
        sys.exit(1)
    else:
        logger.info("Leakage test PASSED: No data leakage detected.")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data leakage detection tests for LightGBM models"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to dataset file (default: complete dataset)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE})",
    )

    args = parser.parse_args()

    try:
        main(dataset_path=args.dataset, random_state=args.random_state)
    except Exception as e:
        logger.error(f"Leakage test failed: {e}")
        raise
