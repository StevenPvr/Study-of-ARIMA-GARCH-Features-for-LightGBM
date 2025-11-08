"""CLI for running GARCH numerical tests (pre-EGARCH, post-SARIMA).

Runs all numerical tests on SARIMA residuals:
- Ljung-Box test on residuals
- Ljung-Box test on squared residuals
- Engle ARCH-LM test
- McLeod-Li test

Saves results to JSON file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_NUMERICAL_TESTS_FILE,
    GARCH_STRUCTURE_DIR,
    LJUNGBOX_SIGNIFICANCE_LEVEL,
)
from src.garch.garch_numerical_test.garch_numerical import run_all_tests
from src.garch.structure_garch.detection import load_garch_dataset, prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


def _prepare_output(results: dict[str, Any], n_residuals: int) -> dict[str, Any]:
    """Prepare output dictionary for test results.

    Args:
        results: Dictionary containing test results.
        n_residuals: Number of residuals tested.

    Returns:
        Formatted output dictionary.
    """
    return {
        "source": str(GARCH_DATASET_FILE),
        "n_residuals": n_residuals,
        "tests": {
            "ljung_box_residuals": {
                "name": "Ljung-Box Test sur les résidus",
                "result": results["ljung_box_residuals"],
            },
            "ljung_box_squared": {
                "name": "Ljung-Box Test sur les résidus au carré",
                "result": results["ljung_box_squared"],
            },
            "engle_arch_lm": {
                "name": "Engle ARCH LM Test",
                "result": results["engle_arch_lm"],
            },
            "mcleod_li": {
                "name": "McLeod-Li Test",
                "result": results["mcleod_li"],
            },
        },
    }


def _log_test_summary(results: dict[str, Any]) -> None:
    """Log summary of test results.

    Args:
        results: Dictionary containing test results.
    """
    logger.info("Test Results Summary:")
    logger.info("  Ljung-Box (residuals): reject=%s", results["ljung_box_residuals"]["reject"])
    logger.info("  Ljung-Box (squared): reject=%s", results["ljung_box_squared"]["reject"])
    logger.info("  Engle ARCH-LM: reject=%s", results["engle_arch_lm"]["reject"])
    logger.info("  McLeod-Li: reject=%s", results["mcleod_li"]["reject"])


def main() -> None:
    """Run all numerical tests on SARIMA residuals and save results."""
    logger.info("=" * 60)
    logger.info("GARCH NUMERICAL TESTS (Pre-EGARCH, Post-SARIMA)")
    logger.info("=" * 60)

    # Load dataset and extract residuals
    df = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(df, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]

    logger.info("Running numerical tests on %d test residuals", resid_test.size)

    # Run all tests
    results = run_all_tests(
        resid_test,
        ljung_box_lags=GARCH_LJUNG_BOX_LAGS_DEFAULT,
        arch_lm_lags=GARCH_LM_LAGS_DEFAULT,
        alpha=LJUNGBOX_SIGNIFICANCE_LEVEL,
    )

    # Prepare and save output
    output = _prepare_output(results, int(resid_test.size))
    GARCH_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    with GARCH_NUMERICAL_TESTS_FILE.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved numerical test results: %s", GARCH_NUMERICAL_TESTS_FILE)
    _log_test_summary(results)


if __name__ == "__main__":
    main()
