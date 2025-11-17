"""CLI for EGARCH hyperparameter optimization (Optuna + walk-forward CV).

This CLI focuses on a single, explicit pipeline:
- Optimize EGARCH(o,p) hyperparameters via Optuna
- Use walk-forward cross-validation on TRAIN only
- Minimize out-of-sample QLIKE
- Save best parameters to JSON

All preparatory/batch estimation steps are intentionally removed to keep the
process simple and explicit as requested.
"""

from __future__ import annotations

from pathlib import Path
import sys


_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import GARCH_DATASET_FILE, GARCH_OPTIMIZATION_RESULTS_FILE
from src.garch.garch_params.data import load_and_prepare_data
from src.garch.garch_params.optimization import (
    optimize_egarch_hyperparameters,
    save_optimization_results,
)
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Optimize EGARCH hyperparameters via Optuna with walk-forward CV.

    Methodology (intentionally minimal):
    - Uses TRAIN data only (no test data leakage)
    - Walk-forward cross-validation with configured burn-in
    - Minimizes QLIKE out-of-sample
    - Searches over: orders (o,p), distribution, refit_freq, window_type, window_size

    Results are saved to results/garch/optimization/ as JSON.
    """
    logger.info("=" * 60)
    logger.info("EGARCH HYPERPARAMETER OPTIMIZATION")
    logger.info("Method: Optuna + Walk-forward CV on TRAIN")
    logger.info("Objective: Minimize QLIKE out-of-sample")
    logger.info("=" * 60)

    # Hyperparameter optimization only (no batch estimation)
    logger.info("-" * 60)
    logger.info("STAGE 2: Hyperparameter Optimization")
    logger.info("-" * 60)
    resid_train, resid_test = load_and_prepare_data()

    logger.info("Training data: %d observations", resid_train.size)
    logger.info("Test data: %d observations (not used in optimization)", resid_test.size)

    # Optimize hyperparameters
    results = optimize_egarch_hyperparameters(resid_train)

    # Add metadata
    results["source"] = str(GARCH_DATASET_FILE)
    results["methodology"] = "Optuna optimization with walk-forward CV on TRAIN"
    results["n_obs_train"] = int(resid_train.size)
    results["n_obs_test"] = int(resid_test.size)

    # Save results
    save_optimization_results(results, GARCH_OPTIMIZATION_RESULTS_FILE)

    logger.info("=" * 60)
    logger.info("Optimization completed successfully")
    logger.info("Best hyperparameters saved to: %s", GARCH_OPTIMIZATION_RESULTS_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
