"""CLI for EGARCH parameter estimation and optimization stages.

Provides two explicit stages that align with the documented methodology:
- ``estimation``: batch EGARCH(1,1) MLE for each innovation distribution.
- ``optimization``: Optuna-based hyperparameter search warm-started from the
  estimation diagnostics to guarantee traceability.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Sequence

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_OPTIMIZATION_DISTRIBUTIONS,
    GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS,
    GARCH_OPTIMIZATION_RESULTS_FILE,
    GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES,
    GARCH_OPTIMIZATION_WINDOW_TYPES,
)
from src.garch.garch_params.data import load_and_prepare_data
from src.garch.garch_params.estimation import estimate_egarch_mle
from src.garch.garch_params.optimization import (
    optimize_egarch_hyperparameters,
    save_optimization_results,
)
from src.garch.training_garch.predictions_io import (
    load_estimation_results,
    save_estimation_results,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="EGARCH estimation and optimization pipeline",
    )
    parser.add_argument(
        "stage",
        choices=("estimation", "optimization"),
        nargs="?",
        default="optimization",
        help="Pipeline stage to execute",
    )
    return parser.parse_args(argv)


def _count_model_parameters(o: int, p: int, dist: str) -> int:
    """Count EGARCH(o,p) parameters including distributional terms."""

    n_params = 1 + 2 * o + p
    if dist == "student":
        n_params += 1
    elif dist == "skewt":
        n_params += 2
    return n_params


def _build_fit_entry(
    dist: str,
    params: dict[str, float],
    convergence: Any,
    n_obs: int,
    o: int,
    p: int,
) -> dict[str, Any]:
    """Build structured estimation payload for persistence."""

    n_params = _count_model_parameters(o, p, dist)
    loglik = float(convergence.final_loglik)
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + math.log(n_obs) * n_params

    entry: dict[str, Any] = {
        "params": params,
        "converged": bool(convergence.converged),
        "log_likelihood": loglik,
        "aic": aic,
        "bic": bic,
    }
    if convergence.n_iterations is not None:
        entry["iterations"] = int(convergence.n_iterations)
    if convergence.message:
        entry["convergence_message"] = convergence.message
    return entry


def _build_warm_trials(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive warm-start trials from estimation diagnostics."""

    baseline_refit = min(GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS)
    warm_trials: list[dict[str, Any]] = []
    for dist in doc.get("distributions_tested", []):
        warm_trials.append(
            {
                "o": 1,
                "p": 1,
                "distribution": dist,
                "refit_freq": baseline_refit,
                "window_type": "expanding",
            }
        )
        entry = doc.get(f"egarch_{dist}", {})
        logger.info(
            "Warm-start %s: loglik=%.2f, aic=%.2f, converged=%s",
            dist,
            entry.get("log_likelihood", float("nan")),
            entry.get("aic", float("nan")),
            entry.get("converged", False),
        )
    return warm_trials


def _load_estimation_baselines() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Load estimation diagnostics and derive warm-start trials."""

    if not GARCH_ESTIMATION_FILE.exists():
        logger.warning(
            "Estimation file missing (%s); run stage 'estimation' for full traceability",
            GARCH_ESTIMATION_FILE,
        )
        return None, []

    try:
        doc = load_estimation_results(GARCH_ESTIMATION_FILE)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Failed to load estimation file %s: %s", GARCH_ESTIMATION_FILE, exc)
        return None, []

    warm_trials = _build_warm_trials(doc)
    if warm_trials:
        logger.info(
            "Derived %d warm-start trials from %s for Optuna initialization",
            len(warm_trials),
            GARCH_ESTIMATION_FILE,
        )
    return doc, warm_trials


def run_estimation_stage() -> None:
    """Run EGARCH(1,1) MLE for each innovation distribution and persist results."""

    logger.info("=" * 60)
    logger.info("STAGE 1: EGARCH BATCH ESTIMATION (MLE)")
    logger.info("Source dataset: %s", GARCH_DATASET_FILE)
    resid_train, resid_test = load_and_prepare_data()
    logger.info("Training observations: %d", resid_train.size)
    logger.info("Test observations (unused): %d", resid_test.size)

    fits: dict[str, dict[str, Any]] = {}
    n_obs = int(resid_train.size)
    for dist in GARCH_OPTIMIZATION_DISTRIBUTIONS:
        logger.info("Estimating EGARCH(1,1) under %s innovations", dist)
        params, convergence = estimate_egarch_mle(resid_train, dist=dist, o=1, p=1)
        fits[dist] = _build_fit_entry(dist, params, convergence, n_obs, 1, 1)

    save_estimation_results(fits, n_observations=n_obs, output_path=GARCH_ESTIMATION_FILE)
    logger.info(
        "Saved EGARCH estimation diagnostics to %s; these baselines feed the optimization stage",
        GARCH_ESTIMATION_FILE,
    )


def run_optimization_stage() -> None:
    """Optimize EGARCH hyperparameters with Optuna, warm-started by estimation baselines."""

    logger.info("=" * 60)
    logger.info("STAGE 2: EGARCH HYPERPARAMETER OPTIMIZATION")
    order_options = (1, 2)
    arch_options = (1, 2)
    base_combos = (
        len(order_options)
        * len(arch_options)
        * len(GARCH_OPTIMIZATION_DISTRIBUTIONS)
        * len(GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS)
    )
    window_breakdown: dict[str, int] = {}
    total_combos = 0
    for window_type in GARCH_OPTIMIZATION_WINDOW_TYPES:
        multiplier = 1
        if window_type == "rolling":
            multiplier = len(GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES)
        window_breakdown[window_type] = base_combos * multiplier
        total_combos += window_breakdown[window_type]
    logger.info(
        "Hyperparameter grid: o∈%s, p∈%s, distributions=%s, refit_freq=%s, window_types=%s",
        order_options,
        arch_options,
        GARCH_OPTIMIZATION_DISTRIBUTIONS,
        GARCH_OPTIMIZATION_REFIT_FREQ_OPTIONS,
        GARCH_OPTIMIZATION_WINDOW_TYPES,
    )
    logger.info(
        "Rolling window sizes (applies when window_type='rolling'): %s",
        GARCH_OPTIMIZATION_ROLLING_WINDOW_SIZES,
    )
    logger.info(
        "Grid size: %d combinations (%s)",
        total_combos,
        ", ".join(f"{name}={count}" for name, count in window_breakdown.items()),
    )
    estimation_doc, warm_trials = _load_estimation_baselines()
    resid_train, resid_test = load_and_prepare_data()
    logger.info("Training data: %d observations", resid_train.size)
    logger.info("Test data: %d observations (not used in optimization)", resid_test.size)
    if estimation_doc is None:
        logger.warning(
            "Proceeding without estimation baselines; optimization will run but lacks documented initial state"
        )

    n_trials = None
    if warm_trials:
        logger.info(
            "Warm-starting optimization with %d queued trials derived from estimation baselines",
            len(warm_trials),
        )

    results = optimize_egarch_hyperparameters(
        resid_train,
        initial_trials=warm_trials,
        n_trials=n_trials,
    )

    results["source"] = str(GARCH_DATASET_FILE)
    results["methodology"] = (
        "Optuna optimization with walk-forward CV on TRAIN, warm-started from MLE baselines"
    )
    results["n_obs_train"] = int(resid_train.size)
    results["n_obs_test"] = int(resid_test.size)
    if estimation_doc:
        results["estimation_generated_at"] = estimation_doc.get("generated_at")

    save_optimization_results(results, GARCH_OPTIMIZATION_RESULTS_FILE)
    logger.info("Optimization completed; results saved to: %s", GARCH_OPTIMIZATION_RESULTS_FILE)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point selecting the requested stage."""

    args = _parse_args(argv)
    if args.stage == "estimation":
        run_estimation_stage()
        return
    run_optimization_stage()


if __name__ == "__main__":
    main()
