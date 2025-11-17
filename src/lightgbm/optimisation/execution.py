"""Parallel execution utilities for LightGBM optimization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.constants import LIGHTGBM_OPTIMIZATION_MAX_WORKERS
from src.lightgbm.optimisation.data_loading import load_dataset
from src.lightgbm.optimisation.optimization import optimize_lightgbm
from src.utils import get_logger

logger = get_logger(__name__)


def _run_single_optimization(
    dataset_path: Path,
    study_name: str,
    n_trials: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run optimization for a single dataset.

    Args:
        dataset_path: Path to the dataset CSV file.
        study_name: Name for the Optuna study.
        n_trials: Number of Optuna trials.

    Returns:
        Tuple of (dataset_name, results_dict, best_params_dict).
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"OPTIMIZATION: {dataset_path.name}")
    logger.info(f"{'=' * 70}")
    X, y = load_dataset(dataset_path)
    results, best_params = optimize_lightgbm(
        X,
        y,
        study_name=study_name,
        n_trials=n_trials,
    )
    logger.info(f"✓ Completed optimization: {study_name}")
    return study_name, results, best_params


def _run_parallel_optimizations(
    tasks: list[tuple[Path, str, int]],
    max_workers: int = LIGHTGBM_OPTIMIZATION_MAX_WORKERS,
) -> dict[str, dict[str, Any]]:
    """Run optimization tasks in parallel.

    Args:
        tasks: List of (dataset_path, study_name, n_trials) tuples.
        max_workers: Maximum number of parallel workers.

    Returns:
        Dictionary mapping study names to results.

    Raises:
        Exception: If any optimization task fails.
    """
    results_dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_run_single_optimization, path, name, trials): name
            for path, name, trials in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                study_name, results, _ = future.result()
                results_dict[study_name] = results
                logger.info(f"✓ Optimization completed: {study_name}")
            except Exception as e:
                logger.error(f"✗ Optimization failed for {name}: {e}")
                raise

    return results_dict


def _prepare_optimization_tasks(
    n_trials: int,
) -> list[tuple[Path, str, int]]:
    """Prepare optimization tasks for all datasets.

    Args:
        n_trials: Number of trials per dataset.

    Returns:
        List of (dataset_path, study_name, n_trials) tuples.
    """
    from src.constants import (
        LIGHTGBM_DATASET_COMPLETE_FILE,
        LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE,
        LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
        LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
        LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
        LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
        LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    )
    from src.lightgbm.data_preparation.dataset_variants import (
        create_dataset_insights_only_from_file,
        ensure_technical_only_no_target_lags_dataset,
        ensure_technical_plus_insights_no_target_lags_dataset,
    )
    from src.lightgbm.data_preparation.utils import (
        ensure_log_volatility_only_dataset,
        ensure_sigma_plus_base_dataset,
    )

    # Ensure sigma-plus-base and log-volatility-only datasets exist
    ensure_sigma_plus_base_dataset(include_lags=True)
    ensure_log_volatility_only_dataset(include_lags=True)

    # Ensure insights-only dataset exists
    insights_only_path = LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE.with_suffix(".parquet")
    if not insights_only_path.exists():
        logger.info("Creating insights-only dataset from sigma-plus-base")
        create_dataset_insights_only_from_file()

    # Ensure technical-only (no target lags, no insights) and
    # technical-plus-insights (no target lags) datasets exist
    ensure_technical_only_no_target_lags_dataset()
    ensure_technical_plus_insights_no_target_lags_dataset()

    tasks = [
        (LIGHTGBM_DATASET_COMPLETE_FILE, "lightgbm_complete", n_trials),
        (LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE, "lightgbm_without_insights", n_trials),
        (LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE, "lightgbm_sigma_plus_base", n_trials),
        (LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE, "lightgbm_log_volatility_only", n_trials),
        (LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE, "lightgbm_insights_only", n_trials),
        (
            LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
            "lightgbm_technical_only",
            n_trials,
        ),
        (
            LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
            "lightgbm_technical_plus_insights",
            n_trials,
        ),
    ]
    return tasks


def _log_optimization_start(n_trials: int, sample_fraction: float) -> None:
    """Log optimization start information.

    Args:
        n_trials: Number of trials per dataset.
        sample_fraction: Fraction of train data to use.
    """
    logger.info("=" * 70)
    logger.info("LightGBM Hyperparameter Optimization (Parallel)")
    logger.info("=" * 70)
    logger.info(f"Running {n_trials} trials per dataset")
    logger.info(f"Using {sample_fraction*100:.0f}% of train data for optimization")
    logger.info(f"Optimizing {LIGHTGBM_OPTIMIZATION_MAX_WORKERS} models in parallel")


def _extract_optimization_results(
    results_dict: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract results from results dictionary.

    Args:
        results_dict: Dictionary mapping study names to results.

    Returns:
        Tuple of (results_complete, results_without, results_sigma_plus_base,
        results_log_volatility, results_insights_only).
    """
    results_complete = results_dict["lightgbm_complete"]
    results_without = results_dict["lightgbm_without_insights"]
    results_sigma_plus_base = results_dict.get("lightgbm_sigma_plus_base")
    results_log_volatility = results_dict.get("lightgbm_log_volatility_only")
    results_insights_only = results_dict.get("lightgbm_insights_only")
    results_technical_only = results_dict.get("lightgbm_technical_only")
    results_technical_plus_insights = results_dict.get("lightgbm_technical_plus_insights")
    return (
        results_complete,
        results_without,
        results_sigma_plus_base,
        results_log_volatility,
        results_insights_only,
        results_technical_only,
        results_technical_plus_insights,
    )


def _save_and_log_results(
    results_complete: dict[str, Any],
    results_without: dict[str, Any],
    results_sigma_plus_base: dict[str, Any] | None,
    results_log_volatility: dict[str, Any] | None,
    results_insights_only: dict[str, Any] | None,
    results_technical_only: dict[str, Any] | None,
    results_technical_plus_insights: dict[str, Any] | None,
) -> None:
    """Save and log optimization results.

    Args:
        results_complete: Results for complete dataset.
        results_without: Results for dataset without insights.
        results_sigma_plus_base: Optional results for sigma-plus-base dataset.
        results_log_volatility: Optional results for log-volatility-only dataset.
        results_insights_only: Optional results for insights-only dataset.
    """
    from src.lightgbm.optimisation.results import (
        _log_optimization_summary,
        save_optimization_results,
    )

    save_optimization_results(
        results_complete,
        results_without,
        results_sigma_plus_base=results_sigma_plus_base,
        results_log_volatility_only=results_log_volatility,
        results_technical=None,
        results_technical_only=results_technical_only,
        results_technical_plus_insights=results_technical_plus_insights,
        results_insights_only=results_insights_only,
    )
    _log_optimization_summary(
        results_complete,
        results_without,
        results_sigma_plus_base,
        results_log_volatility,
        results_technical=None,
        results_technical_only=results_technical_only,
        results_technical_plus_insights=results_technical_plus_insights,
        results_insights_only=results_insights_only,
    )


def run_optimization(
    n_trials: int | None = None,
    sample_fraction: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run hyperparameter optimization for all datasets in parallel.

    Args:
        n_trials: Number of Optuna trials per dataset. Defaults to LIGHTGBM_OPTIMIZATION_N_TRIALS.
        sample_fraction: Fraction of train data to use.
        Defaults to LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION.

    Returns:
        Tuple of (results_complete, results_without_insights).
    """
    from src.constants import LIGHTGBM_OPTIMIZATION_N_TRIALS, LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION

    if n_trials is None:
        n_trials = LIGHTGBM_OPTIMIZATION_N_TRIALS
    if sample_fraction is None:
        sample_fraction = LIGHTGBM_OPTIMIZATION_SAMPLE_FRACTION

    _log_optimization_start(n_trials, sample_fraction)

    # Prepare and run optimizations
    tasks = _prepare_optimization_tasks(n_trials)
    results_dict = _run_parallel_optimizations(tasks, max_workers=LIGHTGBM_OPTIMIZATION_MAX_WORKERS)

    # Extract, save, log, and cleanup
    (
        results_complete,
        results_without,
        results_sigma_plus_base,
        results_log_volatility,
        results_insights_only,
        results_technical_only,
        results_technical_plus_insights,
    ) = _extract_optimization_results(results_dict)
    _save_and_log_results(
        results_complete,
        results_without,
        results_sigma_plus_base,
        results_log_volatility,
        results_insights_only,
        results_technical_only,
        results_technical_plus_insights,
    )

    return results_complete, results_without
