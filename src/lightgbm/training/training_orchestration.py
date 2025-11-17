"""Training orchestration for LightGBM models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.constants import (
    LIGHTGBM_DATASET_COMPLETE_FILE as LIGHTGBM_DATASET_COMPLETE,
    LIGHTGBM_DATASET_INSIGHTS_ONLY_FILE as LIGHTGBM_DATASET_INSIGHTS_ONLY,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE as LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE as LIGHTGBM_DATASET_SIGMA_PLUS_BASE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE as LIGHTGBM_DATASET_WITHOUT_INSIGHTS,
    LIGHTGBM_MODELS_DIR,
    LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    LIGHTGBM_TRAINING_RESULTS_FILE,
)
from src.lightgbm.training.data_loading import load_optimization_results
from src.lightgbm.training.model_training import _run_single_training
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def _create_task(
    dataset_path: Path, model_name: str, params: dict[str, Any]
) -> tuple[Path, str, dict[str, Any]]:
    """Create a single training task tuple.

    Args:
        dataset_path: Path to dataset.
        model_name: Name of the model.
        params: Model parameters.

    Returns:
        Task tuple (dataset_path, model_name, params).
    """
    return (dataset_path, model_name, params)


def _add_optional_task(
    tasks: list[tuple[Path, str, dict[str, Any]]],
    opt_results: dict[str, Any],
    key: str,
    model_name: str,
    default_path: Path,
    custom_path: Path | None,
) -> None:
    """Add optional training task if present in optimization results.

    Args:
        tasks: List of tasks to append to.
        opt_results: Optimization results dictionary.
        key: Key in opt_results to check.
        model_name: Name of the model.
        default_path: Default dataset path.
        custom_path: Custom dataset path (optional).
    """
    if key in opt_results:
        dataset_path = custom_path if custom_path is not None else default_path
        params = opt_results[key]["best_params"]
        tasks.append(_create_task(dataset_path, model_name, params))


def _prepare_training_tasks(
    opt_results: dict[str, Any],
    dataset_complete: Path | None = None,
    dataset_without_insights: Path | None = None,
    dataset_sigma_plus_base: Path | None = None,
    dataset_log_volatility_only: Path | None = None,
    dataset_insights_only: Path | None = None,
) -> list[tuple[Path, str, dict[str, Any]]]:
    """Prepare training tasks from optimization results.

    Args:
        opt_results: Optimization results dictionary.
        dataset_complete: Path to complete dataset (optional, uses default if None).
        dataset_without_insights: Path to dataset without insights (optional).
        dataset_sigma_plus_base: Path to sigma plus base dataset (optional).
        dataset_log_volatility_only: Path to log-volatility-only dataset (optional).
        dataset_insights_only: Path to insights-only dataset (optional).

    Returns:
        List of (dataset_path, model_name, params) tuples.
    """
    complete_path = dataset_complete if dataset_complete is not None else LIGHTGBM_DATASET_COMPLETE
    without_insights_path = (
        dataset_without_insights
        if dataset_without_insights is not None
        else LIGHTGBM_DATASET_WITHOUT_INSIGHTS
    )

    tasks: list[tuple[Path, str, dict[str, Any]]] = [
        _create_task(
            complete_path,
            "lightgbm_complete",
            opt_results["lightgbm_dataset_complete"]["best_params"],
        ),
        _create_task(
            without_insights_path,
            "lightgbm_without_insights",
            opt_results["lightgbm_dataset_without_insights"]["best_params"],
        ),
    ]

    # Add optional tasks
    _add_optional_task(
        tasks,
        opt_results,
        "lightgbm_dataset_sigma_plus_base",
        "lightgbm_sigma_plus_base",
        LIGHTGBM_DATASET_SIGMA_PLUS_BASE,
        dataset_sigma_plus_base,
    )

    _add_optional_task(
        tasks,
        opt_results,
        "lightgbm_dataset_log_volatility_only",
        "lightgbm_log_volatility_only",
        LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY,
        dataset_log_volatility_only,
    )

    _add_optional_task(
        tasks,
        opt_results,
        "lightgbm_dataset_insights_only",
        "lightgbm_insights_only",
        LIGHTGBM_DATASET_INSIGHTS_ONLY,
        dataset_insights_only,
    )

    # Technical-only (no target lags, no insights)
    from src.constants import (
        LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
        LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
    )

    _add_optional_task(
        tasks,
        opt_results,
        "lightgbm_dataset_technical_only",
        "lightgbm_technical_only",
        LIGHTGBM_DATASET_TECHNICAL_ONLY_NO_TARGET_LAGS_FILE,
        None,
    )

    # Technical + insights (no target lags)
    _add_optional_task(
        tasks,
        opt_results,
        "lightgbm_dataset_technical_plus_insights",
        "lightgbm_technical_plus_insights",
        LIGHTGBM_DATASET_TECHNICAL_PLUS_INSIGHTS_NO_TARGET_LAGS_FILE,
        None,
    )

    return tasks


def _run_parallel_training(
    tasks: list[tuple[Path, str, dict[str, Any]]],
    models_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run training tasks in parallel.

    Args:
        tasks: List of (dataset_path, model_name, params) tuples.
        models_dir: Directory to save models (optional, uses default if None).

    Returns:
        Dictionary mapping model names to results.

    Raises:
        Exception: If any training task fails.
    """
    models_path = models_dir if models_dir is not None else LIGHTGBM_MODELS_DIR
    results_dict = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_name = {
            executor.submit(_run_single_training, path, name, params, models_path): name
            for path, name, params in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                model_name, results = future.result()
                results_dict[model_name] = results
                logger.info(f"✓ Training completed: {model_name}")
            except Exception as e:
                logger.error(f"✗ Training failed for {name}: {e}")
                raise

    return results_dict


def _save_training_results(
    results_dict: dict[str, dict[str, Any]],
    results_file: Path | None = None,
) -> None:
    """Save training results to JSON file.

    Args:
        results_dict: Dictionary mapping model names to results.
        results_file: Path to results file (optional, uses default if None).
    """
    training_results = {
        "lightgbm_complete": results_dict["lightgbm_complete"],
        "lightgbm_without_insights": results_dict["lightgbm_without_insights"],
    }

    if "lightgbm_sigma_plus_base" in results_dict:
        training_results["lightgbm_sigma_plus_base"] = results_dict["lightgbm_sigma_plus_base"]

    if "lightgbm_log_volatility_only" in results_dict:
        training_results["lightgbm_log_volatility_only"] = results_dict[
            "lightgbm_log_volatility_only"
        ]

    if "lightgbm_insights_only" in results_dict:
        training_results["lightgbm_insights_only"] = results_dict["lightgbm_insights_only"]

    if "lightgbm_technical_only" in results_dict:
        training_results["lightgbm_technical_only"] = results_dict["lightgbm_technical_only"]
    if "lightgbm_technical_plus_insights" in results_dict:
        training_results["lightgbm_technical_plus_insights"] = results_dict[
            "lightgbm_technical_plus_insights"
        ]

    output_file = results_file if results_file is not None else LIGHTGBM_TRAINING_RESULTS_FILE
    save_json_pretty(training_results, output_file)

    logger.info(f"Training results saved to {output_file}")


def _log_training_summary(results_dict: dict[str, dict[str, Any]]) -> None:
    """Log training summary to console.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for name, res in results_dict.items():
        info = res["train_info"]
        logger.info(f"\n{name}:")
        logger.info(f"  Train RMSE: {info['train_rmse']:.6f} (log-volatility)")
        logger.info(f"  Train size: {info['train_size']}")
        logger.info(f"  N features: {info['n_features']}")
        logger.info(f"  Model saved: {res['model_path']}")

    logger.info("\n" + "=" * 70)


def run_training(
    optimization_results_path: Path = LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    dataset_complete: Path | None = None,
    dataset_without_insights: Path | None = None,
    models_dir: Path | None = None,
    results_file: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Train LightGBM models for both datasets in parallel on train split only.

    Args:
        optimization_results_path: Path to optimization results JSON.
        dataset_complete: Path to complete dataset (optional, uses default if None).
        dataset_without_insights: Path to dataset without insights (optional).
        models_dir: Directory to save models (optional, uses default if None).
        results_file: Path to results file (optional, uses default if None).

    Returns:
        Dictionary with training results for both models.

    Raises:
        FileNotFoundError: If optimization results or datasets are missing.
        ValueError: If optimization results are invalid.
    """
    logger.info("=" * 70)
    logger.info("LightGBM Training (Parallel) - Train Split Only")
    logger.info("=" * 70)

    opt_results = load_optimization_results(optimization_results_path)
    tasks = _prepare_training_tasks(
        opt_results,
        dataset_complete=dataset_complete,
        dataset_without_insights=dataset_without_insights,
    )
    results_dict = _run_parallel_training(tasks, models_dir=models_dir)
    _save_training_results(results_dict, results_file=results_file)
    _log_training_summary(results_dict)

    out = {
        "lightgbm_complete": results_dict["lightgbm_complete"],
        "lightgbm_without_insights": results_dict["lightgbm_without_insights"],
    }

    if "lightgbm_sigma_plus_base" in results_dict:
        out["lightgbm_sigma_plus_base"] = results_dict["lightgbm_sigma_plus_base"]

    if "lightgbm_log_volatility_only" in results_dict:
        out["lightgbm_log_volatility_only"] = results_dict["lightgbm_log_volatility_only"]

    if "lightgbm_insights_only" in results_dict:
        out["lightgbm_insights_only"] = results_dict["lightgbm_insights_only"]

    return out
