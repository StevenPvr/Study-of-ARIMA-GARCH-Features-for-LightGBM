"""LightGBM model evaluation on test set with SHAP analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from pathlib import Path
from typing import Any

from src.lightgbm.eval.data_loading import load_dataset, load_model
from src.lightgbm.eval.model_evaluation import evaluate_model
from src.lightgbm.eval.statistical_comparison import log_model_comparison, perform_statistical_tests
from src.lightgbm.model_utils import filter_existing_models, get_optional_model_configs
from src.path import (
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE,
    LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    LIGHTGBM_EVAL_RESULTS_FILE,
    LIGHTGBM_MODELS_DIR,
)
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)

# Constants for evaluation (aliases for backward compatibility)
LIGHTGBM_DATASET_COMPLETE = LIGHTGBM_DATASET_COMPLETE_FILE
LIGHTGBM_DATASET_SIGMA_PLUS_BASE = LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE
LIGHTGBM_DATASET_WITHOUT_INSIGHTS = LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE
LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY = LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE


def _get_required_evaluation_configs() -> list[tuple[Path, Path, str]]:
    """Get list of required model configurations for evaluation (always included)."""
    return [
        (
            LIGHTGBM_DATASET_COMPLETE_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_complete.joblib",
            "lightgbm_complete",
        ),
        (
            LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
            LIGHTGBM_MODELS_DIR / "lightgbm_without_insights.joblib",
            "lightgbm_without_insights",
        ),
    ]


def _prepare_evaluation_tasks() -> list[tuple[Path, Path, str]]:
    """Prepare evaluation tasks for both models.

    Returns:
        List of tuples (dataset_path, model_path, model_name).
    """
    tasks = _get_required_evaluation_configs()
    optional_configs = get_optional_model_configs()
    existing_optional = filter_existing_models(optional_configs)
    tasks.extend(existing_optional)

    return tasks


def _run_single_evaluation(
    dataset_path: Path,
    model_path: Path,
    model_name: str,
) -> tuple[str, dict[str, Any]]:
    """Evaluate a single model (helper for parallelization).

    Args:
        dataset_path: Path to the dataset CSV file.
        model_path: Path to the trained model joblib file.
        model_name: Name for the model.

    Returns:
        Tuple of (model_name, results_dict).
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"EVALUATION: {model_name}")
    logger.info(f"{'=' * 70}")

    # Load test data
    X_test, y_test = load_dataset(dataset_path, split="test")

    # Load model
    model = load_model(model_path)

    try:
        # Evaluate
        results = evaluate_model(model, X_test, y_test, model_name)
        logger.info(f"✓ Completed evaluation: {model_name}")
        return model_name, results
    finally:
        # Free model resources and memory
        # LightGBM models may keep internal threads, explicit deletion helps
        del model
        del X_test
        del y_test
        gc.collect()


def _run_parallel_evaluations(tasks: list[tuple[Path, Path, str]]) -> dict[str, dict[str, Any]]:
    """Run model evaluations in parallel.

    Args:
        tasks: List of evaluation tasks.

    Returns:
        Dictionary mapping model names to results.

    Raises:
        Exception: If any evaluation fails.
    """
    results_dict: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        future_to_name = {
            executor.submit(_run_single_evaluation, dataset, model, name): name
            for dataset, model, name in tasks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                model_name, results = future.result()
                results_dict[model_name] = results
                logger.info(f"✓ Evaluation completed: {model_name}")
            except Exception as e:
                logger.error(f"✗ Evaluation failed for {name}: {e}")
                raise
    finally:
        # Explicitly shutdown executor to ensure all threads are released
        executor.shutdown(wait=True)
        # Force garbage collection after thread pool shutdown
        gc.collect()

    return results_dict


def _build_eval_results_dict(results_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build evaluation results dictionary with all available models.

    Args:
        results_dict: Dictionary mapping model names to results.

    Returns:
        Dictionary containing evaluation results for all available models.
    """
    eval_results = {
        "lightgbm_complete": results_dict["lightgbm_complete"],
        "lightgbm_without_insights": results_dict["lightgbm_without_insights"],
    }
    if "lightgbm_sigma_plus_base" in results_dict:
        eval_results["lightgbm_sigma_plus_base"] = results_dict["lightgbm_sigma_plus_base"]
    if "lightgbm_log_volatility_only" in results_dict:
        eval_results["lightgbm_log_volatility_only"] = results_dict["lightgbm_log_volatility_only"]
    if "lightgbm_insights_only" in results_dict:
        eval_results["lightgbm_insights_only"] = results_dict["lightgbm_insights_only"]
    if "lightgbm_technical_only" in results_dict:
        eval_results["lightgbm_technical_only"] = results_dict["lightgbm_technical_only"]
    if "lightgbm_technical_plus_insights" in results_dict:
        eval_results["lightgbm_technical_plus_insights"] = results_dict[
            "lightgbm_technical_plus_insights"
        ]

    return eval_results


def _save_evaluation_results(results_dict: dict[str, dict[str, Any]]) -> None:
    """Save evaluation results to JSON file.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    eval_results = _build_eval_results_dict(results_dict)

    save_json_pretty(eval_results, LIGHTGBM_EVAL_RESULTS_FILE)

    logger.info(f"Evaluation results saved to {LIGHTGBM_EVAL_RESULTS_FILE}")


def _log_evaluation_summary(results_dict: dict[str, dict[str, Any]]) -> None:
    """Log evaluation summary for all models.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)

    for name, res in results_dict.items():
        metrics = res["test_metrics"]
        logger.info(f"\n{name}:")
        logger.info(f"  Test MAE:  {metrics['mae']:.6f}")
        logger.info(f"  Test MSE:  {metrics['mse']:.6f}")
        logger.info(f"  Test RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  Test R²:   {metrics['r2']:.6f}")
        logger.info(f"  Test size: {res['test_size']}")
        logger.info(f"  N features: {res['n_features']}")
        logger.info(f"  SHAP plot: {res['shap_plot_path']}")


def run_evaluation() -> dict[str, dict[str, Any]]:
    """Evaluate both LightGBM models in parallel on test set.

    Returns:
        Dictionary with evaluation results for both models.

    Raises:
        FileNotFoundError: If models or datasets are missing.
    """
    logger.info("=" * 70)
    logger.info("LightGBM Evaluation (Parallel) - Test Set")
    logger.info("=" * 70)

    tasks = _prepare_evaluation_tasks()
    results_dict = _run_parallel_evaluations(tasks)
    _save_evaluation_results(results_dict)
    _log_evaluation_summary(results_dict)
    log_model_comparison(results_dict)

    # Perform statistical significance tests
    statistical_tests = perform_statistical_tests(results_dict)

    # Combine all results
    eval_results = _build_eval_results_dict(results_dict)
    eval_results["statistical_tests"] = statistical_tests

    # Persist combined evaluation results
    save_json_pretty(eval_results, LIGHTGBM_EVAL_RESULTS_FILE)
    logger.info(f"Evaluation results (full) saved to {LIGHTGBM_EVAL_RESULTS_FILE}")

    # Final cleanup: free all large objects
    del results_dict
    del statistical_tests
    del tasks
    gc.collect()

    return eval_results
