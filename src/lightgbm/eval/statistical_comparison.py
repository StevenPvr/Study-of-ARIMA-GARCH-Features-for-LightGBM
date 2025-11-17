"""Statistical comparison utilities for LightGBM evaluation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np

from src.constants import (
    LIGHTGBM_DATASET_COMPLETE_FILE,
    LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
    LIGHTGBM_MODELS_DIR,
)
from src.lightgbm.eval.data_loading import load_dataset, load_model
from src.lightgbm.eval.statistical_tests import bootstrap_r2_comparison, compare_models_statistical
from src.lightgbm.model_utils import filter_existing_models, get_optional_model_configs
from src.utils import get_logger

logger = get_logger(__name__)


def _load_model_predictions(
    dataset_path: Path, model_path: Path, model_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset and model, return predictions and targets.

    Args:
        dataset_path: Path to dataset.
        model_path: Path to model.
        model_name: Name of the model.

    Returns:
        Tuple of (predictions, targets) as numpy arrays.
    """
    X_test, y_test = load_dataset(dataset_path, split="test")
    model = load_model(model_path)
    try:
        y_pred = model.predict(X_test)
        y_pred_array = np.asarray(y_pred).flatten()
        y_test_array = np.asarray(y_test).flatten()
        return y_pred_array, y_test_array
    finally:
        # Free model resources and memory
        del model
        del X_test
        del y_test
        gc.collect()


def _prepare_base_models() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Prepare predictions and targets for base models.

    Returns:
        Tuple of (predictions dict, targets dict).
    """
    predictions: dict[str, np.ndarray] = {}
    y_tests: dict[str, np.ndarray] = {}

    # Base models
    yc, y_test_complete = _load_model_predictions(
        LIGHTGBM_DATASET_COMPLETE_FILE,
        LIGHTGBM_MODELS_DIR / "lightgbm_complete.joblib",
        "lightgbm_complete",
    )
    predictions["lightgbm_complete"] = yc
    y_tests["lightgbm_complete"] = y_test_complete

    yw, y_test_without = _load_model_predictions(
        LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE,
        LIGHTGBM_MODELS_DIR / "lightgbm_without_insights.joblib",
        "lightgbm_without_insights",
    )
    predictions["lightgbm_without_insights"] = yw
    y_tests["lightgbm_without_insights"] = y_test_without

    return predictions, y_tests


def _add_optional_models(
    predictions: dict[str, np.ndarray], y_tests: dict[str, np.ndarray]
) -> None:
    """Add optional models if datasets and models exist.

    Args:
        predictions: Dictionary to add predictions to.
        y_tests: Dictionary to add targets to.
    """
    optional_configs = get_optional_model_configs()
    existing_configs = filter_existing_models(optional_configs)

    for dataset_path, model_path, model_name in existing_configs:
        preds, targets = _load_model_predictions(dataset_path, model_path, model_name)
        predictions[model_name] = preds
        y_tests[model_name] = targets


def _can_compare_models(
    model1: str, model2: str, y_tests: dict[str, np.ndarray]
) -> tuple[bool, np.ndarray | None]:
    """Check if two models can be compared (same dataset).

    Args:
        model1: Name of first model.
        model2: Name of second model.
        y_tests: Dictionary mapping model names to targets.

    Returns:
        Tuple of (can_compare, y_test_array).
    """
    if model1 not in y_tests or model2 not in y_tests:
        return False, None

    y_test1 = y_tests[model1]
    y_test2 = y_tests[model2]

    if len(y_test1) != len(y_test2):
        logger.warning(
            f"Skipping comparison {model1} vs {model2}: "
            f"different dataset sizes ({len(y_test1)} vs {len(y_test2)})"
        )
        return False, None

    if not np.array_equal(y_test1, y_test2):
        logger.warning(
            f"Skipping comparison {model1} vs {model2}: "
            "different target values (different datasets)"
        )
        return False, None

    return True, y_test1


def _add_comparison(
    model1: str,
    model2: str,
    predictions: dict[str, np.ndarray],
    y_tests: dict[str, np.ndarray],
    comparisons: dict[str, Any],
) -> None:
    """Add comparison between two models if they use the same dataset.

    Args:
        model1: Name of first model.
        model2: Name of second model.
        predictions: Dictionary mapping model names to predictions.
        y_tests: Dictionary mapping model names to targets.
        comparisons: Dictionary to add comparison results to.
    """
    can_compare, y_test = _can_compare_models(model1, model2, y_tests)
    if not can_compare or y_test is None:
        return

    key = f"{model1}_vs_{model2}"
    comparisons[key] = {
        "diebold_mariano": compare_models_statistical(
            y_test, predictions[model1], predictions[model2], model1, model2
        ),
        "bootstrap_r2": bootstrap_r2_comparison(
            y_test, predictions[model1], predictions[model2], n_bootstrap=None
        ),
    }


def _determine_comparison_pairs(predictions: dict[str, np.ndarray]) -> list[tuple[str, str]]:
    """Determine which model pairs should be compared.

    Args:
        predictions: Dictionary mapping model names to predictions.

    Returns:
        List of (model1, model2) tuples to compare.
    """
    comparison_pairs: list[tuple[str, str]] = [
        ("lightgbm_complete", "lightgbm_without_insights"),
    ]

    # Compare sigma_plus_base with log_volatility_only if both exist (they use same dataset)
    if "lightgbm_sigma_plus_base" in predictions and "lightgbm_log_volatility_only" in predictions:
        comparison_pairs.append(("lightgbm_sigma_plus_base", "lightgbm_log_volatility_only"))

    # Compare technical_plus_insights with technical_only if both exist
    if (
        "lightgbm_technical_plus_insights" in predictions
        and "lightgbm_technical_only" in predictions
    ):
        comparison_pairs.append(("lightgbm_technical_plus_insights", "lightgbm_technical_only"))

    # Compare insights_only with other models if it exists
    if "lightgbm_insights_only" in predictions:
        # Compare with without_insights to see the value of insights
        comparison_pairs.append(("lightgbm_without_insights", "lightgbm_insights_only"))
        # Compare with sigma_plus_base if both exist
        if "lightgbm_sigma_plus_base" in predictions:
            comparison_pairs.append(("lightgbm_sigma_plus_base", "lightgbm_insights_only"))

    return comparison_pairs


def perform_statistical_tests(results_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Perform Diebold-Mariano statistical test to compare models.

    Args:
        results_dict: Dictionary mapping model names to results with predictions.

    Returns:
        Dictionary with statistical test results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 70)

    comparisons: dict[str, Any] = {}

    # Load predictions and targets
    predictions, y_tests = _prepare_base_models()
    _add_optional_models(predictions, y_tests)

    # Determine comparison pairs
    comparison_pairs = _determine_comparison_pairs(predictions)

    # Perform comparisons
    seen_pairs: set[tuple[str, str]] = set()
    for model1, model2 in comparison_pairs:
        if model1 not in predictions or model2 not in predictions:
            continue
        if (model1, model2) in seen_pairs:
            continue
        _add_comparison(model1, model2, predictions, y_tests, comparisons)
        seen_pairs.add((model1, model2))

    logger.info("\n" + "=" * 70)
    return comparisons


def _calculate_metric_difference(
    metrics1: dict[str, float], metrics2: dict[str, float], metric_name: str
) -> tuple[float, str]:
    """Calculate difference between two metrics and determine which is better.

    Args:
        metrics1: First model metrics.
        metrics2: Second model metrics.
        metric_name: Name of metric to compare.

    Returns:
        Tuple of (difference, better_indicator).
    """
    diff = metrics1[metric_name] - metrics2[metric_name]

    if metric_name == "r2":
        better = "better" if diff > 0 else "worse"
    else:  # mae, mse, rmse
        better = "better" if diff < 0 else "worse"

    return diff, better


def _log_model_pair_comparison(
    model1_name: str,
    model2_name: str,
    metrics1: dict[str, float],
    metrics2: dict[str, float],
) -> None:
    """Log comparison between two models.

    Args:
        model1_name: Name of first model.
        model2_name: Name of second model.
        metrics1: Metrics for first model.
        metrics2: Metrics for second model.
    """
    logger.info(f"\n{model1_name} vs {model2_name}:")
    for metric_name in ["mae", "mse", "rmse", "r2"]:
        diff, better = _calculate_metric_difference(metrics1, metrics2, metric_name)
        logger.info(f"  {metric_name.upper()} difference:  {diff:+.6f} ({better})")


def log_model_comparison(results_dict: dict[str, dict[str, Any]]) -> None:
    """Log comparisons across core LightGBM variants.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    complete_metrics = results_dict["lightgbm_complete"]["test_metrics"]
    without_metrics = results_dict["lightgbm_without_insights"]["test_metrics"]

    _log_model_pair_comparison("Complete", "Without Insights", complete_metrics, without_metrics)

    if "lightgbm_sigma_plus_base" in results_dict:
        sigma_metrics = results_dict["lightgbm_sigma_plus_base"]["test_metrics"]
        _log_model_pair_comparison("Complete", "Sigma-Plus-Base", complete_metrics, sigma_metrics)
        _log_model_pair_comparison(
            "Without Insights", "Sigma-Plus-Base", without_metrics, sigma_metrics
        )

    logger.info("\n" + "=" * 70)
