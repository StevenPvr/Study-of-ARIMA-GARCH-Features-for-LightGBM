"""Ablation study: Remove sigma2_garch from complete model to test causal effect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.constants import (
    RF_DATASET_WITHOUT_SIGMA2_FILE,
    RF_MODELS_DIR,
    RF_OPTIMIZATION_N_TRIALS,
    RF_RESULTS_DIR,
)
from src.random_forest.data_preparation.utils import create_dataset_without_sigma2
from src.random_forest.eval.eval import evaluate_model, load_dataset
from src.random_forest.optimisation.optimisation import (
    load_dataset as load_opt_dataset,
    optimize_random_forest,
)
from src.random_forest.training.training import (
    load_dataset as load_train_dataset,
    train_random_forest,
)
from src.utils import get_logger

logger = get_logger(__name__)

ABLATION_RESULTS_FILE = RF_RESULTS_DIR / "ablation_sigma2_results.json"


def run_ablation_study() -> dict[str, Any]:
    """Run ablation study: train and evaluate model without sigma2_garch.

    Steps:
    1. Create dataset without sigma2_garch (but keeping other ARIMA-GARCH features)
    2. Optimize hyperparameters
    3. Train model
    4. Evaluate on test set
    5. Compare with existing models

    Returns:
        Dictionary with ablation study results and comparison.
    """
    logger.info("=" * 70)
    logger.info("ABLATION STUDY: Removing sigma2_garch from complete model")
    logger.info("=" * 70)

    # Step 1: Create dataset without sigma2_garch
    logger.info("\nStep 1: Creating dataset without sigma2_garch")
    create_dataset_without_sigma2()

    # Step 2: Optimize hyperparameters
    logger.info("\nStep 2: Optimizing hyperparameters")
    X, y = load_opt_dataset(RF_DATASET_WITHOUT_SIGMA2_FILE)
    opt_results, _study = optimize_random_forest(
        X, y, study_name="rf_without_sigma2", n_trials=RF_OPTIMIZATION_N_TRIALS
    )

    # Step 3: Train model
    logger.info("\nStep 3: Training model")
    X_train, y_train = load_train_dataset(RF_DATASET_WITHOUT_SIGMA2_FILE, split="train")
    model, train_info = train_random_forest(X_train, y_train, opt_results["best_params"])

    # Save model
    model_name = "rf_without_sigma2"
    model_path = RF_MODELS_DIR / f"{model_name}.joblib"
    RF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved model: {model_path}")

    # Step 4: Evaluate on test set
    logger.info("\nStep 4: Evaluating on test set")
    X_test, y_test = load_dataset(RF_DATASET_WITHOUT_SIGMA2_FILE, split="test")
    eval_results = evaluate_model(model, X_test, y_test, model_name)

    # Step 5: Load existing results for comparison
    logger.info("\nStep 5: Loading existing results for comparison")
    with open(RF_RESULTS_DIR / "eval_results.json") as f:
        existing_results = json.load(f)

    # Step 6: Compare results
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 70)

    complete_metrics = existing_results["rf_complete"]["test_metrics"]
    without_insights_metrics = existing_results["rf_without_insights"]["test_metrics"]
    ablation_metrics = eval_results["test_metrics"]

    logger.info("\nModel Comparison:")
    logger.info(f"{'Model':<30} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    logger.info("-" * 60)
    logger.info(
        f"{'Complete (with sigma2)':<30} "
        f"{complete_metrics['r2']:>10.4f} "
        f"{complete_metrics['mae']:>10.6f} "
        f"{complete_metrics['rmse']:>10.6f}"
    )
    logger.info(
        f"{'Without sigma2 (ablation)':<30} "
        f"{ablation_metrics['r2']:>10.4f} "
        f"{ablation_metrics['mae']:>10.6f} "
        f"{ablation_metrics['rmse']:>10.6f}"
    )
    logger.info(
        f"{'Without insights (baseline)':<30} "
        f"{without_insights_metrics['r2']:>10.4f} "
        f"{without_insights_metrics['mae']:>10.6f} "
        f"{without_insights_metrics['rmse']:>10.6f}"
    )

    # Calculate improvements
    r2_improvement = ablation_metrics["r2"] - complete_metrics["r2"]
    mae_improvement = complete_metrics["mae"] - ablation_metrics["mae"]
    rmse_improvement = complete_metrics["rmse"] - ablation_metrics["rmse"]

    logger.info("\n" + "=" * 70)
    logger.info("CAUSAL EFFECT OF REMOVING sigma2_garch")
    logger.info("=" * 70)
    logger.info(f"R² improvement: {r2_improvement:+.4f} ({r2_improvement*100:+.2f} points)")
    logger.info(f"MAE improvement: {mae_improvement:+.6f} (lower is better)")
    logger.info(f"RMSE improvement: {rmse_improvement:+.6f} (lower is better)")

    if ablation_metrics["r2"] > complete_metrics["r2"]:
        logger.info("\n✓ Removing sigma2_garch IMPROVES performance")
        logger.info(
            f"  Performance is now {ablation_metrics['r2']/complete_metrics['r2']:.2%} of complete model"
        )
    else:
        logger.info("\n✗ Removing sigma2_garch DEGRADES performance")

    # Compare with without_insights baseline
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON WITH BASELINE (without_insights)")
    logger.info("=" * 70)
    r2_diff_baseline = ablation_metrics["r2"] - without_insights_metrics["r2"]
    logger.info(
        f"R² difference vs baseline: {r2_diff_baseline:+.4f} ({r2_diff_baseline*100:+.2f} points)"
    )

    if abs(r2_diff_baseline) < 0.01:
        logger.info("✓ Ablation model performs similarly to baseline (within 1%)")
        logger.info("  This confirms that sigma2_garch is the main culprit")
    elif r2_diff_baseline > 0:
        logger.info("✓ Ablation model performs BETTER than baseline")
        logger.info("  Other ARIMA-GARCH features (arima_pred, sigma_garch, etc.) help!")
    else:
        logger.info("✗ Ablation model performs WORSE than baseline")
        logger.info("  Other ARIMA-GARCH features also contribute negatively")

    # Save results
    results = {
        "ablation_model": {
            "model_name": model_name,
            "test_metrics": ablation_metrics,
            "train_info": train_info,
            "best_params": opt_results["best_params"],
            "n_features": int(X_test.shape[1]),
        },
        "comparison": {
            "complete": complete_metrics,
            "without_sigma2": ablation_metrics,
            "without_insights": without_insights_metrics,
            "r2_improvement": float(r2_improvement),
            "mae_improvement": float(mae_improvement),
            "rmse_improvement": float(rmse_improvement),
            "r2_diff_vs_baseline": float(r2_diff_baseline),
        },
    }

    RF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {ABLATION_RESULTS_FILE}")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    run_ablation_study()
