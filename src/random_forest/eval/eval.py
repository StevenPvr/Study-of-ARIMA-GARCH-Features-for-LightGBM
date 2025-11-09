"""Random Forest model evaluation on test set with SHAP analysis."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, cast

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless/parallel execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.constants import (
    RF_DATASET_COMPLETE_FILE,
    RF_DATASET_WITHOUT_INSIGHTS_FILE,
    RF_DATASET_SIGMA_PLUS_BASE_FILE,
    RF_DATASET_RSI14_ONLY_FILE,
    RF_DATASET_TECHNICAL_INDICATORS_FILE,
    RF_EVAL_RESAMPLE_FRACTION,
    RF_EVAL_RESAMPLE_MIN_SIZE,
    RF_EVAL_RESAMPLE_N_WINDOWS,
    RF_EVAL_RESULTS_FILE,
    RF_MODELS_DIR,
    RF_RESULTS_DIR,
    RF_SHAP_MAX_DISPLAY_DEFAULT,
    RF_SHAP_PLOTS_DIR,
    RF_PERMUTATION_RESULTS_FILE,
)
from src.random_forest.eval.statistical_tests import (
    bootstrap_r2_comparison,
    compare_models_statistical,
)
from src.random_forest.eval.permutation import (
    compute_block_permutation_importance,
    plot_permutation_bars,
    save_permutation_results,
)
from src.utils import get_logger

# Lock for matplotlib operations (not thread-safe)
_matplotlib_lock = Lock()

logger = get_logger(__name__)

# Constants for evaluation (aliases for backward compatibility)
RF_DATASET_COMPLETE = RF_DATASET_COMPLETE_FILE
RF_DATASET_WITHOUT_INSIGHTS = RF_DATASET_WITHOUT_INSIGHTS_FILE


def _filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter dataframe by split column if present.

    Args:
        df: Input dataframe.
        split: Split to filter ('train' or 'test').

    Returns:
        Filtered dataframe.

    Raises:
        ValueError: If no data found for the specified split.
    """
    if "split" in df.columns:
        df = cast(pd.DataFrame, df[df["split"] == split].copy())
        logger.info(f"Filtered to {split} split: {len(df)} rows")
        if df.empty:
            raise ValueError(f"No data found for split '{split}'")
    return df


def _drop_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove date and split columns from dataframe.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with metadata columns removed.
    """
    columns_to_drop = []
    if "date" in df.columns:
        columns_to_drop.append("date")
    if "split" in df.columns:
        columns_to_drop.append("split")
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def load_dataset(dataset_path: Path, split: str = "test") -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split into features and target.

    Args:
        dataset_path: Path to the dataset CSV file.
        split: Split to load ('train' or 'test'). Default is 'test'.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If dataset is empty or missing required columns.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Dataset is empty: {dataset_path}") from e

    if df.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    df = _filter_by_split(df, split)
    df = _drop_metadata_columns(df)

    if "weighted_log_return" not in df.columns:
        raise ValueError("Dataset must contain 'weighted_log_return' column")

    X = cast(pd.DataFrame, df.drop(columns=["weighted_log_return"]))
    y = cast(pd.Series, df["weighted_log_return"].copy())

    logger.info(f"Loaded dataset: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def load_model(model_path: Path) -> RandomForestRegressor:
    """Load trained Random Forest model.

    Args:
        model_path: Path to the model joblib file.

    Returns:
        Loaded Random Forest model.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    if not isinstance(model, RandomForestRegressor):
        raise ValueError(f"Expected RandomForestRegressor, got {type(model)}")

    return model


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with evaluation metrics (MAE, MSE, RMSE, R²).
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return metrics


def _extract_shap_values(explainer: shap.TreeExplainer, X_sample: pd.DataFrame) -> np.ndarray:
    """Extract and validate SHAP values from explainer.

    Args:
        explainer: SHAP TreeExplainer instance.
        X_sample: Features DataFrame.

    Returns:
        Validated SHAP values array.

    Raises:
        ValueError: If SHAP values shape doesn't match input shape.
    """
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    if shap_values.shape[0] != len(X_sample) or shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError(
            f"SHAP values shape {shap_values.shape} doesn't match "
            f"X_sample shape {X_sample.shape}"
        )

    logger.info(
        f"SHAP values computed: shape={shap_values.shape}, "
        f"min={shap_values.min():.6f}, max={shap_values.max():.6f}, "
        f"mean={shap_values.mean():.6f}, std={shap_values.std():.6f}"
    )

    return shap_values


def _create_shap_explanation(
    shap_values: np.ndarray,
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
) -> shap.Explanation:
    """Create SHAP Explanation object.

    Args:
        shap_values: SHAP values array.
        explainer: SHAP TreeExplainer instance.
        X_sample: Features DataFrame.

    Returns:
        SHAP Explanation object.
    """
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist(),
    )


def _calculate_display_max(
    model_name: str,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    max_display: int,
) -> int:
    """Calculate max_display ensuring sigma2_garch is shown for rf_complete.

    Args:
        model_name: Name of the model.
        X_sample: Features DataFrame.
        shap_values: SHAP values array.
        max_display: Initial maximum display value.

    Returns:
        Adjusted max_display value.
    """
    if model_name != "rf_complete" or "sigma2_garch" not in X_sample.columns:
        return max_display

    feature_importances = np.abs(shap_values).mean(axis=0)
    feature_names_list = X_sample.columns.tolist()
    sorted_indices = np.argsort(feature_importances)[::-1]
    sigma2_garch_idx = feature_names_list.index("sigma2_garch")
    sigma2_garch_rank = np.where(sorted_indices == sigma2_garch_idx)[0][0]

    if sigma2_garch_rank >= max_display:
        display_max = sigma2_garch_rank + 1
        logger.info(
            f"Increased max_display to {display_max} to ensure sigma2_garch "
            f"(rank {sigma2_garch_rank + 1}) is displayed for {model_name}"
        )
        return display_max

    return max_display


def _save_shap_plot(
    explanation: shap.Explanation,
    model_name: str,
    output_dir: Path,
    display_max: int,
) -> Path:
    """Save SHAP beeswarm plot to file.

    Args:
        explanation: SHAP Explanation object.
        model_name: Name of the model.
        output_dir: Directory to save plot.
        display_max: Maximum number of features to display.

    Returns:
        Path to saved plot file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_name}_shap_summary.png"

    with _matplotlib_lock:
        shap.plots.beeswarm(explanation, max_display=display_max, show=False)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"SHAP Feature Importance - {model_name}", fontsize=14, y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"SHAP plot saved to {plot_path}")
    return plot_path


def _create_resample_windows(n_samples: int) -> list[tuple[int, int]]:
    """Create contiguous resample windows covering the test split.

    Args:
        n_samples: Number of samples available for evaluation.

    Returns:
        List of (start_index, end_index) tuples representing evaluation windows.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive to create resample windows")

    window_size = max(
        int(n_samples * RF_EVAL_RESAMPLE_FRACTION),
        RF_EVAL_RESAMPLE_MIN_SIZE,
    )
    window_size = min(window_size, n_samples)

    if window_size == n_samples or RF_EVAL_RESAMPLE_N_WINDOWS <= 1:
        logger.info(
            "Resample evaluation uses a single window (size=%d, n_samples=%d)",
            window_size,
            n_samples,
        )
        return [(0, n_samples)]

    max_start = n_samples - window_size
    effective_windows = min(RF_EVAL_RESAMPLE_N_WINDOWS, max_start + 1)
    start_positions = np.linspace(0, max_start, num=effective_windows, dtype=int)
    starts = sorted({int(pos) for pos in start_positions})

    windows: list[tuple[int, int]] = [(start, start + window_size) for start in starts]
    final_window = (max_start, n_samples)
    if windows[-1] != final_window:
        windows.append(final_window)

    seen_windows: set[tuple[int, int]] = set()
    unique_windows: list[tuple[int, int]] = []
    for start, end in windows:
        key = (int(start), int(end))
        if key in seen_windows:
            continue
        seen_windows.add(key)
        unique_windows.append(key)

    logger.info(
        "Resample evaluation windows configured: window_size=%d, count=%d, n_samples=%d",
        window_size,
        len(unique_windows),
        n_samples,
    )
    return unique_windows


def _aggregate_metric_statistics(
    metrics_list: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Aggregate metric statistics across resampled evaluations."""
    if not metrics_list:
        return {}

    aggregated: dict[str, dict[str, float]] = {}
    metric_names = list(metrics_list[0].keys())

    for metric_name in metric_names:
        values = np.array([metrics[metric_name] for metrics in metrics_list], dtype=float)
        aggregated[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return aggregated


def _log_resample_summary(
    model_name: str,
    summary: dict[str, dict[str, float]],
) -> None:
    """Log summary statistics for resampled evaluations."""
    if not summary:
        logger.info("%s - Resampled evaluation summary unavailable (single window)", model_name)
        return

    logger.info("\n%s - RESAMPLED EVALUATION SUMMARY", model_name.upper())
    for metric, stats in summary.items():
        logger.info(
            "  %s -> mean: %.6f | std: %.6f | min: %.6f | max: %.6f",
            metric.upper(),
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
        )


def compute_shap_values(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    model_name: str,
    output_dir: Path = RF_SHAP_PLOTS_DIR,
    max_display: int = RF_SHAP_MAX_DISPLAY_DEFAULT,
) -> tuple[shap.Explanation, Path]:
    """Compute SHAP values and create visualization.

    Args:
        model: Trained Random Forest model.
        X: Features DataFrame.
        model_name: Name for the model (for plot title).
        output_dir: Directory to save SHAP plots.
        max_display: Maximum number of features to display.

    Returns:
        Tuple of (SHAP explanation object, path to saved plot).
    """
    logger.info(f"Computing SHAP values for {model_name}")

    explainer = shap.TreeExplainer(model)
    shap_values = _extract_shap_values(explainer, X)
    explanation = _create_shap_explanation(shap_values, explainer, X)
    display_max = _calculate_display_max(model_name, X, shap_values, max_display)
    plot_path = _save_shap_plot(explanation, model_name, output_dir, display_max)

    return explanation, plot_path


def _log_model_metrics(model_name: str, metrics: dict[str, float]) -> None:
    """Log model evaluation metrics.

    Args:
        model_name: Name of the model.
        metrics: Dictionary of evaluation metrics.
    """
    logger.info(f"{model_name} - Test MAE: {metrics['mae']:.6f}")
    logger.info(f"{model_name} - Test RMSE: {metrics['rmse']:.6f}")
    logger.info(f"{model_name} - Test R²: {metrics['r2']:.6f}")


def _get_sorted_feature_importances(
    model: RandomForestRegressor, feature_names: pd.Index
) -> dict[str, float]:
    """Get feature importances sorted by importance.

    Args:
        model: Trained Random Forest model.
        feature_names: Feature names.

    Returns:
        Dictionary mapping feature names to importances, sorted by importance.
    """
    feature_importances = dict(zip(feature_names, model.feature_importances_, strict=False))
    return dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    *,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Evaluate model on test set with metrics and SHAP analysis.

    Args:
        model: Trained Random Forest model.
        X_test: Test features.
        y_test: Test target.
        model_name: Name for the model.
        compute_shap: Whether to compute SHAP values for this evaluation.

    Returns:
        Dictionary with evaluation results (metrics, SHAP info).
    """
    logger.info(f"Evaluating model: {model_name}")

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    _log_model_metrics(model_name, metrics)

    shap_plot_path: Path | None = None
    if compute_shap:
        _, shap_plot_path = compute_shap_values(
            model,
            X_test,
            model_name,
            output_dir=RF_SHAP_PLOTS_DIR,
        )

    feature_importances = _get_sorted_feature_importances(model, X_test.columns)

    results = {
        "model_name": model_name,
        "test_metrics": metrics,
        "test_size": int(len(X_test)),
        "n_features": int(X_test.shape[1]),
        "feature_importances": {k: float(v) for k, v in feature_importances.items()},
        "shap_plot_path": str(shap_plot_path) if shap_plot_path else None,
    }

    return results


def _evaluate_on_resamples(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate model on contiguous resampled subsets of the test split."""
    windows = _create_resample_windows(len(X_test))
    runs: list[dict[str, Any]] = []

    for idx, (start, end) in enumerate(windows, start=1):
        X_subset = X_test.iloc[start:end]
        y_subset = y_test.iloc[start:end]
        subset_name = f"{model_name}_resample_{idx}"

        subset_results = evaluate_model(
            model,
            X_subset,
            y_subset,
            subset_name,
            compute_shap=False,
        )

        runs.append(
            {
                "index": idx,
                "start_index": int(start),
                "end_index": int(end - 1),
                "n_samples": int(end - start),
                "metrics": subset_results["test_metrics"],
            }
        )

    summary = _aggregate_metric_statistics([run["metrics"] for run in runs])
    _log_resample_summary(model_name, summary)

    return {
        "runs": runs,
        "summary": summary,
    }


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

    # Evaluate
    results = evaluate_model(model, X_test, y_test, model_name)
    resampled_metrics = _evaluate_on_resamples(model, X_test, y_test, model_name)
    results["resampled_metrics"] = resampled_metrics

    logger.info(f"✓ Completed evaluation: {model_name}")
    return model_name, results


def _prepare_evaluation_tasks() -> list[tuple[Path, Path, str]]:
    """Prepare evaluation tasks for both models.

    Returns:
        List of tuples (dataset_path, model_path, model_name).
    """
    tasks: list[tuple[Path, Path, str]] = [
        (
            RF_DATASET_COMPLETE,
            RF_MODELS_DIR / "rf_complete.joblib",
            "rf_complete",
        ),
        (
            RF_DATASET_WITHOUT_INSIGHTS,
            RF_MODELS_DIR / "rf_without_insights.joblib",
            "rf_without_insights",
        ),
    ]
    # Optional third model: sigma-plus-base if both dataset and model exist
    ds_sigma_plus_base = RF_DATASET_SIGMA_PLUS_BASE_FILE
    model_sigma_plus_base = RF_MODELS_DIR / "rf_sigma_plus_base.joblib"
    if ds_sigma_plus_base.exists() and model_sigma_plus_base.exists():
        tasks.append((ds_sigma_plus_base, model_sigma_plus_base, "rf_sigma_plus_base"))
    
    # Optional fourth model: rsi14-only if both dataset and model exist
    ds_rsi14 = RF_DATASET_RSI14_ONLY_FILE
    model_rsi14 = RF_MODELS_DIR / "rf_rsi14_only.joblib"
    if ds_rsi14.exists() and model_rsi14.exists():
        tasks.append((ds_rsi14, model_rsi14, "rf_rsi14_only"))

    ds_technical = RF_DATASET_TECHNICAL_INDICATORS_FILE
    model_technical = RF_MODELS_DIR / "rf_technical_indicators.joblib"
    if ds_technical.exists() and model_technical.exists():
        tasks.append((ds_technical, model_technical, "rf_technical_indicators"))
    
    return tasks


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
    with ThreadPoolExecutor(max_workers=2) as executor:
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

    return results_dict


def _save_evaluation_results(results_dict: dict[str, dict[str, Any]]) -> None:
    """Save evaluation results to JSON file.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    eval_results = {
        "rf_complete": results_dict["rf_complete"],
        "rf_without_insights": results_dict["rf_without_insights"],
    }
    if "rf_sigma_plus_base" in results_dict:
        eval_results["rf_sigma_plus_base"] = results_dict["rf_sigma_plus_base"]
    if "rf_rsi14_only" in results_dict:
        eval_results["rf_rsi14_only"] = results_dict["rf_rsi14_only"]
    if "rf_technical_indicators" in results_dict:
        eval_results["rf_technical_indicators"] = results_dict["rf_technical_indicators"]

    RF_EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RF_EVAL_RESULTS_FILE, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Evaluation results saved to {RF_EVAL_RESULTS_FILE}")


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
        logger.info(f"  SHAP plot: {res['shap_plot_path']}")

        resampled = res.get("resampled_metrics", {})
        summary = resampled.get("summary", {})
        if summary:
            logger.info("  Resampled metrics summary:")
            for metric_name, stats in summary.items():
                logger.info(
                    "    %s -> mean %.6f | std %.6f | min %.6f | max %.6f",
                    metric_name.upper(),
                    stats["mean"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                )


def _log_model_comparison(results_dict: dict[str, dict[str, Any]]) -> None:
    """Log comparisons across core Random Forest variants.

    Args:
        results_dict: Dictionary mapping model names to results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    complete_metrics = results_dict["rf_complete"]["test_metrics"]
    without_metrics = results_dict["rf_without_insights"]["test_metrics"]

    mae_diff = complete_metrics["mae"] - without_metrics["mae"]
    mse_diff = complete_metrics["mse"] - without_metrics["mse"]
    rmse_diff = complete_metrics["rmse"] - without_metrics["rmse"]
    r2_diff = complete_metrics["r2"] - without_metrics["r2"]

    logger.info("\nComplete vs Without Insights:")
    logger.info(f"  MAE difference:  {mae_diff:+.6f} ({'better' if mae_diff < 0 else 'worse'})")
    logger.info(f"  MSE difference:  {mse_diff:+.6f} ({'better' if mse_diff < 0 else 'worse'})")
    logger.info(f"  RMSE difference: {rmse_diff:+.6f} ({'better' if rmse_diff < 0 else 'worse'})")
    logger.info(f"  R² difference:   {r2_diff:+.6f} ({'better' if r2_diff > 0 else 'worse'})")

    if "rf_sigma_plus_base" in results_dict:
        sigma_metrics = results_dict["rf_sigma_plus_base"]["test_metrics"]
        logger.info("\nComplete vs Sigma-Plus-Base:")
        logger.info(
            f"  MAE difference:  {complete_metrics['mae'] - sigma_metrics['mae']:+.6f}"
            f" ({'better' if complete_metrics['mae'] < sigma_metrics['mae'] else 'worse'})"
        )
        logger.info(
            f"  MSE difference:  {complete_metrics['mse'] - sigma_metrics['mse']:+.6f}"
            f" ({'better' if complete_metrics['mse'] < sigma_metrics['mse'] else 'worse'})"
        )
        logger.info(
            f"  RMSE difference: {complete_metrics['rmse'] - sigma_metrics['rmse']:+.6f}"
            f" ({'better' if complete_metrics['rmse'] < sigma_metrics['rmse'] else 'worse'})"
        )
        logger.info(
            f"  R² difference:   {complete_metrics['r2'] - sigma_metrics['r2']:+.6f}"
            f" ({'better' if complete_metrics['r2'] > sigma_metrics['r2'] else 'worse'})"
        )

        logger.info("\nWithout Insights vs Sigma-Plus-Base:")
        logger.info(
            f"  MAE difference:  {without_metrics['mae'] - sigma_metrics['mae']:+.6f}"
            f" ({'better' if without_metrics['mae'] < sigma_metrics['mae'] else 'worse'})"
        )
        logger.info(
            f"  MSE difference:  {without_metrics['mse'] - sigma_metrics['mse']:+.6f}"
            f" ({'better' if without_metrics['mse'] < sigma_metrics['mse'] else 'worse'})"
        )
        logger.info(
            f"  RMSE difference: {without_metrics['rmse'] - sigma_metrics['rmse']:+.6f}"
            f" ({'better' if without_metrics['rmse'] < sigma_metrics['rmse'] else 'worse'})"
        )
        logger.info(
            f"  R² difference:   {without_metrics['r2'] - sigma_metrics['r2']:+.6f}"
            f" ({'better' if without_metrics['r2'] > sigma_metrics['r2'] else 'worse'})"
        )

    if "rf_technical_indicators" in results_dict:
        technical_metrics = results_dict["rf_technical_indicators"]["test_metrics"]
        logger.info("\nComplete vs Technical Indicators:")
        logger.info(
            f"  MAE difference:  {complete_metrics['mae'] - technical_metrics['mae']:+.6f}"
            f" ({'better' if complete_metrics['mae'] < technical_metrics['mae'] else 'worse'})"
        )
        logger.info(
            f"  MSE difference:  {complete_metrics['mse'] - technical_metrics['mse']:+.6f}"
            f" ({'better' if complete_metrics['mse'] < technical_metrics['mse'] else 'worse'})"
        )
        logger.info(
            f"  RMSE difference: {complete_metrics['rmse'] - technical_metrics['rmse']:+.6f}"
            f" ({'better' if complete_metrics['rmse'] < technical_metrics['rmse'] else 'worse'})"
        )
        logger.info(
            f"  R² difference:   {complete_metrics['r2'] - technical_metrics['r2']:+.6f}"
            f" ({'better' if complete_metrics['r2'] > technical_metrics['r2'] else 'worse'})"
        )

    logger.info("\n" + "=" * 70)


def _perform_statistical_tests(results_dict: dict[str, dict[str, Any]]) -> dict[str, Any]:
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

    # Base models
    Xc, y_test = load_dataset(RF_DATASET_COMPLETE, split="test")
    Xw, _ = load_dataset(RF_DATASET_WITHOUT_INSIGHTS, split="test")
    mc = load_model(RF_MODELS_DIR / "rf_complete.joblib")
    mw = load_model(RF_MODELS_DIR / "rf_without_insights.joblib")
    yc = mc.predict(Xc)
    yw = mw.predict(Xw)

    predictions: dict[str, np.ndarray] = {
        "rf_complete": yc,
        "rf_without_insights": yw,
    }

    # Optional datasets
    ds_sigma_plus_base = RF_DATASET_SIGMA_PLUS_BASE_FILE
    model_sigma_plus_base = RF_MODELS_DIR / "rf_sigma_plus_base.joblib"
    if ds_sigma_plus_base.exists() and model_sigma_plus_base.exists():
        Xs, _ = load_dataset(ds_sigma_plus_base, split="test")
        ms = load_model(model_sigma_plus_base)
        ys = ms.predict(Xs)
        predictions["rf_sigma_plus_base"] = ys

    ds_rsi14 = RF_DATASET_RSI14_ONLY_FILE
    model_rsi14 = RF_MODELS_DIR / "rf_rsi14_only.joblib"
    if ds_rsi14.exists() and model_rsi14.exists():
        Xr, _ = load_dataset(ds_rsi14, split="test")
        mr = load_model(model_rsi14)
        yr = mr.predict(Xr)
        predictions["rf_rsi14_only"] = yr

    ds_technical = RF_DATASET_TECHNICAL_INDICATORS_FILE
    model_technical = RF_MODELS_DIR / "rf_technical_indicators.joblib"
    if ds_technical.exists() and model_technical.exists():
        Xt, _ = load_dataset(ds_technical, split="test")
        mt = load_model(model_technical)
        predictions["rf_technical_indicators"] = mt.predict(Xt)

    def _add_comparison(model1: str, model2: str) -> None:
        key = f"{model1}_vs_{model2}"
        comparisons[key] = {
            "diebold_mariano": compare_models_statistical(
                y_test, predictions[model1], predictions[model2], model1, model2
            ),
            "bootstrap_r2": bootstrap_r2_comparison(
                y_test, predictions[model1], predictions[model2], n_bootstrap=1000
            ),
        }

    comparison_pairs: list[tuple[str, str]] = [
        ("rf_complete", "rf_without_insights"),
    ]

    if "rf_sigma_plus_base" in predictions:
        comparison_pairs.extend(
            [
                ("rf_complete", "rf_sigma_plus_base"),
                ("rf_without_insights", "rf_sigma_plus_base"),
            ]
        )

    if "rf_rsi14_only" in predictions:
        comparison_pairs.extend(
            [
                ("rf_complete", "rf_rsi14_only"),
                ("rf_without_insights", "rf_rsi14_only"),
            ]
        )

    if "rf_technical_indicators" in predictions:
        comparison_pairs.extend(
            [
                ("rf_complete", "rf_technical_indicators"),
                ("rf_without_insights", "rf_technical_indicators"),
            ]
        )
        if "rf_sigma_plus_base" in predictions:
            comparison_pairs.append(("rf_sigma_plus_base", "rf_technical_indicators"))
        if "rf_rsi14_only" in predictions:
            comparison_pairs.append(("rf_rsi14_only", "rf_technical_indicators"))

    seen_pairs: set[tuple[str, str]] = set()
    for model1, model2 in comparison_pairs:
        if model1 not in predictions or model2 not in predictions:
            continue
        if (model1, model2) in seen_pairs:
            continue
        _add_comparison(model1, model2)
        seen_pairs.add((model1, model2))

    logger.info("\n" + "=" * 70)
    return comparisons


def run_evaluation() -> dict[str, dict[str, Any]]:
    """Evaluate both Random Forest models in parallel on test set.

    Returns:
        Dictionary with evaluation results for both models.

    Raises:
        FileNotFoundError: If models or datasets are missing.
    """
    logger.info("=" * 70)
    logger.info("Random Forest Evaluation (Parallel) - Test Set")
    logger.info("=" * 70)

    tasks = _prepare_evaluation_tasks()
    results_dict = _run_parallel_evaluations(tasks)
    _save_evaluation_results(results_dict)
    _log_evaluation_summary(results_dict)
    _log_model_comparison(results_dict)

    # Perform statistical significance tests
    statistical_tests = _perform_statistical_tests(results_dict)

    # Compute block permutation importance (rf_complete and rf_without_insights)
    per_model_perm: dict[str, dict[str, dict[str, float]]] = {}
    for dataset, model_path, name in tasks:
        X_test, y_test = load_dataset(dataset, split="test")
        model = load_model(model_path)
        perm_stats = compute_block_permutation_importance(
            model, X_test, y_test, block_size=20, n_repeats=50
        )
        per_model_perm[name] = perm_stats

    if per_model_perm:
        save_permutation_results(per_model_perm, output_json=RF_PERMUTATION_RESULTS_FILE)
        plot_permutation_bars(per_model_perm, top_k=20)

    eval_results = {
        "rf_complete": results_dict["rf_complete"],
        "rf_without_insights": results_dict["rf_without_insights"],
        "statistical_tests": statistical_tests,
    }
    if "rf_sigma_plus_base" in results_dict:
        eval_results["rf_sigma_plus_base"] = results_dict["rf_sigma_plus_base"]
    if "rf_rsi14_only" in results_dict:
        eval_results["rf_rsi14_only"] = results_dict["rf_rsi14_only"]
    if "rf_technical_indicators" in results_dict:
        eval_results["rf_technical_indicators"] = results_dict["rf_technical_indicators"]

    # Persist combined evaluation results (overwrite basic save with full content)
    RF_EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RF_EVAL_RESULTS_FILE, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation results (full) saved to {RF_EVAL_RESULTS_FILE}")

    return eval_results
