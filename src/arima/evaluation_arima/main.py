"""CLI entry point for evaluation_arima module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import SARIMA_DATA_VISU_PLOTS_DIR, SARIMA_REFIT_EVERY_DEFAULT
from src.arima.evaluation_arima.evaluation_arima import (
    evaluate_model,
    save_evaluation_results,
)
from src.arima.evaluation_arima.save_data_for_garch import save_garch_dataset
from src.arima.optimisation_arima.optimisation_arima import load_train_test_data
from src.arima.data_visualisation.data_visualisation import (
    analyze_residuals_sarima_000,
    plot_rolling_forecast_sarima_000,
)
from src.arima.training_arima.training_arima import load_trained_model
from src.utils import get_logger

logger = get_logger(__name__)


def _extract_model_parameters_from_loaded_model(
    model_info: dict[str, Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Extract order and seasonal_order from loaded model metadata.

    Args:
        model_info: Dictionary with model parameters from loaded model

    Returns:
        Tuple of (order, seasonal_order)

    Raises:
        ValueError: If required keys are missing in model_info
    """
    required_keys = ["p", "d", "q", "P", "D", "Q", "s"]
    missing_keys = [k for k in required_keys if k not in model_info]
    if missing_keys:
        msg = (
            f"Loaded model metadata missing required keys: {missing_keys}. "
            "Model must be trained and saved with complete metadata."
        )
        raise ValueError(msg)

    order = (model_info["p"], model_info["d"], model_info["q"])
    seasonal_order = (model_info["P"], model_info["D"], model_info["Q"], model_info["s"])

    return order, seasonal_order


def _load_data_and_model() -> (
    tuple[Any, Any, Any, dict[str, Any], tuple[int, int, int], tuple[int, int, int, int]]
):
    """Load train/test data and trained model.

    Returns:
        Tuple of (train_series, test_series, fitted_model, model_info, order, seasonal_order)
    """
    logger.info("Loading train/test data...")
    train_series, test_series = load_train_test_data()

    logger.info("Loading trained model...")
    fitted_model, model_info = load_trained_model()

    order, seasonal_order = _extract_model_parameters_from_loaded_model(model_info)
    logger.info(f"Using loaded model parameters: order={order}, seasonal_order={seasonal_order}")

    return train_series, test_series, fitted_model, model_info, order, seasonal_order


def _generate_diagnostic_plots(
    test_series: Any,
    results: dict[str, Any],
    order: tuple[int, int, int],
) -> None:
    """Generate diagnostic plots for SARIMA evaluation.

    Args:
        test_series: Test time series
        results: Evaluation results dictionary
        order: SARIMA order (p, d, q)
    """
    try:
        test_series_copy = test_series.copy()
        preds = np.asarray(results["predictions"], dtype=float)
        acts = np.asarray(results["actuals"], dtype=float)
        plot_rolling_forecast_sarima_000(
            test_series=test_series_copy,
            actuals=acts,
            predictions=preds,
            sarima_order=order,
            metrics=results["metrics"],
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "rolling_forecast_sarima_000.png"),
        )
        analyze_residuals_sarima_000(
            test_series=test_series_copy,
            actuals=acts,
            predictions=preds,
            sarima_order=order,
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "residuals_analysis_sarima_000.png"),
        )
    except (ValueError, FileNotFoundError, AttributeError, RuntimeError) as ex:
        logger.warning("SARIMA plotting skipped due to: %s", ex)


def _run_ljung_box_analysis(results: dict[str, Any]) -> None:
    """Run Ljung–Box residual whiteness test and generate plot.

    Args:
        results: Evaluation results dictionary with 'actuals' and 'predictions'
    """
    try:
        from src.arima.evaluation_arima.evaluation_arima import (
            compute_residuals,
            ljung_box_on_residuals,
            plot_residuals_acf_with_ljungbox,
            save_ljung_box_results,
        )

        residuals = compute_residuals(results["actuals"], results["predictions"])
        lb_report = ljung_box_on_residuals(residuals)
        save_ljung_box_results(lb_report)
        plot_residuals_acf_with_ljungbox(residuals)
    except (ValueError, FileNotFoundError, AttributeError, RuntimeError) as ex:
        logger.warning("Ljung–Box analysis skipped due to: %s", ex)


def _log_final_metrics(results: dict[str, Any]) -> None:
    """Log final evaluation metrics.

    Args:
        results: Evaluation results dictionary with 'metrics' key
    """
    logger.info(f"RMSE: {results['metrics']['RMSE']:.6f}")
    logger.info(f"MAE: {results['metrics']['MAE']:.6f}")
    logger.info(f"MSE: {results['metrics']['MSE']:.6f}")


def main() -> None:
    """Main CLI function for SARIMA evaluation."""
    logger.info("=" * 60)
    logger.info("SARIMA EVALUATION MODULE")
    logger.info("=" * 60)

    (
        train_series,
        test_series,
        fitted_model,
        model_info,
        order,
        seasonal_order,
    ) = _load_data_and_model()

    logger.info("Evaluating model on test set...")
    results = evaluate_model(
        train_series,
        test_series,
        order,
        seasonal_order,
        model_info,
        refit_every=SARIMA_REFIT_EVERY_DEFAULT,
    )

    save_evaluation_results(results)
    save_garch_dataset(results, fitted_model=fitted_model)

    _generate_diagnostic_plots(test_series, results, order)
    _run_ljung_box_analysis(results)
    _log_final_metrics(results)

    logger.info("=" * 60)
    logger.info("SARIMA evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
