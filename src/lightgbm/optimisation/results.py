"""Results saving and logging utilities for LightGBM optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.path import LIGHTGBM_OPTIMIZATION_RESULTS_FILE
from src.utils import get_logger, save_json_pretty

logger = get_logger(__name__)


def _build_optimization_results_dict(
    results_complete: dict[str, Any],
    results_without_insights: dict[str, Any],
    results_sigma_plus_base: dict[str, Any] | None = None,
    results_log_volatility_only: dict[str, Any] | None = None,
    results_technical: dict[str, Any] | None = None,
    results_technical_only: dict[str, Any] | None = None,
    results_technical_plus_insights: dict[str, Any] | None = None,
    results_insights_only: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build optimization results dictionary with all available results.

    Args:
        results_complete: Results for complete dataset.
        results_without_insights: Results for dataset without insights.
        results_sigma_plus_base: Optional results for sigma-plus-base dataset.
        results_log_volatility_only: Optional results for log-volatility-only dataset.
        results_technical: Optional results for technical indicators dataset.
        results_insights_only: Optional results for insights-only dataset.

    Returns:
        Dictionary containing all optimization results.
    """
    results: dict[str, Any] = {
        "lightgbm_dataset_complete": results_complete,
        "lightgbm_dataset_without_insights": results_without_insights,
    }
    if results_sigma_plus_base is not None:
        results["lightgbm_dataset_sigma_plus_base"] = results_sigma_plus_base
    if results_log_volatility_only is not None:
        results["lightgbm_dataset_log_volatility_only"] = results_log_volatility_only
    if results_technical is not None:
        results["lightgbm_dataset_technical_indicators"] = results_technical
    if results_technical_only is not None:
        results["lightgbm_dataset_technical_only"] = results_technical_only
    if results_technical_plus_insights is not None:
        results["lightgbm_dataset_technical_plus_insights"] = results_technical_plus_insights
    if results_insights_only is not None:
        results["lightgbm_dataset_insights_only"] = results_insights_only

    return results


def save_optimization_results(
    results_complete: dict[str, Any],
    results_without_insights: dict[str, Any],
    output_path: Path = LIGHTGBM_OPTIMIZATION_RESULTS_FILE,
    *,
    results_sigma_plus_base: dict[str, Any] | None = None,
    results_log_volatility_only: dict[str, Any] | None = None,
    results_technical: dict[str, Any] | None = None,
    results_technical_only: dict[str, Any] | None = None,
    results_technical_plus_insights: dict[str, Any] | None = None,
    results_insights_only: dict[str, Any] | None = None,
) -> None:
    """Save optimization results to JSON file.

    Args:
        results_complete: Results for complete dataset.
        results_without_insights: Results for dataset without insights.
        output_path: Path to save results JSON file.
        results_sigma_plus_base: Optional results for sigma-plus-base dataset.
        results_log_volatility_only: Optional results for log-volatility-only dataset.
        results_technical: Optional results for technical indicators dataset.
        results_insights_only: Optional results for insights-only dataset.
    """
    results = _build_optimization_results_dict(
        results_complete=results_complete,
        results_without_insights=results_without_insights,
        results_sigma_plus_base=results_sigma_plus_base,
        results_log_volatility_only=results_log_volatility_only,
        results_technical=results_technical,
        results_technical_only=results_technical_only,
        results_technical_plus_insights=results_technical_plus_insights,
        results_insights_only=results_insights_only,
    )

    save_json_pretty(results, output_path)

    logger.info(f"Optimization results saved to {output_path}")


def _log_dataset_results(dataset_name: str, results: dict[str, Any]) -> None:
    """Log results for a single dataset.

    Args:
        dataset_name: Name of the dataset.
        results: Results dictionary.
    """
    logger.info(f"\n{dataset_name}:")
    logger.info(f"  Best RMSE (CV): {results['best_rmse_cv']:.6f} (log-volatility)")
    logger.info(f"  Best params: {results['best_params']}")


def _log_optimization_summary(
    results_complete: dict[str, Any],
    results_without: dict[str, Any],
    results_sigma_plus_base: dict[str, Any] | None = None,
    results_log_volatility: dict[str, Any] | None = None,
    results_technical: dict[str, Any] | None = None,
    results_technical_only: dict[str, Any] | None = None,
    results_technical_plus_insights: dict[str, Any] | None = None,
    results_insights_only: dict[str, Any] | None = None,
) -> None:
    """Log optimization summary results.

    Args:
        results_complete: Results for complete dataset.
        results_without: Results for dataset without insights.
        results_sigma_plus_base: Optional results for sigma-plus-base dataset.
        results_log_volatility: Optional results for log-volatility-only dataset.
        results_technical: Optional results for technical indicators dataset.
        results_insights_only: Optional results for insights-only dataset.
    """
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 70)

    _log_dataset_results("Complete Dataset", results_complete)
    _log_dataset_results("Without Insights Dataset", results_without)

    if results_sigma_plus_base is not None:
        _log_dataset_results("Sigma-Plus-Base Dataset", results_sigma_plus_base)

    if results_log_volatility is not None:
        _log_dataset_results("Log Volatility Only Dataset", results_log_volatility)

    if results_technical is not None:
        _log_dataset_results("Technical Indicators Dataset", results_technical)
    if results_technical_only is not None:
        _log_dataset_results("Technical Only (no target lags)", results_technical_only)
    if results_technical_plus_insights is not None:
        _log_dataset_results(
            "Technical + Insights (no target lags)", results_technical_plus_insights
        )

    if results_insights_only is not None:
        _log_dataset_results("Insights-Only Dataset", results_insights_only)

    logger.info("\n" + "=" * 70)
