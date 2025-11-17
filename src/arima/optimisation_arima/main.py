"""ARIMA optimization module - automatic execution."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Add project root to Python path for direct execution.
# This must be done before importing src modules.
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.arima.optimisation_arima.optimisation_arima import (  # noqa: E402
    load_train_data,
    optimize_arima_models,
)
from src.constants import (  # noqa: E402
    ARIMA_OPTIMIZATION_N_SPLITS,
    ARIMA_OPTIMIZATION_N_TRIALS,
    ARIMA_REFIT_EVERY_OPTIONS,
)
from src.path import ARIMA_ARTIFACTS_DIR, WEIGHTED_LOG_RETURNS_SPLIT_FILE  # noqa: E402

# Default configuration values
DEFAULT_VALUE_COL = "weighted_log_return"
DEFAULT_REFIT_EVERY = ARIMA_REFIT_EVERY_OPTIONS[0]  # Use first option (1)




def _format_model_info(label: str, model: Dict[str, Any], primary: str, secondary: str) -> str:
    """Format model information for display.

    Args:
        label: Model label (e.g., "Best AIC model").
        model: Model dictionary with params and metrics.
        primary: Primary metric name (e.g., "aic").
        secondary: Secondary metric name (e.g., "bic").

    Returns:
        Formatted string with model information.
    """
    params = model.get("params")
    primary_val = model.get(primary, float("nan"))
    secondary_val = model.get(secondary, float("nan"))
    return (
        f"{label}: params={params} | {primary.upper()}={primary_val:.6f} | "
        f"{secondary.upper()}={secondary_val:.6f}"
    )


def print_best_models(
    best_aic: Dict[str, Any], best_bic: Optional[Dict[str, Any]], out_dir: Path
) -> None:
    """Print best AIC and BIC models.

    Args:
        best_aic: Dictionary with best AIC model parameters and metrics.
        best_bic: Dictionary with best BIC model parameters and metrics, or None if BIC
        optimization is disabled.
        out_dir: Output directory path.
    """
    separator_length = 80
    print("=" * separator_length)
    print(_format_model_info("Best AIC model", best_aic, "aic", "bic"))
    if best_bic is not None:
        print(_format_model_info("Best BIC model", best_bic, "bic", "aic"))
    print("Results saved to:", out_dir)
    print("=" * separator_length)


def main() -> None:
    """Main entry point for ARIMA optimization.

    Loads data, runs optimization with default parameters, and prints results.
    """
    print("Starting ARIMA optimization with default parameters...")
    print(f"Data file: {WEIGHTED_LOG_RETURNS_SPLIT_FILE}")
    print(f"Value column: {DEFAULT_VALUE_COL}")
    print(f"Refit every: {DEFAULT_REFIT_EVERY}")
    print(f"Trials: {ARIMA_OPTIMIZATION_N_TRIALS}")
    print(f"Splits: {ARIMA_OPTIMIZATION_N_SPLITS}")
    print(f"Output directory: {ARIMA_ARTIFACTS_DIR}")
    print("=" * 50)

    train = load_train_data(
        csv_path=WEIGHTED_LOG_RETURNS_SPLIT_FILE,
        value_col=DEFAULT_VALUE_COL,
        date_col=None,
    )

    _, best_aic, best_bic = optimize_arima_models(
        train_series=train,
        test_series=None,
        n_trials=ARIMA_OPTIMIZATION_N_TRIALS,
        n_jobs=1,
        backtest_n_splits=ARIMA_OPTIMIZATION_N_SPLITS,
        backtest_test_size=None,
        backtest_refit_every=DEFAULT_REFIT_EVERY,
        out_dir=ARIMA_ARTIFACTS_DIR,
    )

    print_best_models(best_aic, best_bic, ARIMA_ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
