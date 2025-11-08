"""CLI entry point for data_visualisation module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.data_visualisation.data_visualisation import (
    plot_acf_pacf,
    plot_seasonality_daily,
    plot_seasonality_full_period,
    plot_seasonality_monthly,
    plot_weighted_series,
)
from src.constants import SARIMA_DATA_VISU_PLOTS_DIR, WEIGHTED_LOG_RETURNS_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main CLI function to visualize S&P 500 data."""
    # Ensure the output directory exists
    SARIMA_DATA_VISU_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        plot_weighted_series(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "weighted_log_returns_series.png"),
        )
    except Exception as e:
        logger.error(f"Failed to plot weighted series: {e}", exc_info=True)
        raise

    try:
        plot_acf_pacf(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "acf_pacf.png"),
        )
    except Exception as e:
        logger.error(f"Failed to plot ACF/PACF: {e}", exc_info=True)
        raise

    # Plot seasonality for full period (weekly resampling)
    try:
        plot_seasonality_full_period(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "seasonal_full_period.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot seasonality (weekly): {e}", exc_info=True)

    # Plot seasonality for daily data (weekly pattern - 5 business days)
    try:
        plot_seasonality_daily(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "seasonal_daily.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot seasonality (daily): {e}", exc_info=True)

    # Plot seasonality for monthly data (annual pattern - 12 months)
    try:
        plot_seasonality_monthly(
            data_file=str(WEIGHTED_LOG_RETURNS_FILE),
            output_file=str(SARIMA_DATA_VISU_PLOTS_DIR / "seasonal_monthly.png"),
        )
    except Exception as e:
        logger.warning(f"Failed to plot seasonality (monthly): {e}", exc_info=True)


if __name__ == "__main__":
    main()
