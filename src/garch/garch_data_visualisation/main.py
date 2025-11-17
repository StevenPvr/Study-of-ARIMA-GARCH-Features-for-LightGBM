"""CLI for visualizing returns and squared/absolute returns.

Generates plots for volatility clustering and autocorrelation of returns and squared returns.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_data_visualisation.plots import (
    plot_residuals_distribution,
    plot_squared_residuals_acf,
    test_arch_effect,
)
from src.garch.garch_data_visualisation.utils import (
    extract_dates_from_dataframe,
    prepare_test_dataframe,
)
from src.garch.structure_garch.utils import load_garch_dataset
from src.path import GARCH_DATA_VISU_PLOTS_DIR, GARCH_DATASET_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_returns_plot(df: pd.DataFrame) -> None:
    """Generate returns clustering plot.

    Args:
        df: DataFrame with returns data

    Raises:
        ValueError: If test dataframe is None or missing required columns
        ValueError: If no finite returns found
    """
    df_test = prepare_test_dataframe(df)
    if df_test is None:
        raise ValueError("Failed to prepare test dataframe for returns plot")
    if "weighted_return" not in df_test.columns:
        raise ValueError("DataFrame missing 'weighted_return' column")

    returns_array = df_test["weighted_return"].to_numpy().astype(float)
    # Filter out non-finite values
    returns_finite = returns_array[np.isfinite(returns_array)]
    if returns_finite.size == 0:
        raise ValueError("No finite returns found in test dataframe")



def _generate_residuals_distribution_plot(df: pd.DataFrame) -> None:
    """Generate distribution plot for ARIMA residuals.

    Args:
        df: DataFrame with ARIMA residuals

    Raises:
        ValueError: If dataframe is missing required columns
        ValueError: If no finite residuals found
    """
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        col_name = _find_residual_column(df)
        residuals_array = df[col_name].to_numpy().astype(float)
    except ValueError:
        msg = (
            "DataFrame missing residual column ('arima_resid' or 'sarima_resid'). "
            "Cannot compute residuals for visualization."
        )
        raise ValueError(msg) from None
    # Filter out non-finite values
    residuals_finite = residuals_array[np.isfinite(residuals_array)]
    if residuals_finite.size == 0:
        raise ValueError("No finite residuals found in dataframe")

    try:
        plot_residuals_distribution(
            residuals_array,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename="arima_residuals_distribution.png",
        )
    except Exception as e:
        logger.error(f"Failed to generate residuals distribution plot: {e}")
        raise


def main() -> None:
    """Create returns and squared returns visualizations for test split."""
    logger.info("=" * 60)
    logger.info(
        "GARCH VISUALIZATION (returns, |returns|, returns^2, ACF(returns), "
        "ACF(returns^2), ARIMA residuals distribution, ACF(squared residuals), ARCH test)"
    )
    logger.info("=" * 60)

    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load GARCH dataset: {e}")
        raise

    # Generate returns clustering plot (point 3)
    _generate_returns_plot(df)

    # Generate ARIMA residuals distribution plot
    _generate_residuals_distribution_plot(df)

    # Generate ACF of squared residuals plot
    _generate_squared_residuals_acf_plot(df)

    logger.info("Saved visualization plots to: %s", GARCH_DATA_VISU_PLOTS_DIR)


def _generate_squared_residuals_acf_plot(df: pd.DataFrame) -> None:
    """Generate ACF plot for squared ARIMA residuals and ARCH effect test.

    Args:
        df: DataFrame with ARIMA residuals

    Raises:
        ValueError: If dataframe is missing required columns
        ValueError: If no finite residuals found
    """
    from src.garch.structure_garch.utils import _find_residual_column

    try:
        col_name = _find_residual_column(df)
        residuals_array = df[col_name].to_numpy().astype(float)
    except ValueError:
        msg = (
            "DataFrame missing residual column ('arima_resid' or 'sarima_resid'). "
            "Cannot compute ACF of squared residuals for visualization."
        )
        raise ValueError(msg) from None

    # Filter out non-finite values
    residuals_finite = residuals_array[np.isfinite(residuals_array)]
    if residuals_finite.size == 0:
        raise ValueError("No finite residuals found in dataframe")

    # Perform ARCH effect test
    arch_test = test_arch_effect(residuals_finite)
    logger.info(
        "ARCH-LM test: LM-stat=%.4f, p-value=%.4f, ARCH effect present: %s",
        arch_test["lm_stat"],
        arch_test["p_value"],
        arch_test["arch_present"],
    )

    try:
        plot_squared_residuals_acf(
            residuals_array,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename="arima_squared_residuals_acf.png",
        )
    except Exception as e:
        logger.error(f"Failed to generate squared residuals ACF plot: {e}")
        raise


if __name__ == "__main__":
    main()
