"""CLI for visualizing returns and squared/absolute returns.

Generates plots for volatility clustering and autocorrelation of returns and squared returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATA_VISU_PLOTS_DIR,
    GARCH_DATASET_FILE,
    GARCH_RETURNS_CLUSTERING_PLOT,
)
from src.garch.garch_data_visualisation.plots import (
    plot_returns_autocorrelation,
    save_returns_and_squared_plots,
)
from src.garch.structure_garch.detection import load_garch_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def _prepare_test_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare test split DataFrame for returns plotting.

    Args:
        df: Input DataFrame

    Returns:
        Test DataFrame or None if data unavailable
    """
    if "split" not in df.columns:
        logger.warning("Dataset missing 'split' column, using all data for returns plot")
        df_test = df.copy()
    else:
        df_test = df.loc[df["split"] == "test"].copy()
        if len(df_test) == 0:
            logger.warning("Test split is empty, using all data for returns plot")
            df_test = df.copy()

    # Ensure weighted_return exists
    if "weighted_return" not in df_test.columns:
        if "weighted_log_return" in df_test.columns:
            log_returns = np.asarray(df_test["weighted_log_return"].values, dtype=float)
            # Protect against overflow: clip extreme values
            log_returns_clipped = np.clip(log_returns, -10.0, 10.0)
            df_test["weighted_return"] = np.expm1(log_returns_clipped)  # type: ignore[index]
        else:
            logger.warning(
                "Neither 'weighted_return' nor 'weighted_log_return' found, skipping returns plot"
            )
            return None

    return df_test


def _extract_dates_from_dataframe(df_test: pd.DataFrame) -> pd.Series | None:
    """Extract dates column from DataFrame if available.

    Args:
        df_test: DataFrame to extract dates from

    Returns:
        Dates Series or None
    """
    if "date" not in df_test.columns:
        return None
    return df_test["date"]  # type: ignore[assignment]


def _generate_returns_plot(df: pd.DataFrame) -> None:
    """Generate returns clustering plot.

    Args:
        df: DataFrame with returns data
    """
    df_test = _prepare_test_dataframe(df)
    if df_test is None or "weighted_return" not in df_test.columns:
        return

    returns_array = df_test["weighted_return"].to_numpy().astype(float)
    # Filter out non-finite values
    returns_finite = returns_array[np.isfinite(returns_array)]
    if returns_finite.size == 0:
        logger.warning("No finite returns found, skipping returns plot")
        return

    try:
        dates_param = _extract_dates_from_dataframe(df_test)
        save_returns_and_squared_plots(
            returns_array,
            dates=dates_param,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename=GARCH_RETURNS_CLUSTERING_PLOT.name,
        )
    except Exception as e:
        logger.error(f"Failed to generate returns clustering plot: {e}")
        raise


def _generate_autocorrelation_plot(df: pd.DataFrame) -> None:
    """Generate autocorrelation plots for returns and squared returns.

    Args:
        df: DataFrame with returns data
    """
    df_test = _prepare_test_dataframe(df)
    if df_test is None or "weighted_return" not in df_test.columns:
        return

    returns_array = df_test["weighted_return"].to_numpy().astype(float)
    # Filter out non-finite values
    returns_finite = returns_array[np.isfinite(returns_array)]
    if returns_finite.size == 0:
        logger.warning("No finite returns found, skipping autocorrelation plot")
        return

    try:
        plot_returns_autocorrelation(
            returns_array,
            lags=GARCH_ACF_LAGS_DEFAULT,
            outdir=GARCH_DATA_VISU_PLOTS_DIR,
            filename="garch_returns_autocorrelation.png",
        )
    except Exception as e:
        logger.error(f"Failed to generate autocorrelation plot: {e}")
        raise


def main() -> None:
    """Create returns and squared returns visualizations for test split."""
    logger.info("=" * 60)
    logger.info("GARCH VISUALIZATION (returns, |returns|, returns^2, ACF(returns), ACF(returns^2))")
    logger.info("=" * 60)

    try:
        df = load_garch_dataset(str(GARCH_DATASET_FILE))
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load GARCH dataset: {e}")
        raise

    # Generate returns clustering plot (point 3)
    _generate_returns_plot(df)

    # Generate autocorrelation plots (point 4)
    _generate_autocorrelation_plot(df)

    logger.info("Saved visualization plots to: %s", GARCH_DATA_VISU_PLOTS_DIR)


if __name__ == "__main__":
    main()
