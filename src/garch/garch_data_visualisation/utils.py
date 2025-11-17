"""Utility functions for GARCH data visualization.

Contains helper functions for data preparation and plotting operations.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger, validate_temporal_split
from src.visualization import add_grid, add_zero_line, prepare_temporal_axis

_PLOT_LINEWIDTH = 0.5
_PLOT_ALPHA_MAIN = 0.7
_PLOT_ALPHA_GRID = 0.3
_COLOR_RETURNS = "blue"
_COLOR_ABSOLUTE_RETURNS = "orange"
_COLOR_SQUARED_RETURNS = "red"
_COLOR_ZERO_LINE = "black"


def prepare_x_axis(
    dates: Sequence[Any] | np.ndarray | pd.Series | pd.Index | None, length: int
) -> np.ndarray:
    """Prepare x-axis values from dates or indices.

    Wrapper for backward compatibility. Use prepare_temporal_axis directly.

    Args:
        dates: Optional sequence, array, or series of dates
        length: Length of data

    Returns:
        X-axis values array
    """
    return prepare_temporal_axis(dates, length)


def plot_returns_panel(ax: Any, x_vals: np.ndarray, returns: np.ndarray) -> None:
    """Plot returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        returns: Returns array
    """
    ax.plot(
        x_vals,
        returns,
        linewidth=_PLOT_LINEWIDTH,
        alpha=_PLOT_ALPHA_MAIN,
        color=_COLOR_RETURNS,
    )
    add_zero_line(ax, color=_COLOR_ZERO_LINE, linewidth=_PLOT_LINEWIDTH)
    ax.set_title("Rendements (returns)")
    ax.set_ylabel("Rendement")
    add_grid(ax, alpha=_PLOT_ALPHA_GRID)


def plot_absolute_returns_panel(ax: Any, x_vals: np.ndarray, abs_returns: np.ndarray) -> None:
    """Plot absolute returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        abs_returns: Absolute returns array
    """
    ax.plot(
        x_vals,
        abs_returns,
        linewidth=_PLOT_LINEWIDTH,
        alpha=_PLOT_ALPHA_MAIN,
        color=_COLOR_ABSOLUTE_RETURNS,
    )
    ax.set_title("Rendements absolus (|returns|)")
    ax.set_ylabel("|Rendement|")
    add_grid(ax, alpha=_PLOT_ALPHA_GRID)


def plot_squared_returns_panel(
    ax: Any, x_vals: np.ndarray, squared_returns: np.ndarray, has_dates: bool
) -> None:
    """Plot squared returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        squared_returns: Squared returns array
        has_dates: Whether dates are available
    """
    ax.plot(
        x_vals,
        squared_returns,
        linewidth=_PLOT_LINEWIDTH,
        alpha=_PLOT_ALPHA_MAIN,
        color=_COLOR_SQUARED_RETURNS,
    )
    ax.set_title("Rendements au carré (returns²)")
    ax.set_ylabel("Rendement²")
    ax.set_xlabel("Date" if has_dates else "Observation")
    add_grid(ax, alpha=_PLOT_ALPHA_GRID)


def _extract_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """Extract test split from DataFrame.

    Args:
        df: Input DataFrame with split column

    Returns:
        Test split DataFrame
    """
    logger = get_logger(__name__)

    df_test = df.loc[df["split"] == "test"].copy()
    if len(df_test) == 0:
        logger.warning("Test split is empty, using all data for returns plot")
        return df.copy()
    return df_test


def _ensure_weighted_return(df_test: pd.DataFrame) -> pd.DataFrame | None:
    """Ensure weighted_return column exists, converting from log returns if needed.

    Args:
        df_test: DataFrame to process

    Returns:
        DataFrame with weighted_return column or None if conversion impossible
    """
    logger = get_logger(__name__)

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


def prepare_test_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare test split DataFrame for returns plotting.

    Ensures temporal split validation to prevent look-ahead bias.
    Uses only test split data for visualization.

    Args:
        df: Input DataFrame

    Returns:
        Test DataFrame or None if data unavailable
    """
    logger = get_logger(__name__)

    if "split" not in df.columns:
        logger.warning("Dataset missing 'split' column, using all data for returns plot")
        df_test = df.copy()
    else:
        # Validate temporal split to prevent look-ahead bias
        try:
            validate_temporal_split(df, function_name="prepare_test_dataframe")
        except ValueError as e:
            logger.warning(f"Temporal split validation failed: {e}. Proceeding with caution.")
        df_test = _extract_test_split(df)

    df_test_result = _ensure_weighted_return(df_test)
    if df_test_result is None:
        return None
    return df_test_result


def extract_dates_from_dataframe(df_test: pd.DataFrame) -> pd.Series | None:
    """Extract dates column from DataFrame if available.

    Args:
        df_test: DataFrame to extract dates from

    Returns:
        Dates Series or None
    """
    if "date" not in df_test.columns:
        return None
    return df_test["date"]  # type: ignore[assignment]
