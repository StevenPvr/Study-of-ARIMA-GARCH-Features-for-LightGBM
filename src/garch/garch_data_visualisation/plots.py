"""GARCH data visualization and exploration module.

Implements methodology for GARCH data visualization:
3. Visualize returns and their squared/absolute values (volatility clustering)
4. Calculate autocorrelation of returns and squared returns
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from statsmodels.graphics.tsaplots import plot_acf

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATA_VISU_PLOTS_DIR,
)
from src.utils import get_logger

logger = get_logger(__name__)


def save_returns_and_squared_plots(
    returns: np.ndarray,
    *,
    dates: np.ndarray | pd.Series | None = None,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "garch_returns_clustering.png",
) -> Path:
    """
    Visualize returns, absolute returns, and squared returns for volatility clustering detection.

    Creates a 3-panel plot showing:
    - Returns time series
    - Absolute returns (|returns|)
    - Squared returns (returns^2)

    Args:
        returns: Array of returns (arithmetic returns)
        dates: Optional date array/series for x-axis
        outdir: Output directory for plot
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If returns array is empty or invalid
    """
    if returns is None:
        raise ValueError("returns cannot be None")

    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    returns_clean = np.asarray(returns, dtype=float)
    returns_clean = returns_clean[np.isfinite(returns_clean)]

    if returns_clean.size == 0:
        logger.warning("No finite returns to plot")
        return out_path

    abs_returns = np.abs(returns_clean)
    squared_returns = returns_clean**2

    # Create figure with 3 subplots
    fig = Figure(figsize=(12, 8), constrained_layout=True)
    canvas = FigureCanvas(fig)
    axes = fig.subplots(3, 1)

    # X-axis: dates if available, otherwise indices
    x_vals = _prepare_x_axis(dates, len(returns_clean))

    # Plot all three panels
    _plot_returns_panel(axes[0], x_vals, returns_clean)
    _plot_absolute_returns_panel(axes[1], x_vals, abs_returns)
    _plot_squared_returns_panel(axes[2], x_vals, squared_returns, dates is not None)

    fig.suptitle("Visualisation du clustering de volatilité", fontsize=14, fontweight="bold")

    canvas.print_png(str(out_path))
    logger.info(f"Saved returns clustering plot: {out_path}")
    return out_path


def _prepare_x_axis(dates: np.ndarray | pd.Series | None, length: int) -> np.ndarray:
    """Prepare x-axis values from dates or indices.

    Args:
        dates: Optional date array/series
        length: Length of data

    Returns:
        X-axis values array
    """
    if dates is not None:
        try:
            dates_clean = pd.to_datetime(dates)  # type: ignore[arg-type]
            return np.asarray(dates_clean.values, dtype=object)
        except Exception:
            return np.arange(length, dtype=float)
    return np.arange(length, dtype=float)


def _plot_returns_panel(ax: Any, x_vals: np.ndarray, returns: np.ndarray) -> None:
    """Plot returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        returns: Returns array
    """
    ax.plot(x_vals, returns, linewidth=0.5, alpha=0.7, color="blue")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_title("Rendements (returns)")
    ax.set_ylabel("Rendement")
    ax.grid(True, alpha=0.3)


def _plot_absolute_returns_panel(ax: Any, x_vals: np.ndarray, abs_returns: np.ndarray) -> None:
    """Plot absolute returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        abs_returns: Absolute returns array
    """
    ax.plot(x_vals, abs_returns, linewidth=0.5, alpha=0.7, color="orange")
    ax.set_title("Rendements absolus (|returns|)")
    ax.set_ylabel("|Rendement|")
    ax.grid(True, alpha=0.3)


def _plot_squared_returns_panel(
    ax: Any, x_vals: np.ndarray, squared_returns: np.ndarray, has_dates: bool
) -> None:
    """Plot squared returns panel.

    Args:
        ax: Matplotlib axis
        x_vals: X-axis values
        squared_returns: Squared returns array
        has_dates: Whether dates are available
    """
    ax.plot(x_vals, squared_returns, linewidth=0.5, alpha=0.7, color="red")
    ax.set_title("Rendements au carré (returns²)")
    ax.set_ylabel("Rendement²")
    ax.set_xlabel("Date" if has_dates else "Observation")
    ax.grid(True, alpha=0.3)


def plot_returns_autocorrelation(
    returns: np.ndarray,
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: Path | str = GARCH_DATA_VISU_PLOTS_DIR,
    filename: str = "garch_returns_autocorrelation.png",
) -> Path:
    """
    Plot autocorrelation of returns and squared returns side by side.

    This visualization helps identify:
    - Low autocorrelation in returns (typical for financial returns)
    - Significant autocorrelation in squared returns (motivates GARCH modeling)

    Args:
        returns: Returns array
        lags: Number of lags for ACF (must be positive)
        outdir: Output directory
        filename: Output filename

    Returns:
        Path to saved plot

    Raises:
        ValueError: If returns is None or lags is invalid
    """
    if returns is None:
        raise ValueError("returns cannot be None")
    if lags < 1:
        raise ValueError(f"lags must be positive, got {lags}")

    out_path = Path(outdir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    returns_clean = np.asarray(returns, dtype=float)
    returns_clean = returns_clean[np.isfinite(returns_clean)]

    if returns_clean.size < 3:
        logger.warning(f"Insufficient data for ACF plot (n={returns_clean.size})")
        return out_path

    squared_returns = returns_clean**2

    fig = Figure(figsize=(12, 5), constrained_layout=True)
    canvas = FigureCanvas(fig)
    axes = fig.subplots(1, 2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plot_acf(returns_clean, lags=lags, ax=axes[0], zero=False)
        plot_acf(squared_returns, lags=lags, ax=axes[1], zero=False)

    axes[0].set_title("ACF des rendements")
    axes[0].set_xlabel("Décalage (lag)")
    axes[0].set_ylabel("Autocorrélation")

    axes[1].set_title("ACF des rendements au carré")
    axes[1].set_xlabel("Décalage (lag)")
    axes[1].set_ylabel("Autocorrélation")

    fig.suptitle(
        "Autocorrélation des rendements et des rendements au carré",
        fontsize=14,
        fontweight="bold",
    )

    canvas.print_png(str(out_path))
    logger.info(f"Saved returns autocorrelation plot: {out_path}")
    return out_path


