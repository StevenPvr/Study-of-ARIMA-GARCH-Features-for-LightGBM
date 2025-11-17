"""Pre-modeling visualizations for ARIMA analysis.

This module contains all visualizations used before model fitting:
- Time series plots
- Stationarity analysis
- Seasonality decomposition
- ACF/PACF analysis
- Distribution of log returns
"""

from __future__ import annotations

from typing import cast

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.arima.stationnarity_check.stationnarity_check import evaluate_stationarity
from src.constants import (
    ACF_PACF_DEFAULT_LAGS,
    ACF_PACF_MIN_LAGS,
    STATIONARITY_ROLLING_WINDOW_DEFAULT,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_legend,
    add_zero_line,
    create_standard_figure,
    plot_histogram_with_normal_overlay,
)

from .data_loading import load_and_validate_data
from .plotting import (
    _format_date_axis,
    _save_plot,
    add_stationarity_text_box,
    format_stationarity_test_text,
    plot_stationarity_series_with_bands,
)

logger = get_logger(__name__)
_PLOT_ALPHA_DEFAULT = 0.8
_PLOT_ALPHA_LIGHT = 0.3
_COLOR_NORMAL_FIT = "#A23B72"
_COLOR_RESIDUAL = "#2E86AB"
_COLOR_SERIES_ORIGINAL = "blue"
_FIGURE_SIZE_ACF_PACF = (14, 4)
_FIGURE_SIZE_DEFAULT = (10, 4)
_FIGURE_SIZE_STATIONARITY = (16, 12)
_FIGURE_SIZE_WEIGHTED_SERIES = (18, 6)
_FONTSIZE_AXIS = 10
_FONTSIZE_LABEL = 12
_FONTSIZE_SUBTITLE = 12
_FONTSIZE_TITLE = 14
_LINEWIDTH_BOLD = 1.5
_LINEWIDTH_DEFAULT = 0.6
_TEXTBOX_STYLE_DEFAULT = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
_DISTRIBUTION_HISTOGRAM_BINS = 100
_SEASONAL_RESAMPLE_FREQ_WEEKLY = "W"


def plot_weighted_series(
    data_file: str,
    output_file: str,
) -> None:
    """Plot the time series of weighted log-returns.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    logger.info("Loading weighted log-returns data")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    # Resample to weekly for readability
    weighted_series_resampled = weighted_series.resample(_SEASONAL_RESAMPLE_FREQ_WEEKLY).mean()

    _, ax = create_standard_figure(figsize=_FIGURE_SIZE_WEIGHTED_SERIES)
    ax.plot(
        weighted_series_resampled.index,
        weighted_series_resampled,
        linewidth=_LINEWIDTH_DEFAULT,
        alpha=_PLOT_ALPHA_DEFAULT,
        color=_COLOR_SERIES_ORIGINAL,
    )
    add_zero_line(ax)
    ax.set_title(
        "Rendements logarithmiques pondérés du portefeuille (10 ans)",
        fontsize=_FONTSIZE_TITLE,
    )
    ax.set_ylabel("Log-return", fontsize=_FONTSIZE_LABEL)
    ax.set_xlabel("Date", fontsize=_FONTSIZE_LABEL)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    _format_date_axis(ax)
    plt.tight_layout()
    _save_plot(output_file)


def plot_log_returns_distribution(
    data_file: str,
    output_file: str,
    bins: int = _DISTRIBUTION_HISTOGRAM_BINS,
) -> None:
    """Plot histogram of log returns with fitted normal distribution overlay.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        bins: Number of histogram bins.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If required column is missing or data is empty.
    """
    logger.info("Loading weighted log-returns data for distribution plot")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    returns = aggregated_returns["weighted_log_return"].dropna()

    if returns.empty:
        msg = "No valid data found in weighted_log_return column"
        raise ValueError(msg)

    _, ax = create_standard_figure(figsize=_FIGURE_SIZE_DEFAULT)

    mean, std = plot_histogram_with_normal_overlay(
        ax,
        returns,
        bins=bins,
        show_mean_line=True,
        hist_color=_COLOR_RESIDUAL,
        fit_color=_COLOR_NORMAL_FIT,
    )

    # Add statistics
    skewness = float(stats.skew(np.asarray(returns)))
    kurtosis = float(stats.kurtosis(np.asarray(returns)))

    stats_text = (
        f"N = {len(returns):,}\n"
        f"Mean = {mean:.6f}\n"
        f"Std Dev = {std:.6f}\n"
        f"Skewness = {skewness:.4f}\n"
        f"Kurtosis = {kurtosis:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=_TEXTBOX_STYLE_DEFAULT,
    )

    ax.set_title(
        "Distribution des rendements logarithmiques pondérés",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Log-return", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Density", fontsize=_FONTSIZE_LABEL)
    ax.legend(loc="upper right", fontsize=10)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)
    logger.info(
        "Distribution plot saved: mean=%.6f, std=%.6f, skew=%.4f, kurtosis=%.4f",
        mean,
        std,
        skewness,
        kurtosis,
    )


def plot_acf_pacf(
    data_file: str,
    output_file: str,
    lags: int = ACF_PACF_DEFAULT_LAGS,
) -> None:
    """Plot autocorrelation (ACF) and partial autocorrelation (PACF) functions.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        lags: Number of lags to display (excluding lag 0).

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If lags is invalid or required column is missing.
    """
    if lags <= 0:
        msg = f"lags must be positive, got {lags}"
        raise ValueError(msg)

    logger.info("Loading weighted log-returns data for ACF/PACF")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series = aggregated_returns["weighted_log_return"].dropna()

    if len(weighted_series) < lags + 1:
        logger.warning(
            f"Series length ({len(weighted_series)}) < requested lags ({lags}). "
            f"Using {len(weighted_series) - 1}"
        )
        lags = max(ACF_PACF_MIN_LAGS, len(weighted_series) - 1)

    _, axes = create_standard_figure(n_rows=1, n_cols=2, figsize=_FIGURE_SIZE_ACF_PACF)

    # ACF
    plot_acf(
        weighted_series,
        lags=lags,
        ax=axes[0],
        zero=False,
        alpha=0.05,
        fft=False,  # Use direct method for numerical stability
    )
    axes[0].set_title(
        "Fonction d'autocorrélation (ACF)",
        fontsize=_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[0].set_xlabel("Lag", fontsize=_FONTSIZE_AXIS)
    axes[0].set_ylabel("Autocorrelation", fontsize=_FONTSIZE_AXIS)
    axes[0].grid(alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    # PACF
    plot_pacf(
        weighted_series,
        lags=lags,
        ax=axes[1],
        method="ywm",
        zero=False,
        alpha=0.05,
    )
    axes[1].set_title(
        "Fonction d'autocorrélation partielle (PACF)",
        fontsize=_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    axes[1].set_xlabel("Lag", fontsize=_FONTSIZE_AXIS)
    axes[1].set_ylabel("Partial Autocorrelation", fontsize=_FONTSIZE_AXIS)
    axes[1].grid(alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    plt.tight_layout()
    _save_plot(output_file)


def _validate_stationarity_params(rolling_window: int, alpha: float) -> None:
    """Validate stationarity plotting parameters."""
    if rolling_window <= 0:
        msg = f"rolling_window must be positive, got {rolling_window}"
        raise ValueError(msg)

    if not 0 < alpha < 1:
        msg = f"alpha must be in (0, 1), got {alpha}"
        raise ValueError(msg)


def _load_stationarity_data(
    data_file: str,
    rolling_window: int,
) -> tuple[pd.Series, int]:
    """Load and prepare data for stationarity analysis."""
    logger.info("Loading weighted log-returns data for stationarity analysis")
    aggregated_returns = load_and_validate_data(data_file, "weighted_log_return")
    weighted_series_raw = aggregated_returns["weighted_log_return"].dropna()

    if not isinstance(weighted_series_raw, pd.Series):
        msg = f"Expected Series, got {type(weighted_series_raw)}"
        raise ValueError(msg)

    weighted_series = cast(pd.Series, weighted_series_raw)

    if len(weighted_series) < rolling_window:
        logger.warning(
            f"Series length ({len(weighted_series)}) < rolling_window ({rolling_window}). "
            f"Using {len(weighted_series)}"
        )
        rolling_window = len(weighted_series)

    return weighted_series, rolling_window






def plot_stationarity_timeseries_with_bands(
    data_file: str,
    output_file: str,
    rolling_window: int = STATIONARITY_ROLLING_WINDOW_DEFAULT,
    alpha: float = 0.05,
) -> None:
    """Plot time series with rolling mean and standard deviation bands.

    Publication-ready plot showing the original time series with rolling statistics
    confidence bands. Optimized for academic/research presentation.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_file: Path to save the plot.
        rolling_window: Window size for rolling statistics.
        alpha: Significance level for stationarity tests.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If rolling_window is invalid, alpha is not in (0, 1),
            or required column is missing.
    """
    _validate_stationarity_params(rolling_window, alpha)
    weighted_series, adjusted_window = _load_stationarity_data(data_file, rolling_window)

    rolling_mean = cast(
        pd.Series, weighted_series.rolling(window=adjusted_window, center=False).mean()
    )
    rolling_std = cast(
        pd.Series, weighted_series.rolling(window=adjusted_window, center=False).std()
    )

    logger.info("Running ADF and KPSS tests for stationarity")
    stationarity_report = evaluate_stationarity(weighted_series, alpha=alpha)

    # Create publication-quality figure
    fig, ax = create_standard_figure(figsize=_FIGURE_SIZE_STATIONARITY)
    plot_stationarity_series_with_bands(
        ax, weighted_series, rolling_mean, rolling_std, adjusted_window
    )

    # Add stationarity test results as text box
    adf_res = stationarity_report.adf
    kpss_res = stationarity_report.kpss
    adf_lags = adf_res["lags"]
    kpss_lags = kpss_res["lags"]
    if adf_lags is None or kpss_lags is None:
        msg = "Stationarity test lags cannot be None"
        raise ValueError(msg)
    test_text = format_stationarity_test_text(
        alpha,
        adf_res["statistic"],
        adf_res["p_value"],
        adf_lags,
        kpss_res["statistic"],
        kpss_res["p_value"],
        kpss_lags,
        "STATIONNAIRE" if stationarity_report.stationary else "NON STATIONNAIRE",
    )
    add_stationarity_text_box(ax, test_text)

    # Add overall verdict in title
    verdict_color = "green" if stationarity_report.stationary else "red"
    verdict_text = "STATIONNAIRE" if stationarity_report.stationary else "NON STATIONNAIRE"
    ax.set_title(
        f"Analyse de stationnarité - Verdict: {verdict_text}",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
        color=verdict_color,
        pad=20,
    )

    plt.tight_layout()
    _save_plot(output_file)

    logger.info(
        "Stationarity time series plot saved: stationary=%s (ADF p-value=%.6f, KPSS p-value=%.6f)",
        stationarity_report.stationary,
        stationarity_report.adf["p_value"],
        stationarity_report.kpss["p_value"],
    )






def plot_stationarity(
    data_file: str,
    output_dir: str,
    rolling_window: int = STATIONARITY_ROLLING_WINDOW_DEFAULT,
    alpha: float = 0.05,
) -> None:
    """Generate publication-ready stationarity analysis plot.

    Creates a high-quality plot showing time series with rolling statistics bands
    optimized for academic publication.

    Args:
        data_file: Path to weighted log-returns CSV file.
        output_dir: Directory path to save the plot.
        rolling_window: Window size for rolling statistics.
        alpha: Significance level for stationarity tests.

    Raises:
        FileNotFoundError: If data_file does not exist.
        ValueError: If rolling_window is invalid, alpha is not in (0, 1),
            or required column is missing.
    """
    from pathlib import Path

    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating stationarity analysis plot for publication")

    # Generate the plot
    plot_stationarity_timeseries_with_bands(
        data_file=data_file,
        output_file=str(output_path / "stationarity_timeseries_with_bands.png"),
        rolling_window=rolling_window,
        alpha=alpha,
    )

    logger.info("Stationarity analysis plot generated successfully")
