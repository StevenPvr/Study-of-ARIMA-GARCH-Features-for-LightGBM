"""Residual diagnostics visualizations for ARIMA models.

This module contains visualizations for analyzing ARIMA model residuals:
- Residuals time series
- Residuals distribution (histogram with normal overlay)
- Q-Q plot for normality testing
- ACF of residuals with Ljung-Box test
- Comprehensive residuals dashboard
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from src.constants import (
    ACF_PACF_DEFAULT_LAGS,
    ACF_PACF_MIN_LAGS,
)
from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_statistics_textbox,
    add_zero_line,
    create_standard_figure,
    get_test_result_style,
    load_json_if_exists,
    plot_histogram_with_normal_overlay,
    plot_qq_normal,
    plot_series_with_train_test_split,
)

from .data_loading import load_residuals
from .plotting import _save_plot

logger = get_logger(__name__)
_PLOT_ALPHA_LIGHT = 0.3
_COLOR_NORMAL_FIT = "#A23B72"
_COLOR_RESIDUAL = "#2E86AB"
_COLOR_TEST = "#A23B72"
_COLOR_TRAIN = "#2E86AB"
_FIGURE_SIZE_DEFAULT = (10, 4)
_FONTSIZE_LABEL = 12
_FONTSIZE_TITLE = 14
_LINEWIDTH_DEFAULT = 0.6
_TEXTBOX_STYLE_DEFAULT = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
_RESIDUALS_HISTOGRAM_BINS = 50
_STATISTICS_PRECISION = 6


def _plot_residuals_timeseries_panel(
    ax: Axes,
    residuals: pd.Series,
    train_test_split_date: str | None,
) -> None:
    """Plot residuals time series on given axes."""
    plot_series_with_train_test_split(
        ax,
        residuals,
        train_test_split_date,
        train_color=_COLOR_TRAIN,
        test_color=_COLOR_TEST,
        linewidth=_LINEWIDTH_DEFAULT,
    )

    add_zero_line(ax)
    add_statistics_textbox(
        ax,
        residuals,
        position=(0.5, 1.02),
        statistics=["mean", "std", "n"],
        precision=_STATISTICS_PRECISION,
        fontsize=10,
    )

    ax.set_title(
        "Série temporelle des résidus ARIMA",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Residual", fontsize=_FONTSIZE_LABEL)
    ax.legend(loc="best", fontsize=10)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")


def _plot_residuals_histogram_panel(
    ax: Axes,
    residuals: pd.Series,
    normality_tests: dict | None,
    bins: int,
) -> tuple[float, float]:
    """Plot residuals histogram with normal overlay on given axes."""
    mean, std = plot_histogram_with_normal_overlay(
        ax,
        residuals,
        bins=bins,
        show_mean_line=True,
        hist_color=_COLOR_RESIDUAL,
        fit_color=_COLOR_NORMAL_FIT,
    )

    # Add statistics textbox with test results
    skewness = float(stats.skew(np.asarray(residuals)))
    kurtosis = float(stats.kurtosis(np.asarray(residuals)))

    test_text = ""
    if normality_tests:
        jb = normality_tests.get("jarque_bera", {})
        sw = normality_tests.get("shapiro_wilk", {})
        test_text = (
            f"\nJarque-Bera: stat={jb.get('statistic', 0):.4f}, "
            f"p={jb.get('p_value', 0):.4f}"
            f"\nShapiro-Wilk: stat={sw.get('statistic', 0):.4f}, "
            f"p={sw.get('p_value', 0):.4f}"
        )

    stats_text = (
        f"N = {len(residuals):,}\n"
        f"Mean = {mean:.6f}\n"
        f"Std Dev = {std:.6f}\n"
        f"Skewness = {skewness:.4f}\n"
        f"Kurtosis = {kurtosis:.4f}"
        f"{test_text}"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=_TEXTBOX_STYLE_DEFAULT,
    )

    ax.set_title(
        "Distribution des résidus ARIMA",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Residual", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Density", fontsize=_FONTSIZE_LABEL)
    ax.legend(loc="upper right", fontsize=10)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    return mean, std


def _plot_residuals_qq_panel(ax: Axes, residuals: pd.Series) -> float:
    """Plot Q-Q plot on given axes."""
    corr = plot_qq_normal(
        ax,
        residuals,
        color=_COLOR_RESIDUAL,
        add_reference_line=True,
        show_correlation=True,
    )

    ax.set_title(
        "Q-Q Plot des résidus ARIMA standardisés",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Quantiles théoriques (Normal)", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Quantiles de l'échantillon", fontsize=_FONTSIZE_LABEL)
    ax.legend(loc="upper left", fontsize=10)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    return corr


def _plot_residuals_acf_panel(
    ax: Axes,
    residuals: pd.Series,
    ljungbox_results: dict | None,
    lags: int,
) -> None:
    """Plot ACF of residuals on given axes."""
    effective_lags = lags
    if len(residuals) < lags + 1:
        logger.warning(
            f"Series length ({len(residuals)}) < requested lags ({lags}). "
            f"Using {len(residuals) - 1}"
        )
        effective_lags = max(ACF_PACF_MIN_LAGS, len(residuals) - 1)

    plot_acf(
        residuals,
        lags=effective_lags,
        ax=ax,
        zero=False,
        alpha=0.05,
    )

    ax.set_title(
        "Fonction d'autocorrélation des résidus ARIMA",
        fontsize=_FONTSIZE_TITLE,
        fontweight="bold",
    )
    ax.set_xlabel("Lag", fontsize=_FONTSIZE_LABEL)
    ax.set_ylabel("Autocorrelation", fontsize=_FONTSIZE_LABEL)
    ax.grid(alpha=_PLOT_ALPHA_LIGHT, linestyle="--")

    # Add Ljung-Box results
    if ljungbox_results:
        lb_pval = ljungbox_results.get("lb_pvalue", 1.0)
        lb_stat = ljungbox_results.get("lb_stat", 0)
        passes = "PASS" if lb_pval > 0.05 else "FAIL"
        style = get_test_result_style(lb_pval, alpha=0.05)

        text = (
            f"Ljung-Box Test (lag {effective_lags}):\n"
            f"Statistic: {lb_stat:.4f}\n"
            f"P-value: {lb_pval:.4f}\n"
            f"Result: {passes} (α=0.05)"
        )

        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=style,
        )


def plot_residuals_timeseries(
    predictions_file: str,
    output_file: str,
    train_test_split_date: str | None = None,
) -> None:
    """Plot ARIMA residuals time series with optional train/test split.

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.
        train_test_split_date: Optional date string to mark train/test boundary.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading residuals data from predictions file")

    # Load residuals as time series
    df = pd.read_csv(predictions_file, parse_dates=["date"], index_col="date")
    residuals_series = df["residual"].dropna()

    _, ax = create_standard_figure(figsize=(14, 5))
    _plot_residuals_timeseries_panel(ax, residuals_series, train_test_split_date)

    plt.tight_layout()
    _save_plot(output_file)

    mean_res = float(residuals_series.mean())
    std_res = float(residuals_series.std())
    logger.info("Residuals time series plot saved: mean=%.6f, std=%.6f", mean_res, std_res)


def plot_residuals_histogram(
    predictions_file: str,
    output_file: str,
    normality_tests_file: str | None = None,
    bins: int = _RESIDUALS_HISTOGRAM_BINS,
) -> None:
    """Plot histogram of residuals with fitted normal distribution overlay.

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.
        normality_tests_file: Optional path to normality_tests.json for test results.
        bins: Number of histogram bins.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading residuals data for histogram")

    residuals = load_residuals(predictions_file)
    normality_tests = load_json_if_exists(normality_tests_file)

    _, ax = create_standard_figure(figsize=_FIGURE_SIZE_DEFAULT)
    mean, std = _plot_residuals_histogram_panel(ax, residuals, normality_tests, bins)

    plt.tight_layout()
    _save_plot(output_file)

    skewness = float(stats.skew(np.asarray(residuals)))
    kurtosis = float(stats.kurtosis(np.asarray(residuals)))
    logger.info(
        "Residuals histogram saved: mean=%.6f, std=%.6f, skew=%.4f, kurtosis=%.4f",
        mean,
        std,
        skewness,
        kurtosis,
    )


def plot_residuals_qq(
    predictions_file: str,
    output_file: str,
) -> None:
    """Plot Q-Q (quantile-quantile) plot for residuals normality testing.

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading residuals data for Q-Q plot")

    residuals = load_residuals(predictions_file)

    _, ax = create_standard_figure(figsize=_FIGURE_SIZE_DEFAULT)
    corr = _plot_residuals_qq_panel(ax, residuals)

    plt.tight_layout()
    _save_plot(output_file)
    logger.info("Q-Q plot saved: correlation=%.4f", corr)


def plot_residuals_acf(
    predictions_file: str,
    output_file: str,
    ljungbox_file: str | None = None,
    lags: int = ACF_PACF_DEFAULT_LAGS,
) -> None:
    """Plot ACF of residuals with optional Ljung-Box test results.

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.
        ljungbox_file: Optional path to ljungbox_residuals.json for test results.
        lags: Number of lags to display.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing or lags invalid.
    """
    if lags <= 0:
        msg = f"lags must be positive, got {lags}"
        raise ValueError(msg)

    logger.info("Loading residuals data for ACF plot")

    residuals = load_residuals(predictions_file)
    ljungbox_results = load_json_if_exists(ljungbox_file)

    _, ax = create_standard_figure(figsize=_FIGURE_SIZE_DEFAULT)
    _plot_residuals_acf_panel(ax, residuals, ljungbox_results, lags)

    plt.tight_layout()
    _save_plot(output_file)
    logger.info("Residuals ACF plot saved with %d lags", lags)


def _create_comprehensive_dashboard_panels(
    residuals: pd.Series,
    normality_tests: dict | None,
    ljungbox_results: dict | None,
    train_test_split_date: str | None,
    lags: int,
) -> tuple[Figure, list[Axes]]:
    """Create 2x2 dashboard with all four diagnostic panels."""
    fig, axes = create_standard_figure(n_rows=2, n_cols=2, figsize=(16, 10))
    axes_flat = axes.flatten()

    # Top-left: Time series
    _plot_residuals_timeseries_panel(axes_flat[0], residuals, train_test_split_date)

    # Top-right: Histogram
    _plot_residuals_histogram_panel(axes_flat[1], residuals, normality_tests, bins=50)

    # Bottom-left: Q-Q plot
    _plot_residuals_qq_panel(axes_flat[2], residuals)

    # Bottom-right: ACF
    _plot_residuals_acf_panel(axes_flat[3], residuals, ljungbox_results, lags)

    return fig, axes_flat


def plot_comprehensive_residuals(
    predictions_file: str,
    output_file: str,
    normality_tests_file: str | None = None,
    ljungbox_file: str | None = None,
    train_test_split_date: str | None = None,
    lags: int = ACF_PACF_DEFAULT_LAGS,
) -> None:
    """Create a comprehensive 2x2 dashboard of residual diagnostics.

    Combines all residual diagnostics in one figure:
    - Top-left: Residuals time series
    - Top-right: Histogram with normal overlay
    - Bottom-left: Q-Q plot
    - Bottom-right: ACF of residuals

    Args:
        predictions_file: Path to rolling_predictions.csv with residual column.
        output_file: Path to save the plot.
        normality_tests_file: Optional path to normality_tests.json.
        ljungbox_file: Optional path to ljungbox_residuals.json.
        train_test_split_date: Optional date string for train/test split.
        lags: Number of lags for ACF.

    Raises:
        FileNotFoundError: If predictions_file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Creating comprehensive residuals dashboard")

    # Load data
    data_path = Path(predictions_file)
    if not data_path.exists():
        msg = f"Predictions file not found: {predictions_file}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    if "residual" not in df.columns:
        msg = "Column 'residual' not found in predictions file"
        raise ValueError(msg)

    residuals = df["residual"].dropna()

    if residuals.empty:
        msg = "No valid residuals found"
        raise ValueError(msg)

    # Load optional test results
    normality_tests = load_json_if_exists(normality_tests_file)
    ljungbox_results = load_json_if_exists(ljungbox_file)

    # Create dashboard
    fig, _ = _create_comprehensive_dashboard_panels(
        residuals,
        normality_tests,
        ljungbox_results,
        train_test_split_date,
        lags,
    )

    # Overall title
    fig.suptitle(
        "Diagnostics Complets des Résidus ARIMA",
        fontsize=_FONTSIZE_TITLE + 2,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.99))
    _save_plot(output_file)
    logger.info("Comprehensive residuals dashboard saved")
