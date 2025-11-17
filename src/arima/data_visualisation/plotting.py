"""Plotting utilities for visualization."""

from __future__ import annotations

from matplotlib.axes import Axes
import pandas as pd

from src.utils import get_logger
from src.visualization import (
    add_grid,
    add_legend,
    add_zero_line,
    format_date_axis,
    save_plot_wrapper,
)

logger = get_logger(__name__)
_PLOT_ALPHA_DEFAULT = 0.8
_PLOT_ALPHA_FILL = 0.2
_PLOT_ALPHA_LIGHT = 0.3
_PLOT_ALPHA_MEDIUM = 0.7
_FONTSIZE_AXIS = 10
_FONTSIZE_SUBTITLE = 12
_FONTSIZE_TEXT = 9
_LINEWIDTH_BOLD = 1.5
_LINEWIDTH_THIN = 0.8
_STATIONARITY_SEPARATOR_LENGTH = 60
_STATIONARITY_TEXT_BOX_X = 0.02
_STATIONARITY_TEXT_BOX_Y = 0.98


def _format_date_axis(ax: Axes) -> None:
    """Format x-axis dates for better readability.

    Wrapper for backward compatibility.

    Args:
        ax: Matplotlib axes to format.
    """
    format_date_axis(ax)


def _save_plot(output_file: str) -> None:
    """Save the current plot to file.

    Wrapper for backward compatibility.

    Args:
        output_file: Path to save the plot.
    """
    save_plot_wrapper(output_file)


# Legacy seasonal decomposition plotting helpers removed.


def plot_stationarity_series_with_bands(
    ax: Axes,
    weighted_series: pd.Series,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    rolling_window: int,
) -> None:
    """Plot original series with rolling mean and std bands.

    Args:
        ax: Matplotlib axes to plot on.
        weighted_series: Original time series.
        rolling_mean: Rolling mean series.
        rolling_std: Rolling standard deviation series.
        rolling_window: Window size for rolling statistics.
    """
    ax.plot(
        weighted_series.index,
        weighted_series,
        label="Série originale",
        linewidth=_LINEWIDTH_THIN,
        alpha=_PLOT_ALPHA_MEDIUM,
        color="blue",
    )
    ax.plot(
        rolling_mean.index,
        rolling_mean,
        label=f"Moyenne mobile (window={rolling_window})",
        linewidth=_LINEWIDTH_BOLD,
        color="red",
    )
    ax.fill_between(
        rolling_mean.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        alpha=_PLOT_ALPHA_FILL,
        color="red",
        label=f"±1 écart-type (window={rolling_window})",
    )
    add_zero_line(ax, linewidth=_LINEWIDTH_THIN, alpha=0.5)
    ax.set_title(
        "Série temporelle avec statistiques mobiles",
        fontsize=_FONTSIZE_SUBTITLE,
        fontweight="bold",
    )
    ax.set_ylabel("Rendement logarithmique pondéré", fontsize=_FONTSIZE_AXIS)
    add_legend(ax)
    add_grid(ax, alpha=_PLOT_ALPHA_LIGHT, linestyle="--")
    _format_date_axis(ax)




def format_stationarity_test_text(
    alpha: float,
    adf_stat: float,
    adf_pval: float,
    adf_lags: int,
    kpss_stat: float,
    kpss_pval: float,
    kpss_lags: int,
    overall_verdict: str,
) -> str:
    """Format stationarity test results as text.

    Args:
        alpha: Significance level.
        adf_stat: ADF test statistic.
        adf_pval: ADF test p-value.
        adf_lags: ADF test lags.
        kpss_stat: KPSS test statistic.
        kpss_pval: KPSS test p-value.
        kpss_lags: KPSS test lags.
        overall_verdict: Overall stationarity verdict.

    Returns:
        Formatted text string with test results.
    """
    adf_verdict = "Stationnaire" if adf_pval < alpha else "Non stationnaire"
    kpss_verdict = (
        "Stationnaire" if kpss_pval > alpha or (kpss_pval != kpss_pval) else "Non stationnaire"
    )

    return (
        f"Résultats des tests de stationnarité (α = {alpha})\n"
        f"{'=' * _STATIONARITY_SEPARATOR_LENGTH}\n"
        f"Verdict global: {overall_verdict}\n"
        f"\n"
        f"Test ADF (Augmented Dickey-Fuller):\n"
        f"  Statistique: {adf_stat:.6f}\n"
        f"  p-value: {adf_pval:.6f}\n"
        f"  Lags: {adf_lags}\n"
        f"  Verdict: {adf_verdict} (p-value {'<' if adf_pval < alpha else '≥'} {alpha})\n"
        f"\n"
        f"Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):\n"
        f"  Statistique: {kpss_stat:.6f}\n"
        f"  p-value: {kpss_pval:.6f}\n"
        f"  Lags: {kpss_lags}\n"
        f"  Verdict: {kpss_verdict} "
        f"(p-value {'>' if kpss_pval > alpha or (kpss_pval != kpss_pval) else '≤'} {alpha})"
    )


def add_stationarity_text_box(ax: Axes, test_text: str) -> None:
    """Add stationarity test results as text box to axes.

    Args:
        ax: Matplotlib axes to add text box to.
        test_text: Formatted test results text.
    """
    ax.text(
        _STATIONARITY_TEXT_BOX_X,
        _STATIONARITY_TEXT_BOX_Y,
        test_text,
        transform=ax.transAxes,
        fontsize=_FONTSIZE_TEXT,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=_PLOT_ALPHA_DEFAULT, edgecolor="black"),
        family="monospace",
    )
