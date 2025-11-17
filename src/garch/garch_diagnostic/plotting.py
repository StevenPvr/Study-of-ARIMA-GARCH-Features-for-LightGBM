"""Plotting utilities for GARCH diagnostics.

Contains functions for creating and saving diagnostic plots.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.garch.garch_diagnostic.statistics import autocorr, pacf_from_autocorr
from src.utils import get_logger
from src.visualization import add_zero_line, create_figure_canvas

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

logger = get_logger(__name__)
_Z_CONF = 1.96
_FIGURE_SIZE: tuple[int, int] = (10, 6)
_PLOT_ALPHA = 0.8
_BAR_WIDTH = 0.8
_COLOR_ACF = "#1f77b4"
_COLOR_PACF = "#ff7f0e"
_COLOR_CONFIDENCE = "red"
_COLOR_ZERO_LINE = "black"
_COLOR_GRAY = "gray"
_COLOR_TRAIN = "#1f77b4"
_COLOR_TEST = "#ff7f0e"
_COLOR_TRAIN_STD = "#2ca02c"
_COLOR_TEST_STD = "#d62728"
_LINEWIDTH = 1.0
_LINESTYLE_DASHED = "--"
_LINESTYLE_DOTTED = ":"
_LEGEND_LOC = "upper right"
_SCATTER_SIZE = 8
_QQ_PROB_OFFSET = 0.5
_STD_ERROR_DENOMINATOR = 1.0


def write_placeholder_png(path: Path) -> None:
    """Write a tiny valid PNG file to `path` (fallback when matplotlib missing)."""
    png_b64 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2W"
        b"2ZYAAAAASUVORK5CYII="
    )
    data = base64.b64decode(png_b64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def prepare_output_path(
    outdir: str | Path,
    filename: str | None,
    default_prefix: str,
) -> Path:
    """Prepare output path for plot file."""
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{default_prefix}_{ts}.png"
    return out_dir / filename


def _create_diagnostic_figure_canvas(
    figsize: tuple[float, float] = _FIGURE_SIZE,
) -> tuple[Any, Any]:
    """Create matplotlib figure and canvas for diagnostics.

    Wrapper around common create_figure_canvas for backward compatibility.

    Args:
        figsize: Figure size as (width, height) in inches.

    Returns:
        Tuple of (figure, canvas) objects.

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    fig, canvas, _ = create_figure_canvas(figsize, n_rows=1, n_cols=1)
    return fig, canvas


def save_figure_or_placeholder(canvas: FigureCanvasAgg, out_path: Path, log_message: str) -> None:
    """Save figure to file or write placeholder if matplotlib unavailable.

    Args:
        canvas: Matplotlib canvas object.
        out_path: Output file path.
        log_message: Log message prefix.
    """
    try:
        canvas.print_png(str(out_path))
        logger.info("%s: %s", log_message, out_path)
    except Exception as ex:  # pragma: no cover - matplotlib optional
        write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable (%s); wrote placeholder PNG: %s", ex, out_path)


def _plot_correlation_bars(
    ax: Any,
    values: np.ndarray,
    se: float,
    title: str,
    lags: int,
    color: str,
    ylabel: str,
) -> None:
    """Plot correlation values (ACF/PACF) with confidence bands."""
    lags_idx = np.arange(1, lags + 1)
    ax.bar(
        lags_idx,
        values[:lags] if len(values) > lags else values,
        color=color,
        width=_BAR_WIDTH,
    )
    ax.axhline(
        _Z_CONF * se,
        color=_COLOR_CONFIDENCE,
        linestyle=_LINESTYLE_DASHED,
        linewidth=_LINEWIDTH,
    )
    ax.axhline(
        -_Z_CONF * se,
        color=_COLOR_CONFIDENCE,
        linestyle=_LINESTYLE_DASHED,
        linewidth=_LINEWIDTH,
    )
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)


def plot_acf_subplot(ax: Any, acf: np.ndarray, se: float, title: str, lags: int) -> None:
    """Plot ACF on a subplot with confidence bands.

    Args:
        ax: Matplotlib axes object.
        acf: Autocorrelation function values.
        se: Standard error for confidence bands.
        title: Plot title.
        lags: Number of lags to plot.
    """
    _plot_correlation_bars(ax, acf, se, title, lags, _COLOR_ACF, "ACF")


def plot_pacf_subplot(ax: Any, pacf: np.ndarray, se: float, title: str, lags: int) -> None:
    """Plot PACF on a subplot with confidence bands.

    Args:
        ax: Matplotlib axes object.
        pacf: Partial autocorrelation function values.
        se: Standard error for confidence bands.
        title: Plot title.
        lags: Number of lags to plot.
    """
    _plot_correlation_bars(ax, pacf, se, title, lags, _COLOR_PACF, "PACF")


def compute_acf_pacf_data(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ACF, PACF, and standard error for a series.

    Standard error is computed as 1/√T for confidence bands ±1.96/√T.
    This is the correct asymptotic standard error for ACF of white noise.

    Args:
    ----
        series: Time series to compute ACF/PACF for.
        lags: Number of lags to compute.

    Returns:
    -------
        Tuple of (acf, pacf, standard_error).
        standard_error = 1/√T where T is sample size.

    """
    r = autocorr(series, lags)
    acf = r[1 : lags + 1]
    pacf_vals = pacf_from_autocorr(r, lags)
    n = len(series)
    # Standard error for ACF: 1/√T (asymptotic for white noise)
    # Confidence bands: ±1.96/√T
    se = _STD_ERROR_DENOMINATOR / np.sqrt(max(_STD_ERROR_DENOMINATOR, float(n)))
    return acf, pacf_vals, se


def compute_qq_data(
    z: np.ndarray,
    dist: str,
    nu: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute theoretical quantiles for QQ plot."""
    from scipy.stats import norm, t  # type: ignore

    z_sorted = np.sort(z)
    n = len(z_sorted)
    probs = (np.arange(1, n + 1) - _QQ_PROB_OFFSET) / n
    if dist.lower() == "student" and nu is not None and nu > 2:
        theo_q = t.ppf(probs, df=nu)
        title = f"QQ-plot standardized residuals vs t(df={nu:.1f})"
    else:
        theo_q = norm.ppf(probs)
        title = "QQ-plot standardized residuals vs N(0,1)"
    return theo_q, z_sorted, title


def plot_qq_scatter(ax: Any, theo_q: np.ndarray, z_sorted: np.ndarray, title: str) -> None:
    """Plot QQ scatter plot with reference line.

    Args:
        ax: Matplotlib axes object.
        theo_q: Theoretical quantiles.
        z_sorted: Sorted standardized residuals.
        title: Plot title.
    """
    ax.scatter(
        theo_q,
        z_sorted,
        s=_SCATTER_SIZE,
        color=_COLOR_ACF,
        alpha=_PLOT_ALPHA,
    )
    lo = min(theo_q[0], z_sorted[0])
    hi = max(theo_q[-1], z_sorted[-1])
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=_COLOR_CONFIDENCE,
        linestyle=_LINESTYLE_DASHED,
        linewidth=_LINEWIDTH,
    )
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")


def prepare_residual_data(
    resid_train: np.ndarray,
    resid_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare and validate residual data for plotting."""
    train = np.asarray(resid_train, dtype=float)
    test = np.asarray(resid_test, dtype=float)
    train = train[np.isfinite(train)]
    test = test[np.isfinite(test)]
    if train.size == 0 and test.size == 0:
        msg = "No residuals to plot."
        raise ValueError(msg)
    n_train = int(train.size)
    return train, test, n_train


def plot_raw_residuals(
    ax: Any,
    train: np.ndarray,
    test: np.ndarray,
    n_train: int,
) -> None:
    """Plot raw residuals on axis.

    Args:
        ax: Matplotlib axes object.
        train: Training residuals.
        test: Test residuals.
        n_train: Number of training samples.
    """
    ax.plot(
        np.arange(n_train),
        train,
        label="train",
        color=_COLOR_TRAIN,
        linewidth=_LINEWIDTH,
    )
    if test.size:
        x_test = np.arange(n_train, n_train + test.size)
        ax.plot(
            x_test,
            test,
            label="test",
            color=_COLOR_TEST,
            linewidth=_LINEWIDTH,
        )
        ax.axvline(
            n_train - _QQ_PROB_OFFSET,
            color=_COLOR_GRAY,
            linestyle=_LINESTYLE_DOTTED,
            linewidth=_LINEWIDTH,
        )
    ax.set_title("Residuals (train/test)")
    ax.set_xlabel("t")
    ax.set_ylabel("e_t")
    ax.legend(loc=_LEGEND_LOC)


def plot_standardized_residuals(
    ax: Any,
    z_train: np.ndarray,
    z_test: np.ndarray | None,
    n_train: int,
    n_test: int,
) -> None:
    """Plot standardized residuals on axis.

    Args:
        ax: Matplotlib axes object.
        z_train: Training standardized residuals.
        z_test: Test standardized residuals (optional).
        n_train: Number of training samples.
        n_test: Number of test samples.
    """
    ax.plot(
        np.arange(n_train),
        z_train,
        label="train",
        color=_COLOR_TRAIN_STD,
        linewidth=_LINEWIDTH,
    )
    if z_test is not None and n_test > 0:
        ax.plot(
            np.arange(n_train, n_train + n_test),
            z_test,
            label="test",
            color=_COLOR_TEST_STD,
            linewidth=_LINEWIDTH,
        )
        ax.axvline(
            n_train - _QQ_PROB_OFFSET,
            color=_COLOR_GRAY,
            linestyle=_LINESTYLE_DOTTED,
            linewidth=_LINEWIDTH,
        )
    add_zero_line(ax, color=_COLOR_ZERO_LINE, linewidth=_LINEWIDTH)
    ax.set_title("Standardized residuals (with GARCH variance)")
    ax.set_xlabel("t")
    ax.set_ylabel("z_t = e_t / sigma_t")
    ax.legend(loc=_LEGEND_LOC)


def create_residual_plots_figure(
    train: np.ndarray,
    test: np.ndarray,
    z_all: np.ndarray | None,
    n_train: int,
) -> tuple[Figure, FigureCanvasAgg]:
    """Create figure with residual plots.

    Args:
        train: Training residuals.
        test: Test residuals.
        z_all: Standardized residuals (optional).
        n_train: Number of training samples.

    Returns:
        Tuple of (figure, canvas) objects.
    """
    rows = 2 if z_all is not None else 1
    fig, canvas = _create_diagnostic_figure_canvas(figsize=_FIGURE_SIZE)
    axes = np.atleast_1d(fig.subplots(rows, 1))
    plot_raw_residuals(axes[0], train, test, n_train)

    if z_all is not None and rows == 2:
        z_train = z_all[:n_train]
        z_test = z_all[n_train:] if test.size else None
        plot_standardized_residuals(axes[1], z_train, z_test, n_train, test.size)

    fig.suptitle("Residuals (raw and standardized)")
    return fig, canvas
