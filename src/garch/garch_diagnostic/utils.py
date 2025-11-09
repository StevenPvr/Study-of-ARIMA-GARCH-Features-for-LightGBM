"""Utility functions for GARCH diagnostics.

Contains helper functions for parameter loading, data preparation,
autocorrelation calculations, plotting utilities, and statistical computations.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_PLOT_Z_CONF,
)
from src.garch.garch_params.estimation import egarch11_variance
from src.garch.structure_garch.utils import load_garch_dataset, prepare_residuals
from src.utils import get_logger

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

logger = get_logger(__name__)


# ============================================================================
# Parameter Loading Utilities
# ============================================================================


def check_converged_params(params: dict | None) -> bool:
    """Check if parameters dictionary indicates convergence."""
    return isinstance(params, dict) and params.get("converged", False)


def try_new_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from new format keys.

    Checks in preference order: egarch_skewt → egarch_student → egarch_normal.
    """
    egarch_skewt = est_payload.get("egarch_skewt")
    if check_converged_params(egarch_skewt):
        return "skewt", egarch_skewt
    egarch_student = est_payload.get("egarch_student")
    if check_converged_params(egarch_student):
        return "student", egarch_student
    egarch_normal = est_payload.get("egarch_normal")
    if check_converged_params(egarch_normal):
        return "normal", egarch_normal
    return None, None


def try_legacy_format_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Try to extract parameters from legacy format (student, normal)."""
    student = est_payload.get("student")
    if check_converged_params(student):
        return "student", student
    normal = est_payload.get("normal")
    if check_converged_params(normal):
        return "normal", normal
    return None, None


def choose_best_params(est_payload: dict) -> tuple[str | None, dict | None]:
    """Choose best converged EGARCH parameters from estimation payload."""
    dist, params = try_new_format_params(est_payload)
    if params is not None:
        return dist, params
    return try_legacy_format_params(est_payload)


def load_estimation_file() -> dict:
    """Load estimation JSON file.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If JSON is invalid.
    """
    try:
        with open(GARCH_ESTIMATION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Estimation file not found: %s", GARCH_ESTIMATION_FILE)
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in estimation file: %s", e)
        raise ValueError(f"Invalid JSON in {GARCH_ESTIMATION_FILE}") from e


def extract_nu_from_params(best: dict) -> float | None:
    """Extract nu parameter from best params dictionary."""
    nu_value = best.get("nu")
    return float(nu_value) if nu_value is not None else None  # type: ignore[arg-type]


def load_and_prepare_residuals() -> np.ndarray:
    """Load dataset and prepare test residuals.

    Raises:
        ValueError: If no valid residuals found.
    """
    data_frame = load_garch_dataset(str(GARCH_DATASET_FILE))
    resid_test = prepare_residuals(data_frame, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]
    if resid_test.size == 0:
        logger.error("No valid residuals found in test set")
        raise ValueError("No valid residuals found in test set")
    return resid_test


def load_data_and_params() -> tuple[np.ndarray, str | None, dict, float | None] | None:
    """Load dataset, residuals, and best EGARCH parameters.

    Returns:
        Tuple of (residuals, distribution, params, nu) or None if no converged model found.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data loading fails.
    """
    est = load_estimation_file()
    dist, best = choose_best_params(est)
    if best is None:
        logger.warning("No converged EGARCH model found in %s", GARCH_ESTIMATION_FILE)
        return None

    nu = extract_nu_from_params(best)
    try:
        resid_test = load_and_prepare_residuals()
        return resid_test, dist, best, nu
    except Exception as e:
        logger.error("Failed to load dataset or prepare residuals: %s", e)
        raise


# ============================================================================
# Residual Standardization Utilities
# ============================================================================


def standardize_residuals(
    residuals: np.ndarray, params: dict[str, float], dist: str = "normal", nu: float | None = None
) -> np.ndarray:
    """Return standardized residuals z_t = e_t / sigma_t using EGARCH(1,1) params."""
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params.get("gamma", 0.0))
    sigma2 = egarch11_variance(e, omega, alpha, gamma, beta, dist=dist, nu=nu)
    if not (np.all(np.isfinite(sigma2)) and np.all(sigma2 > 0)):
        msg = "Invalid variance path for standardization."
        raise ValueError(msg)
    return e / np.sqrt(sigma2)


def compute_standardized_residuals_for_plot(
    all_res: np.ndarray,
    garch_params: dict[str, float] | None,
    dist: str = "normal",
    nu: float | None = None,
) -> np.ndarray | None:
    """Compute standardized residuals if GARCH params are provided."""
    if garch_params is None:
        return None
    try:
        omega = float(garch_params["omega"])
        alpha = float(garch_params["alpha"])
        beta = float(garch_params["beta"])
        gamma = float(garch_params.get("gamma", 0.0))
        sigma2 = egarch11_variance(all_res, omega, alpha, gamma, beta, dist=dist, nu=nu)
        if np.all(np.isfinite(sigma2)) and np.all(sigma2 > 0):
            return all_res / np.sqrt(sigma2)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not standardize residuals: %s", e)
    return None


def prepare_standardized_residuals_for_plotting(
    all_res: np.ndarray,
    garch_params: dict[str, float] | None,
    dist: str,
    nu: float | None,
) -> np.ndarray | None:
    """Prepare standardized residuals for plotting."""
    return compute_standardized_residuals_for_plot(all_res, garch_params, dist=dist, nu=nu)


# ============================================================================
# Autocorrelation Utilities
# ============================================================================


def compute_autocorr_denominator(x: np.ndarray) -> float:
    """Compute denominator for autocorrelation calculation."""
    return float(np.sum(x * x))


def compute_autocorr_lag(x: np.ndarray, k: int, denom: float) -> float:
    """Compute autocorrelation for a single lag."""
    if denom == 0.0:
        return 0.0
    num = float(np.sum(x[k:] * x[:-k]))
    return num / denom


def autocorr(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return sample autocorrelation r_k for k=0..nlags.

    Uses a mean-centered series with a biased denominator (sum of squares).
    This lightweight implementation avoids the statsmodels dependency.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(nlags + 1, dtype=float)
    x = x - float(np.nanmean(x))
    denom = compute_autocorr_denominator(x)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros(nlags + 1, dtype=float)
    r = np.empty(nlags + 1, dtype=float)
    r[0] = 1.0
    for k in range(1, nlags + 1):
        r[k] = compute_autocorr_lag(x, k, denom)
    return r


# ============================================================================
# Partial Autocorrelation Utilities
# ============================================================================


def pacf_init_first_lag(r: np.ndarray, phi_prev: np.ndarray) -> float:
    """Initialize PACF for first lag (k=1)."""
    phi_kk = r[1]
    phi_prev[0] = phi_kk
    return phi_kk


def pacf_compute_lag(
    r: np.ndarray, k: int, phi_prev: np.ndarray, den_prev: float
) -> tuple[float, float]:
    """Compute PACF for lag k > 1."""
    num = r[k] - float(np.dot(phi_prev[: k - 1], r[1:k][::-1]))
    den = den_prev
    phi_kk = 0.0 if den <= 0.0 or not np.isfinite(den) else num / den
    phi_new = phi_prev[: k - 1] - phi_kk * phi_prev[: k - 1][::-1]
    phi_prev[: k - 1] = phi_new
    phi_prev[k - 1] = phi_kk
    den_prev = 1.0 - float(np.dot(phi_prev[:k], r[1 : k + 1]))
    return phi_kk, den_prev


def pacf_from_autocorr(r: np.ndarray, nlags: int) -> np.ndarray:
    """Compute PACF(1..nlags) via Durbin-Levinson recursion from r[0..nlags].

    This mirrors the Yule-Walker approach for partial autocorrelations and
    is sufficient for diagnostics without requiring statsmodels.
    """
    nlags = int(nlags)
    if nlags <= 0:
        return np.asarray([], dtype=float)
    # Ensure r has at least nlags+1 entries; pad with zeros if needed
    if r.size < (nlags + 1):
        r = np.pad(r, (0, nlags + 1 - r.size), constant_values=0.0)
    # phi will hold current AR coefficients up to order k
    pacf = np.empty(nlags, dtype=float)
    phi_prev = np.zeros(nlags, dtype=float)
    den_prev = 1.0
    for k in range(1, nlags + 1):
        if k == 1:
            pacf_init_first_lag(r, phi_prev)
            den_prev = 1.0 - phi_prev[0] * r[1]
        else:
            _, den_prev = pacf_compute_lag(r, k, phi_prev, den_prev)
        pacf[k - 1] = float(np.clip(phi_prev[k - 1], -1.0, 1.0))
    return pacf


# ============================================================================
# Plotting Utilities
# ============================================================================


def write_placeholder_png(path: Path) -> None:
    """Write a tiny valid PNG file to `path` (fallback when matplotlib missing)."""
    png_b64 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2W"
        b"2ZYAAAAASUVORK5CYII="
    )
    data = base64.b64decode(png_b64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:  # pylint: disable=unspecified-encoding
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


def create_figure_canvas(
    figsize: tuple[float, float] = (10, 6),
) -> tuple[Figure, FigureCanvasAgg]:
    """Create matplotlib figure and canvas, handling import errors.

    Args:
        figsize: Figure size as (width, height) in inches.

    Returns:
        Tuple of (figure, canvas) objects.

    Raises:
        ImportError: If matplotlib is unavailable.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
        from matplotlib.figure import Figure  # type: ignore

        fig = Figure(figsize=figsize, constrained_layout=True)
        canvas = FigureCanvas(fig)
        return fig, canvas
    except Exception as ex:  # pragma: no cover - matplotlib optional
        msg = f"Matplotlib unavailable: {ex}"
        raise ImportError(msg) from ex


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


def plot_acf_subplot(ax: Any, acf: np.ndarray, se: float, title: str, lags: int) -> None:
    """Plot ACF on a subplot with confidence bands.

    Args:
        ax: Matplotlib axes object.
        acf: Autocorrelation function values.
        se: Standard error for confidence bands.
        title: Plot title.
        lags: Number of lags to plot.
    """
    lags_idx = np.arange(1, lags + 1)
    ax.bar(lags_idx, acf, color="#1f77b4", width=0.8)
    ax.axhline(GARCH_PLOT_Z_CONF * se, color="red", linestyle="--", linewidth=1)
    ax.axhline(-GARCH_PLOT_Z_CONF * se, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")


def plot_pacf_subplot(ax: Any, pacf: np.ndarray, se: float, title: str, lags: int) -> None:
    """Plot PACF on a subplot with confidence bands.

    Args:
        ax: Matplotlib axes object.
        pacf: Partial autocorrelation function values.
        se: Standard error for confidence bands.
        title: Plot title.
        lags: Number of lags to plot.
    """
    lags_idx = np.arange(1, lags + 1)
    ax.bar(lags_idx, pacf[:lags], color="#ff7f0e", width=0.8)
    ax.axhline(GARCH_PLOT_Z_CONF * se, color="red", linestyle="--", linewidth=1)
    ax.axhline(-GARCH_PLOT_Z_CONF * se, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")


def compute_acf_pacf_data(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ACF, PACF, and standard error for a series."""
    r = autocorr(series, lags)
    acf = r[1 : lags + 1]
    pacf_vals = pacf_from_autocorr(r, lags)
    n = len(series)
    se = 1.0 / np.sqrt(max(1.0, float(n)))
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
    probs = (np.arange(1, n + 1) - 0.5) / n
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
    ax.scatter(theo_q, z_sorted, s=8, color="#1f77b4", alpha=0.8)
    lo = min(theo_q[0], z_sorted[0])
    hi = max(theo_q[-1], z_sorted[-1])
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
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
    ax.plot(np.arange(n_train), train, label="train", color="#1f77b4", linewidth=1)
    if test.size:
        x_test = np.arange(n_train, n_train + test.size)
        ax.plot(x_test, test, label="test", color="#ff7f0e", linewidth=1)
        ax.axvline(n_train - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Residuals (train/test)")
    ax.set_xlabel("t")
    ax.set_ylabel("e_t")
    ax.legend(loc="upper right")


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
    ax.plot(np.arange(n_train), z_train, label="train", color="#2ca02c", linewidth=1)
    if z_test is not None and n_test > 0:
        ax.plot(
            np.arange(n_train, n_train + n_test),
            z_test,
            label="test",
            color="#d62728",
            linewidth=1,
        )
        ax.axvline(n_train - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Standardized residuals (with GARCH variance)")
    ax.set_xlabel("t")
    ax.set_ylabel("z_t = e_t / sigma_t")
    ax.legend(loc="upper right")


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
    fig, canvas = create_figure_canvas(figsize=(10, 6))
    axes = np.atleast_1d(fig.subplots(rows, 1))
    plot_raw_residuals(axes[0], train, test, n_train)

    if z_all is not None and rows == 2:
        z_train = z_all[:n_train]
        z_test = z_all[n_train:] if test.size else None
        plot_standardized_residuals(axes[1], z_train, z_test, n_train, test.size)

    fig.suptitle("Residuals (raw and standardized)")
    return fig, canvas


# ============================================================================
# Statistical Test Utilities
# ============================================================================


def compute_ljung_box_statistics(
    series: np.ndarray,
    lags: int,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test statistics for a given series."""
    lags_list = list(range(1, int(lags) + 1))
    r = autocorr(series, max(lags_list))
    n = float(np.sum(np.isfinite(series)))
    q_stats = []
    p_values = []
    try:
        from scipy.stats import chi2  # type: ignore

        has_scipy = True
    except Exception:  # pragma: no cover - optional
        has_scipy = False
    s = 0.0
    for h in lags_list:
        rk = r[h]
        s += (rk * rk) / max(1.0, (n - h))
        q = n * (n + 2.0) * s
        q_stats.append(float(q))
        if has_scipy:
            # Survival function is numerically stable for upper tail
            p = float(chi2.sf(q, df=h))  # type: ignore[attr-defined]
        else:
            p = float("nan")
        p_values.append(p)
    return {"lags": lags_list, "lb_stat": q_stats, "lb_pvalue": p_values}
