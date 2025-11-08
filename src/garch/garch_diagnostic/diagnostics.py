"""Post-estimation diagnostics for GARCH models.

Implements methodology for verifying GARCH model adequacy:
1. Verify standardized residuals εt/σt behave as centered white noise
2. Verify squared standardized residuals show no significant autocorrelation
   (ACF/PACF plots + Ljung-Box tests)
3. Verify distribution adequacy for zt (Normal or Student-t)
   (graphical diagnostics + normality tests)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DIAGNOSTICS_PLOTS_DIR,
    GARCH_LJUNG_BOX_LAGS_DEFAULT,
    GARCH_PLOT_Z_CONF,
    GARCH_STD_EPSILON,
)
from src.garch.garch_params.estimation import egarch11_variance
from src.utils import get_logger

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

logger = get_logger(__name__)


def _standardize_residuals(
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


def _compute_autocorr_denominator(x: np.ndarray) -> float:
    """Compute denominator for autocorrelation calculation."""
    return float(np.sum(x * x))


def _compute_autocorr_lag(x: np.ndarray, k: int, denom: float) -> float:
    """Compute autocorrelation for a single lag."""
    if denom == 0.0:
        return 0.0
    num = float(np.sum(x[k:] * x[:-k]))
    return num / denom


def _autocorr(x: np.ndarray, nlags: int) -> np.ndarray:
    """Return sample autocorrelation r_k for k=0..nlags.

    Uses a mean-centered series with a biased denominator (sum of squares).
    This lightweight implementation avoids the statsmodels dependency.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(nlags + 1, dtype=float)
    x = x - float(np.nanmean(x))
    denom = _compute_autocorr_denominator(x)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros(nlags + 1, dtype=float)
    r = np.empty(nlags + 1, dtype=float)
    r[0] = 1.0
    for k in range(1, nlags + 1):
        r[k] = _compute_autocorr_lag(x, k, denom)
    return r


def _pacf_init_first_lag(r: np.ndarray, phi_prev: np.ndarray) -> float:
    """Initialize PACF for first lag (k=1)."""
    phi_kk = r[1]
    phi_prev[0] = phi_kk
    return phi_kk


def _pacf_compute_lag(
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


def _pacf_from_autocorr(r: np.ndarray, nlags: int) -> np.ndarray:
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
            _pacf_init_first_lag(r, phi_prev)
            den_prev = 1.0 - phi_prev[0] * r[1]
        else:
            phi_kk, den_prev = _pacf_compute_lag(r, k, phi_prev, den_prev)
        pacf[k - 1] = float(np.clip(phi_prev[k - 1], -1.0, 1.0))
    return pacf


def _write_placeholder_png(path: Path) -> None:
    """Write a tiny valid PNG file to `path` (fallback when matplotlib missing)."""
    import base64

    png_b64 = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2W"
        b"2ZYAAAAASUVORK5CYII="
    )
    data = base64.b64decode(png_b64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _prepare_output_path(
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


def _create_figure_canvas(
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


def _save_figure_or_placeholder(canvas: FigureCanvasAgg, out_path: Path, log_message: str) -> None:
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
        _write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable (%s); wrote placeholder PNG: %s", ex, out_path)


def _compute_ljung_box_statistics(
    series: np.ndarray,
    lags: int,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test statistics for a given series."""
    lags_list = list(range(1, int(lags) + 1))
    r = _autocorr(series, max(lags_list))
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


def compute_ljung_box_on_std_squared(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "normal",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    If the model captures volatility correctly, z² should be uncorrelated.
    """
    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    y = (z**2) - np.mean(z**2)
    return _compute_ljung_box_statistics(y, lags)


def compute_ljung_box_on_std(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_LJUNG_BOX_LAGS_DEFAULT,
    dist: str = "normal",
    nu: float | None = None,
) -> dict[str, list[int] | list[float]]:
    """Compute Ljung-Box test on standardized residuals (z).

    Tests for white noise behavior: standardized residuals should show
    no significant autocorrelation if the model captures volatility correctly.
    """
    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    return _compute_ljung_box_statistics(z, lags)


def _compute_acf_pacf_data(series: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ACF, PACF, and standard error for a series."""
    r = _autocorr(series, lags)
    acf = r[1 : lags + 1]
    pacf_vals = _pacf_from_autocorr(r, lags)
    n = len(series)
    se = 1.0 / np.sqrt(max(1.0, float(n)))
    return acf, pacf_vals, se


def _plot_acf_subplot(ax: Any, acf: np.ndarray, se: float, title: str, lags: int) -> None:
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


def _plot_pacf_subplot(ax: Any, pacf: np.ndarray, se: float, title: str, lags: int) -> None:
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


def save_acf_pacf_std_squared_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized squared residuals (z²).

    Verifies that squared standardized residuals show no significant autocorrelation.
    ACF/PACF should be near zero for all lags if volatility is correctly captured.
    """
    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    y = z**2 - np.mean(z**2)
    acf, pacf_vals, se = _compute_acf_pacf_data(y, lags)
    out_path = _prepare_output_path(outdir, filename, "garch_std_squared_acf_pacf")

    try:
        fig, canvas = _create_figure_canvas(figsize=(10, 6))
        ax1, ax2 = fig.subplots(2, 1)
        _plot_acf_subplot(ax1, acf, se, "ACF of standardized squared residuals (z^2)", lags)
        _plot_pacf_subplot(ax2, pacf_vals, se, "PACF of standardized squared residuals (z^2)", lags)
        fig.suptitle("ACF/PACF of standardized squared residuals")
        _save_figure_or_placeholder(canvas, out_path, "Saved ACF/PACF(z^2) plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        _write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def save_acf_pacf_std_plots(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    lags: int = GARCH_ACF_LAGS_DEFAULT,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save ACF and PACF plots of standardized residuals (z).

    Verifies that standardized residuals εt/σt behave as centered white noise.
    For white noise, ACF/PACF should be near zero for all lags.
    """
    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    zc = z - np.mean(z)
    acf, pacf_vals, se = _compute_acf_pacf_data(zc, lags)
    out_path = _prepare_output_path(outdir, filename, "garch_std_acf_pacf")

    try:
        fig, canvas = _create_figure_canvas(figsize=(10, 6))
        ax1, ax2 = fig.subplots(2, 1)
        _plot_acf_subplot(ax1, acf, se, "ACF of standardized residuals (z)", lags)
        _plot_pacf_subplot(ax2, pacf_vals, se, "PACF of standardized residuals (z)", lags)
        fig.suptitle("ACF/PACF of standardized residuals")
        _save_figure_or_placeholder(canvas, out_path, "Saved ACF/PACF(z) plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        _write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def compute_distribution_diagnostics(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
) -> dict[str, float | str | None]:
    """Compute distribution diagnostics for standardized residuals.

    Verifies adequacy of chosen distribution (Normal or Student-t) for zt:
    - Skewness and kurtosis
    - Jarque-Bera test (normality)
    - Kolmogorov-Smirnov test (distribution fit)

    Args:
        residuals: Raw residuals εt from mean model.
        garch_params: GARCH parameter dictionary.
        dist: Distribution name ('normal' or 'student').
        nu: Degrees of freedom for Student-t (if applicable).

    Returns:
        Dictionary with diagnostic statistics and test results.
    """
    from scipy.stats import jarque_bera, kstest, norm, t  # type: ignore

    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    zc = (z - np.mean(z)) / (np.std(z) + GARCH_STD_EPSILON)
    skew = float(np.nanmean(zc**3))
    kurt = float(np.nanmean(zc**4))
    jb = jarque_bera(z)
    jb_stat = float(jb.statistic)  # type: ignore[attr-defined]
    jb_p = float(jb.pvalue)  # type: ignore[attr-defined]
    if dist.lower() == "student" and nu is not None and nu > 2:
        ks = kstest(z, lambda x: t.cdf(x, df=nu))
        used = "student"
    else:
        ks = kstest(z, lambda x: norm.cdf(x))
        used = "normal"
    return {
        "dist": used,
        "nu": float(nu) if nu is not None else None,
        "skewness": skew,
        "kurtosis": kurt,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_p,
        "ks_stat": float(ks.statistic),  # type: ignore[attr-defined]
        "ks_pvalue": float(ks.pvalue),  # type: ignore[attr-defined]
    }


def _compute_qq_data(
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


def _plot_qq_scatter(ax: Any, theo_q: np.ndarray, z_sorted: np.ndarray, title: str) -> None:
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


def save_qq_plot_std_residuals(
    residuals: np.ndarray,
    garch_params: dict[str, float],
    *,
    dist: str,
    nu: float | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
) -> Path:
    """Save QQ-plot of standardized residuals vs chosen distribution.

    Graphical diagnostic to verify distribution adequacy:
    - Points should lie along the diagonal line if distribution is appropriate
    - Deviations indicate distribution misspecification
    """
    z = _standardize_residuals(residuals, garch_params, dist=dist, nu=nu)
    theo_q, z_sorted, title = _compute_qq_data(z, dist, nu)
    out_path = _prepare_output_path(outdir, filename, "garch_std_residuals_qq")

    try:
        fig, canvas = _create_figure_canvas(figsize=(6, 6))
        ax = fig.subplots(1, 1)
        _plot_qq_scatter(ax, theo_q, z_sorted, title)
        _save_figure_or_placeholder(canvas, out_path, "Saved QQ plot")
    except ImportError:  # pragma: no cover - matplotlib optional
        _write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


def _prepare_residual_data(
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


def _compute_standardized_residuals_for_plot(
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


def _plot_raw_residuals(
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


def _plot_standardized_residuals(
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


def _prepare_standardized_residuals_for_plotting(
    all_res: np.ndarray,
    garch_params: dict[str, float] | None,
    dist: str,
    nu: float | None,
) -> np.ndarray | None:
    """Prepare standardized residuals for plotting."""
    return _compute_standardized_residuals_for_plot(all_res, garch_params, dist=dist, nu=nu)


def _create_residual_plots_figure(
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
    fig, canvas = _create_figure_canvas(figsize=(10, 6))
    axes = np.atleast_1d(fig.subplots(rows, 1))
    _plot_raw_residuals(axes[0], train, test, n_train)

    if z_all is not None and rows == 2:
        z_train = z_all[:n_train]
        z_test = z_all[n_train:] if test.size else None
        _plot_standardized_residuals(axes[1], z_train, z_test, n_train, test.size)

    fig.suptitle("Residuals (raw and standardized)")
    return fig, canvas


def save_residual_plots(
    resid_train: np.ndarray,
    resid_test: np.ndarray,
    *,
    garch_params: dict[str, float] | None = None,
    outdir: str | Path = GARCH_DIAGNOSTICS_PLOTS_DIR,
    filename: str | None = None,
    dist: str = "normal",
    nu: float | None = None,
) -> Path:
    """Save plots of residuals and standardized residuals (if params provided)."""
    train, test, n_train = _prepare_residual_data(resid_train, resid_test)
    all_res = np.concatenate([train, test]) if test.size else train
    z_all = _prepare_standardized_residuals_for_plotting(all_res, garch_params, dist, nu)
    out_path = _prepare_output_path(outdir, filename, "garch_residuals")

    try:
        fig, canvas = _create_residual_plots_figure(train, test, z_all, n_train)
        _save_figure_or_placeholder(canvas, out_path, "Saved residual plots")
    except ImportError:  # pragma: no cover - matplotlib optional
        _write_placeholder_png(out_path)
        logger.warning("Matplotlib unavailable; wrote placeholder PNG: %s", out_path)
    return out_path


__all__ = [
    "save_acf_pacf_std_squared_plots",
    "compute_ljung_box_on_std_squared",
    "compute_ljung_box_on_std",
    "save_acf_pacf_std_plots",
    "compute_distribution_diagnostics",
    "save_qq_plot_std_residuals",
    "save_residual_plots",
    "_standardize_residuals",
]
