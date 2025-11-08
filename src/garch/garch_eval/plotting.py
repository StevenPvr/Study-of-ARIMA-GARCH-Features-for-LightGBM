"""Plotting utilities for GARCH evaluation (variance and VaR visuals)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_EVAL_FIGURE_SIZE_DEFAULT,
    GARCH_EVAL_FIGURE_SIZE_RESIDUALS,
    GARCH_EVAL_FIGURE_SIZE_SCATTER,
    GARCH_EVAL_PLOT_LIMIT_MULTIPLIER,
    GARCH_EVAL_VAR_RESIDUALS_PLOT,
    GARCH_EVAL_VAR_SCATTER_PLOT,
    GARCH_EVAL_VAR_TIMESERIES_PLOT,
    GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE,
    GARCH_STUDENT_NU_MIN,
    GARCH_VARIANCE_OUTPUTS_FILE,
    PLOTS_DIR,
)
from src.garch.garch_eval.metrics import _load_test_resid_sigma2
from src.garch.garch_eval.distributions import skewt_ppf
from src.utils import get_logger

logger = get_logger(__name__)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_numpy(series_like: Iterable[float]) -> np.ndarray:
    return np.asarray(list(series_like), dtype=float)


def plot_variance_timeseries(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Plot time series of realized variance (e^2) vs predicted variance (σ²)."""
    import matplotlib.pyplot as plt  # local import to keep import cost low
    import seaborn as sns

    _ensure_parent(output)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=GARCH_EVAL_FIGURE_SIZE_DEFAULT)
    ax.plot(dates, e_test**2, label="Réalisé e²", color="#1f77b4", linewidth=1.2)
    ax.plot(dates, s2_test, label="Prévu σ²", color="#ff7f0e", linewidth=1.2)
    ax.set_title("Variance réalisée vs prévision GARCH (test)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance time series plot to: %s", output)


def plot_variance_scatter(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Scatter e² vs σ² with y=x reference line."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ensure_parent(output)
    y = e_test**2
    x = s2_test

    # Robust limit: handle empty arrays or all-NaN safely
    def _safe_nanmax(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        m = np.nanmax(arr)
        return float(m) if np.isfinite(m) else float("nan")

    xm = _safe_nanmax(x)
    ym = _safe_nanmax(y)
    base = max([v for v in [xm, ym] if np.isfinite(v)], default=1.0)
    lim = float(base * GARCH_EVAL_PLOT_LIMIT_MULTIPLIER) if base > 0 else 1.0

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=GARCH_EVAL_FIGURE_SIZE_SCATTER)
    ax.scatter(x, y, s=10, alpha=0.6, color="#1f77b4", label="Points (e² vs σ²)")
    ax.plot([0, lim], [0, lim], color="#2ca02c", linestyle="--", label="y = x")
    ax.set_title("Dispersion e² (réalisé) vs σ² (prévu)")
    ax.set_xlabel("σ² (prévu)")
    ax.set_ylabel("e² (réalisé)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance scatter plot to: %s", output)


def plot_variance_residuals(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
    output: Path,
) -> None:
    """Plot residuals (e² - σ²) over time."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ensure_parent(output)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=GARCH_EVAL_FIGURE_SIZE_RESIDUALS)
    residuals = (e_test**2) - s2_test
    ax.plot(dates, residuals, color="#d62728", linewidth=1.0, label="e² - σ²")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title("Résidus de variance (e² - σ²) sur test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Différence")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved variance residuals plot to: %s", output)


def _var_quantile(alpha: float, dist: str, nu: float | None, lambda_skew: float | None = None) -> float:
    from scipy.stats import norm, t  # type: ignore

    d = dist.lower()
    if d == "student":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN:
            msg = "Student-t requires nu>2 for VaR plots"
            raise ValueError(msg)
        return float(t.ppf(alpha, df=float(nu)))
    if d == "skewt":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN or lambda_skew is None:
            msg = "Skew-t requires nu>2 and lambda for VaR plots"
            raise ValueError(msg)
        return float(skewt_ppf(alpha, float(nu), float(lambda_skew)))
    return float(norm.ppf(alpha))


def plot_var_violations(
    dates: pd.Series | np.ndarray,
    e_test: np.ndarray,
    s2_test: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    output: Path,
    lambda_skew: float | None = None,
) -> None:
    """Plot returns vs VaR band at given alpha and mark violations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    _ensure_parent(output)
    sns.set_style("whitegrid")
    q = _var_quantile(alpha, dist, nu, lambda_skew)
    var_series = q * np.sqrt(s2_test)
    hits = e_test < var_series

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, e_test, color="#1f77b4", linewidth=1.0, label="Retours (e)")
    ax.plot(dates, var_series, color="#ff7f0e", linewidth=1.2, label=f"VaR (α={alpha:.2f})")
    if np.any(hits):
        ax.scatter(
            np.asarray(dates)[hits],
            e_test[hits],
            color="#d62728",
            s=18,
            label="Violations",
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"VaR et violations (α={alpha:.2f}) sur test")
    ax.set_xlabel("Date")
    ax.set_ylabel("Retour")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved VaR violations plot to: %s", output)


def generate_eval_plots_from_artifacts(
    *,
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> None:
    """Load artifacts and dataset, build test series, and save evaluation plots."""
    # Prefer variance outputs CSV for dates/splits if present, else dataset
    try:
        dataset_df = pd.read_csv(GARCH_VARIANCE_OUTPUTS_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    except Exception:
        dataset_df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]

    df_sorted = dataset_df.sort_values("date").reset_index(drop=True)
    e_test, s2_test = _load_test_resid_sigma2(
        params, df_sorted, model_name=model_name, dist=dist, nu=nu, lambda_skew=lambda_skew
    )

    test_mask = (df_sorted["split"].astype(str) == "test").to_numpy()
    n = min(int(test_mask.sum()), int(e_test.size), int(s2_test.size))
    start_idx = int(np.argmax(test_mask)) if np.any(test_mask) else 0
    dates_test = df_sorted.loc[start_idx : start_idx + n - 1, "date"].to_numpy()

    # Core variance plots
    plot_variance_timeseries(dates_test, e_test, s2_test, GARCH_EVAL_VAR_TIMESERIES_PLOT)
    plot_variance_scatter(e_test, s2_test, GARCH_EVAL_VAR_SCATTER_PLOT)
    plot_variance_residuals(dates_test, e_test, s2_test, GARCH_EVAL_VAR_RESIDUALS_PLOT)

    # VaR violations plots per alpha
    for a in alphas:
        file_name = GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE.format(alpha=f"{a:.2f}")
        output = PLOTS_DIR / file_name
        plot_var_violations(dates_test, e_test, s2_test, float(a), dist, nu, output, lambda_skew)
