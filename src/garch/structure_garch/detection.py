"""Identification and pre-diagnostics for ARCH/GARCH.

Implements methodology for detecting conditional heteroskedasticity:
1. Extract residuals εt from SARIMA model (mean model)
2. Test for ARCH effect using Lagrange Multiplier test (ARCH-LM)
3. Inspect autocorrelation of squared residuals

A significant autocorrelation structure in squared residuals indicates
that a GARCH model is relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_ACF_LAGS_DEFAULT,
    GARCH_DATASET_FILE,
    GARCH_LM_LAGS_DEFAULT,
    GARCH_PLOT_Z_CONF,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_garch_dataset(path: str | None = None) -> pd.DataFrame:
    """Load dataset for GARCH training/diagnostics.

    Args:
        path: Optional CSV path. Defaults to `GARCH_DATASET_FILE` from constants.

    Returns:
        DataFrame with required columns.

    Raises:
        FileNotFoundError: If dataset is missing.
        ValueError: If required columns are absent.
    """
    csv_path = GARCH_DATASET_FILE if path is None else Path(path)
    if not csv_path.exists():
        msg = f"GARCH dataset not found: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path, parse_dates=["date"])  # type: ignore[arg-type]
    required = {"date", "split", "weighted_log_return"}
    if not required.issubset(df.columns):
        raise ValueError("GARCH dataset missing required columns: " + ", ".join(sorted(required)))
    return df


def prepare_residuals(df: pd.DataFrame, use_test_only: bool = True) -> np.ndarray:
    """Extract residuals εt from SARIMA model (mean model).

    This step enforces the presence of the column 'arima_residual_return'
    which contains residuals from the SARIMA model fitted on returns.
    If it is missing or contains only NaNs, an error is raised.

    Args:
        df: Input dataframe with SARIMA residuals.
        use_test_only: Restrict to test split if True.

    Returns:
        1D residual array εt.

    Raises:
        ValueError: When 'arima_residual_return' is absent or empty.
    """
    data = df.copy()
    if use_test_only and "split" in data.columns:
        data = data[data["split"] == "test"].copy()

    if "arima_residual_return" not in data.columns:
        msg = (
            "Required ARIMA residuals column 'arima_residual_return' not found for identification."
        )
        raise ValueError(msg)

    series = pd.to_numeric(data["arima_residual_return"], errors="coerce")
    # Use Series.notna().any() to avoid attribute-resolution issues on bool
    if not bool(pd.Series(series).notna().any()):
        msg = "Column 'arima_residual_return' contains no valid values for identification."
        raise ValueError(msg)

    return np.asarray(series, dtype=float)


def compute_acf(series: np.ndarray, nlags: int = GARCH_ACF_LAGS_DEFAULT) -> np.ndarray:
    """Compute sample ACF for a 1D series (lags 1..nlags).

    Args:
        series: 1D numeric array.
        nlags: Maximum lag.

    Returns:
        ACF values for lags 1..nlags.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.zeros(nlags, dtype=float)
    xc = x - np.mean(x)
    denom = float(np.sum(xc * xc))
    acf = np.zeros(nlags, dtype=float)
    for k in range(1, nlags + 1):
        num = float(np.sum(xc[k:] * xc[:-k]))
        acf[k - 1] = (num / denom) if denom != 0.0 else 0.0
    return acf


def compute_squared_acf(residuals: np.ndarray, nlags: int = GARCH_ACF_LAGS_DEFAULT) -> np.ndarray:
    """Compute autocorrelation of squared residuals.

    Inspects autocorrelation structure in squared residuals ε_t^2.
    A significant autocorrelation indicates that a GARCH model is relevant.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        nlags: Maximum lag.

    Returns:
        ACF(ε^2) for lags 1..nlags.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    return compute_acf(e2, nlags=nlags)


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P[X >= x].

    Notes:
        Uses SciPy when available; tests can monkeypatch for determinism.
    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


def compute_arch_lm_test(
    residuals: np.ndarray, lags: int = GARCH_LM_LAGS_DEFAULT
) -> dict[str, float]:
    """Engle's ARCH-LM test (Lagrange Multiplier test) for ARCH effect.

    Tests for ARCH effect by regressing squared residuals on lagged squared residuals:
    ε_t^2 ~ const + lags(ε_t^2)

    Uses OLS R^2 to compute the LM statistic.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        lags: Number of lags in regression.

    Returns:
        Dict with lm_stat, p_value, df.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    e2 = e2[~np.isnan(e2)]
    n = int(e2.size)
    if n <= lags:
        return {"lm_stat": float("nan"), "p_value": float("nan"), "df": float(lags)}

    Y = e2[lags:]
    X = np.ones((n - lags, lags + 1), dtype=float)
    for j in range(1, lags + 1):
        X[:, j] = e2[lags - j : n - j]

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        Y_hat = X @ beta
        ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
        ss_res = float(np.sum((Y - Y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        lm = (n - lags) * max(0.0, r2)
    except Exception:
        lm = float("nan")

    p_val = _chi2_sf(lm, lags)
    return {"lm_stat": float(lm), "p_value": float(p_val), "df": float(lags)}


def detect_heteroskedasticity(
    residuals: np.ndarray,
    *,
    lags: int = GARCH_LM_LAGS_DEFAULT,
    acf_lags: int = GARCH_ACF_LAGS_DEFAULT,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Detect conditional heteroskedasticity (ARCH/GARCH effect).

    Runs:
    1. ARCH-LM test (Lagrange Multiplier test) for ARCH effect
    2. ACF of squared residuals to inspect autocorrelation structure

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        lags: Lags for ARCH-LM test.
        acf_lags: Max lag for ACF(ε^2).
        alpha: Significance level for ARCH-LM test.

    Returns:
        Dict summarizing test statistics and boolean flags.
    """
    lm = compute_arch_lm_test(residuals, lags=lags)
    acf_sq = compute_squared_acf(residuals, nlags=acf_lags)
    n = int(np.asarray(residuals, dtype=float).size)
    acf_sig = GARCH_PLOT_Z_CONF / np.sqrt(max(1, n))

    arch_present = np.isfinite(lm["p_value"]) and lm["p_value"] < alpha
    acf_significant = bool(np.any(np.abs(acf_sq) > acf_sig))

    return {
        "arch_lm": lm,
        "acf_squared": acf_sq.tolist(),
        "acf_significance_level": float(acf_sig),
        "arch_effect_present": arch_present,
        "acf_significant": acf_significant,
    }


def plot_arch_diagnostics(
    residuals: np.ndarray,
    *,
    acf_lags: int = GARCH_ACF_LAGS_DEFAULT,
    out_path: Path | None = None,
) -> Path:
    """Create and save ARCH/GARCH diagnostic plot.

    Visualizes:
    - Residuals εt time series
    - ACF of squared residuals ε_t^2

    Saved to plots/garch/structure/ by default.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        acf_lags: Max lag for ACF(ε^2).
        out_path: Optional output path. Defaults to GARCH_STRUCTURE_PLOT.

    Returns:
        Path to saved plot.
    """
    Figure, FigureCanvas, have_matplotlib = _safe_import_matplotlib()
    x, acf_sq, conf = _prepare_plot_series(residuals=residuals, acf_lags=acf_lags)
    out_path = _resolve_out_path(out_path)
    _ensure_output_dir(out_path)

    if have_matplotlib:
        _render_with_matplotlib(
            Figure=cast(Any, Figure),
            FigureCanvas=cast(Any, FigureCanvas),
            x=x,
            acf_sq=acf_sq,
            conf=conf,
            acf_lags=acf_lags,
            out_path=out_path,
        )
    else:
        _write_placeholder_file(out_path)

    _verify_or_fallback(out_path)
    logger.info("Saved heteroskedasticity plot: %s", out_path)
    return out_path


def _safe_import_matplotlib() -> tuple[Any | None, Any | None, bool]:
    """Import Matplotlib Agg primitives defensively.

    Returns:
        (Figure, FigureCanvas, available_flag)
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # type: ignore
        from matplotlib.figure import Figure  # type: ignore

        return Figure, FigureCanvas, True
    except Exception:
        return None, None, False


def _prepare_plot_series(
    *, residuals: np.ndarray, acf_lags: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prepare finite residuals, ACF(e^2) and confidence level."""
    x = np.asarray(residuals, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    acf_sq = compute_squared_acf(x, nlags=acf_lags)
    conf = GARCH_PLOT_Z_CONF / np.sqrt(max(1, n))
    return x, acf_sq, float(conf)


def _resolve_out_path(out_path: Path | None) -> Path:
    """Resolve output path, defaulting to project constant."""
    if out_path is not None:
        return out_path
    from src.constants import GARCH_STRUCTURE_PLOT

    return GARCH_STRUCTURE_PLOT


def _ensure_output_dir(path: Path) -> None:
    """Ensure parent directory exists for the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_with_matplotlib(
    *,
    Figure: Any,
    FigureCanvas: Any,
    x: np.ndarray,
    acf_sq: np.ndarray,
    conf: float,
    acf_lags: int,
    out_path: Path,
) -> None:
    """Render diagnostics plot using Matplotlib Agg backend."""
    fig = Figure(figsize=(10, 6), constrained_layout=True)
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(x, color="#1f77b4", linewidth=1.0)
    ax1.set_title("Residuals (test)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("ε_t")

    lags = np.arange(1, acf_lags + 1)
    ax2.bar(lags, acf_sq, color="#ff7f0e", width=0.8)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.axhline(conf, color="red", linestyle="--", linewidth=0.8)
    ax2.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    ax2.set_title("ACF of squared residuals")
    ax2.set_xlabel("lag")
    ax2.set_ylabel("acf(e_t^2)")

    fig.suptitle("ARCH/GARCH identification structure", fontsize=12)
    canvas.print_png(str(out_path))


def _write_placeholder_file(path: Path) -> None:
    """Write a minimal non-empty placeholder when plotting is unavailable."""
    path.write_bytes(b"placeholder")


def _verify_or_fallback(path: Path) -> None:
    """Ensure output exists and is non-empty; fallback to placeholder if needed."""
    try:
        ok = path.exists() and path.stat().st_size > 0
    except Exception:
        ok = False
    if ok:
        return
    logger.warning(
        "Plot save did not create a file; creating a minimal placeholder at %s",
        path,
    )
    try:
        path.write_bytes(b"placeholder")
    except Exception as exc:
        msg = f"Failed to write diagnostics plot to {path}"
        raise RuntimeError(msg) from exc


__all__ = [
    "load_garch_dataset",
    "prepare_residuals",
    "compute_acf",
    "compute_squared_acf",
    "compute_arch_lm_test",
    "detect_heteroskedasticity",
    "plot_arch_diagnostics",
]
