"""Volatility backtest orchestrator: EGARCH vs simple baselines.

This module prepares a fair comparison for volatility forecasting
(not directional decisions) by computing one-step-ahead variance paths
for multiple models on the same test split:
- ARIMA-EGARCH (rolling, periodic refits)
- EWMA
- Rolling variance/std
- ARCH(1)
- HAR(3) on squared returns

Inputs are expected to include either 'arima_residual_return' (preferred)
or 'weighted_log_return'. Only the residual column is used for EGARCH.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.constants as C
from src.models import egarch as eg
from src.garch.structure_garch.detection import load_garch_dataset
from src.utils import get_logger

logger = get_logger(__name__)


# Exposed for tests to monkeypatch (keeps CI fast when needed)
def run_rolling_garch_from_artifacts(**kwargs):  # noqa: D401 - thin wrapper
    """Delegate to EGARCH artifacts runner."""
    return eg.run_from_artifacts(**kwargs)


def _select_residual_column(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (e_all, dates_all, test_mask) from df using best available column.

    Prefers 'arima_residual_return'; falls back to 'weighted_log_return' for
    baselines-only scenarios in tests.

    Args:
        df: DataFrame with date, split, and residual/return columns.

    Returns:
        Tuple of (residuals array, dates array, test mask array).

    Raises:
        ValueError: If required columns are missing.
    """
    if "date" not in df.columns or "split" not in df.columns:
        msg = "DataFrame must contain 'date' and 'split' columns"
        raise ValueError(msg)
    data = df.sort_values("date").reset_index(drop=True)
    dates_series = pd.to_datetime(data["date"])
    dates = np.array(dates_series, dtype="datetime64[ns]")
    test_mask = np.array(data["split"].astype(str) == "test", dtype=bool)
    col = (
        "arima_residual_return"
        if "arima_residual_return" in data.columns
        else "weighted_log_return"
    )
    if col not in data.columns:
        msg = "Neither 'arima_residual_return' nor 'weighted_log_return' found in DataFrame"
        raise ValueError(msg)
    e_series = pd.to_numeric(data[col], errors="coerce")
    e_all = np.array(e_series, dtype=np.float64)
    return e_all, dates, test_mask


def _pos_test_valid(e_all: np.ndarray, test_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return positions for valid series and test indices mapped onto it."""
    valid = np.isfinite(e_all)
    idx_all = np.arange(e_all.size)
    idx_valid = idx_all[valid]
    idx_test = idx_all[test_mask]
    idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)
    pos_in_valid = np.full(e_all.size, -1, dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)
    pos_test = pos_in_valid[idx_test_valid]
    pos_test.sort()
    return valid, pos_test


def _ewma_forecast(e_valid: np.ndarray, pos_test: np.ndarray, lam: float) -> np.ndarray:
    """One-step EWMA variance: s2_{t+1} = lam*s2_t + (1-lam)*e_t^2."""
    if pos_test.size == 0:
        return np.array([], dtype=float)
    # Initialize with train variance up to first test pos
    p0 = int(pos_test[0])
    s2_prev = float(np.var(e_valid[:p0])) if p0 > 1 else float(np.var(e_valid[: max(1, p0)]))
    s2_prev = max(s2_prev, 1e-12)
    out = np.empty(pos_test.size, dtype=float)
    last_pos = p0
    for i, pos in enumerate(pos_test):
        # Advance across any gap by iterating sequentially
        while last_pos < pos:
            s2_prev = lam * s2_prev + (1.0 - lam) * float(e_valid[last_pos - 1] ** 2)
            last_pos += 1
        out[i] = lam * s2_prev + (1.0 - lam) * float(e_valid[pos - 1] ** 2)
        s2_prev = out[i]
        last_pos = pos + 1
    return out


def _rolling_var_forecast(e_valid: np.ndarray, pos_test: np.ndarray, window: int) -> np.ndarray:
    """One-step rolling variance forecast using last `window` observations."""
    out = np.empty(pos_test.size, dtype=float)
    for i, pos in enumerate(pos_test):
        start = max(0, int(pos) - int(window))
        if (int(pos) - start) > 1:
            s = float(np.var(e_valid[start : int(pos)]))
        else:
            s = float(np.var(e_valid[: max(1, pos)]))
        out[i] = max(s, 1e-12)
    return out


def _arch1_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """ARCH(1): sigma2_{t+1} = omega + alpha * e_t^2 (estimated on train)."""
    out = np.empty(pos_test.size, dtype=float)
    if pos_test.size == 0:
        return out
    p0 = int(pos_test[0])
    train = e_valid[:p0]
    if train.size <= 2:
        omega, alpha = 1e-6, 0.0
    else:
        y = (train[1:] ** 2).astype(float)
        x = (train[:-1] ** 2).astype(float)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        omega = float(max(beta[0], 1e-12))
        alpha = float(max(beta[1], 0.0))
    for i, pos in enumerate(pos_test):
        out[i] = float(omega + alpha * float(e_valid[int(pos) - 1] ** 2))
        out[i] = max(out[i], 1e-12)
    return out


def _has_sufficient_history(t: int) -> bool:
    """Check if index t has sufficient history for HAR(3) features."""
    return t - 1 >= 0 and t - C.VOL_HAR_WEEK_WINDOW >= 0 and t - C.VOL_HAR_MONTH_WINDOW >= 0


def _build_har_features(rv: np.ndarray, t: int) -> np.ndarray:
    """Build HAR(3) feature vector for index t."""
    d = rv[t - 1]
    w = float(np.mean(rv[t - C.VOL_HAR_WEEK_WINDOW : t]))
    m = float(np.mean(rv[t - C.VOL_HAR_MONTH_WINDOW : t]))
    return np.array([1.0, d, w, m], dtype=float)


def _fit_har_model(rv: np.ndarray, p0: int) -> np.ndarray:
    """Fit HAR(3) model on training data."""
    rows = []
    ys = []
    for t in range(1, p0):
        if not _has_sufficient_history(t):
            continue
        features = _build_har_features(rv, t)
        rows.append(features.tolist())
        ys.append(rv[t])

    if len(ys) >= 4:
        X = np.asarray(rows, dtype=float)
        y = np.asarray(ys, dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    mean_rv = float(np.mean(rv[:p0])) if p0 > 0 else 1e-6
    return np.array([mean_rv, 0.0, 0.0, 0.0], dtype=float)


def _har3_forecast(e_valid: np.ndarray, pos_test: np.ndarray) -> np.ndarray:
    """HAR(3) on squared returns: d (lag1), w (5-day avg), m (22-day avg)."""
    out = np.empty(pos_test.size, dtype=float)
    if pos_test.size == 0:
        return out

    p0 = int(pos_test[0])
    rv = (e_valid**2).astype(float)
    beta = _fit_har_model(rv, p0)

    for i, pos in enumerate(pos_test):
        t = int(pos)
        if _has_sufficient_history(t):
            features = _build_har_features(rv, t)
            s2 = float(np.dot(beta, features))
        else:
            s2 = float(np.var(rv[:t])) if t > 1 else float(np.var(rv[: max(1, t)]))
        out[i] = max(s2, 1e-12)
    return out


def _qlike(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute QLIKE loss for variance forecasts.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        QLIKE loss (lower is better).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.log(s2[m]) + (e[m] ** 2) / s2[m]))


def _mse_variance(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute MSE between realized e^2 and forecast sigma^2.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        Mean squared error (lower is better).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return float("nan")
    y = e[m] ** 2
    f = s2[m]
    return float(np.mean((y - f) ** 2))


def _mae_variance(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute MAE between realized e^2 and forecast sigma^2.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        Mean absolute error (lower is better).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return float("nan")
    y = e[m] ** 2
    f = s2[m]
    return float(np.mean(np.abs(y - f)))


def _rmse_variance(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute RMSE between realized e^2 and forecast sigma^2.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        Root mean squared error (lower is better).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return float("nan")
    y = e[m] ** 2
    f = s2[m]
    mse = float(np.mean((y - f) ** 2))
    return float(np.sqrt(mse))


def _r2_variance(e: np.ndarray, s2: np.ndarray) -> float:
    """Compute R² between realized e^2 and forecast sigma^2.

    Args:
        e: Realized residuals.
        s2: Forecast variance.

    Returns:
        R-squared coefficient (higher is better, max 1.0).
    """
    e = np.asarray(e, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m) or m.sum() < 2:
        return float("nan")
    y = e[m] ** 2
    f = s2[m]
    ss_res = float(np.sum((y - f) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _merge_outputs(
    dates_test: np.ndarray,
    e_test: np.ndarray,
    s2_garch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_roll_std: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates_test,
            "e": e_test,
            "sigma2_forecast": s2_garch,
            "s2_arima_garch": s2_garch,
            "s2_ewma": s2_ewma,
            "s2_roll_var": s2_roll_var,
            "s2_roll_std": s2_roll_std,
            "s2_arch1": s2_arch1,
            "s2_har3": s2_har3,
        }
    )


def _validate_ewma_lambda(ewma_lambda: float) -> None:
    """Validate EWMA lambda parameter."""
    if not (0.0 < ewma_lambda < 1.0):
        msg = f"ewma_lambda must be in (0, 1), got {ewma_lambda}"
        raise ValueError(msg)


def _validate_rolling_window(rolling_window: int) -> None:
    """Validate rolling window parameter."""
    if rolling_window < 1:
        msg = f"rolling_window must be >= 1, got {rolling_window}"
        raise ValueError(msg)


def _validate_var_alphas(var_alphas: list[float]) -> None:
    """Validate VaR alphas parameter."""
    if not isinstance(var_alphas, list):
        msg = f"var_alphas must be a list, got {type(var_alphas)}"
        raise ValueError(msg)
    if not all(isinstance(a, (int, float)) and 0.0 < a < 1.0 for a in var_alphas):
        msg = f"var_alphas must be a list of floats in (0, 1), got {var_alphas}"
        raise ValueError(msg)


def _validate_backtest_params(
    ewma_lambda: float | None,
    rolling_window: int | None,
    var_alphas: list[float] | None,
) -> tuple[float, int]:
    """Validate and resolve backtest parameters.

    Args:
        ewma_lambda: EWMA decay factor (lambda).
        rolling_window: Window size for rolling baselines.
        var_alphas: VaR alphas (validated but not returned).

    Returns:
        Tuple of (validated ewma_lambda, validated rolling_window).

    Raises:
        ValueError: If parameters are invalid.
    """
    if ewma_lambda is None:
        ewma_lambda = float(C.VOL_EWMA_LAMBDA_DEFAULT)
    else:
        ewma_lambda = float(ewma_lambda)

    if rolling_window is None:
        rolling_window = int(C.VOL_ROLLING_WINDOW_DEFAULT)
    else:
        rolling_window = int(rolling_window)

    _validate_ewma_lambda(ewma_lambda)
    _validate_rolling_window(rolling_window)
    if var_alphas is not None:
        _validate_var_alphas(var_alphas)

    return ewma_lambda, rolling_window


def _run_garch_forecasts(
    df: pd.DataFrame | None,
    var_alphas: list[float] | None,
    refit_every: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run GARCH forecasts from DataFrame or artifacts.

    Args:
        df: DataFrame with residuals, or None to use artifacts.
        var_alphas: VaR alphas for GARCH runner.
        refit_every: Refit frequency (in test observations).

    Returns:
        Tuple of (forecasts DataFrame, metrics dict with dist/nu info).
    """
    if (df is None) or ("arima_residual_return" not in getattr(df, "columns", [])):
        fore_garch, metr_garch = run_rolling_garch_from_artifacts(
            refit_every=refit_every,
            window=C.GARCH_REFIT_WINDOW_DEFAULT,
            window_size=C.GARCH_REFIT_WINDOW_SIZE_DEFAULT,
            dist_preference="auto",
            keep_nu_between_refits=True,
            var_alphas=var_alphas,
        )
    else:
        fore_garch, metr_garch = eg.run_from_df(
            df,
            refit_every=refit_every,
            window=C.GARCH_REFIT_WINDOW_DEFAULT,
            window_size=C.GARCH_REFIT_WINDOW_SIZE_DEFAULT,
            dist_preference="auto",
            keep_nu_between_refits=True,
            var_alphas=var_alphas,
        )
    return fore_garch, metr_garch


def _compute_baseline_forecasts(
    e_valid: np.ndarray,
    pos_test: np.ndarray,
    ewma_lambda: float,
    rolling_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all baseline volatility forecasts.

    Args:
        e_valid: Valid residuals array.
        pos_test: Test positions in valid array.
        ewma_lambda: EWMA decay factor.
        rolling_window: Rolling window size.

    Returns:
        Tuple of (s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3).
    """
    s2_ewma = _ewma_forecast(e_valid, pos_test, lam=ewma_lambda)
    s2_roll_var = _rolling_var_forecast(e_valid, pos_test, window=rolling_window)
    s2_roll_std = s2_roll_var.copy()
    s2_arch1 = _arch1_forecast(e_valid, pos_test)
    s2_har3 = _har3_forecast(e_valid, pos_test)
    return s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3


def _align_garch_forecasts(
    s2_garch: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Align GARCH forecasts to target size.

    Args:
        s2_garch: GARCH variance forecasts.
        target_size: Target array size.

    Returns:
        Aligned GARCH forecasts array.
    """
    if s2_garch.size == target_size:
        return s2_garch
    if s2_garch.size >= target_size:
        return s2_garch[:target_size]
    pad = np.full(target_size - s2_garch.size, float("nan"))
    return np.concatenate([s2_garch, pad])


def _compute_model_metrics(e_test: np.ndarray, s2: np.ndarray) -> dict[str, float]:
    """Compute volatility forecast metrics for a single model.

    Args:
        e_test: Test residuals.
        s2: Forecast variance.

    Returns:
        Dictionary with qlike, mse, mae, rmse, and r2.
    """
    return {
        "qlike": _qlike(e_test, s2),
        "mse": _mse_variance(e_test, s2),
        "mae": _mae_variance(e_test, s2),
        "rmse": _rmse_variance(e_test, s2),
        "r2": _r2_variance(e_test, s2),
    }


def _rank_models(metrics: dict[str, dict[str, float]]) -> list[tuple[str, float, str]]:
    """Rank models by QLIKE score.

    Args:
        metrics: Dictionary with model metrics.

    Returns:
        List of (model_name, qlike_score, rank) tuples, sorted by QLIKE.
    """
    model_scores = []
    for model_name, model_metrics in metrics.items():
        if model_name == "n_test":
            continue
        qlike = model_metrics.get("qlike", float("inf"))
        if not np.isnan(qlike):
            model_scores.append((model_name, qlike))

    model_scores.sort(key=lambda x: x[1])  # Sort by QLIKE (lower is better)
    return [(name, score, f"#{i+1}") for i, (name, score) in enumerate(model_scores)]


def _compute_metrics(
    e_test: np.ndarray,
    s2_garch: np.ndarray,
    s2_ewma: np.ndarray,
    s2_roll_var: np.ndarray,
    s2_roll_std: np.ndarray,
    s2_arch1: np.ndarray,
    s2_har3: np.ndarray,
) -> dict[str, Any]:
    """Compute volatility forecast metrics for all models.

    Args:
        e_test: Test residuals.
        s2_garch: GARCH variance forecasts.
        s2_ewma: EWMA variance forecasts.
        s2_roll_var: Rolling variance forecasts.
        s2_roll_std: Rolling std forecasts.
        s2_arch1: ARCH(1) variance forecasts.
        s2_har3: HAR(3) variance forecasts.

    Returns:
        Metrics dictionary with QLIKE, MSE, MAE, RMSE, R², and rankings.
    """
    models_metrics = {
        "arima_garch": _compute_model_metrics(e_test, s2_garch),
        "ewma": _compute_model_metrics(e_test, s2_ewma),
        "roll_var": _compute_model_metrics(e_test, s2_roll_var),
        "roll_std": _compute_model_metrics(e_test, s2_roll_std),
        "arch1": _compute_model_metrics(e_test, s2_arch1),
        "har3": _compute_model_metrics(e_test, s2_har3),
    }

    rankings = _rank_models(models_metrics)
    ranking_dict = {name: {"rank": rank, "qlike": qlike} for name, qlike, rank in rankings}

    return {
        "n_test": int(e_test.size),
        **models_metrics,
        "rankings": ranking_dict,
    }


def run_vol_backtest(
    df: pd.DataFrame | None = None,
    *,
    ewma_lambda: float | None = None,
    rolling_window: int | None = None,
    var_alphas: list[float] | None = None,
    refit_every: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run volatility backtest: EGARCH vs baselines on same test split.

    Args:
        df: Optional input DataFrame. If provided, uses it for residuals and
            passes it to the EGARCH runner. If None, automatically loads the
            default GARCH dataset from `C.GARCH_DATASET_FILE` to extract
            residuals, and runs EGARCH from artifacts. This removes the edge
            case when no DataFrame is supplied.
        ewma_lambda: EWMA decay factor (lambda).
        rolling_window: Window size for rolling baselines.
        var_alphas: VaR alphas; passed to the EGARCH runner.
        refit_every: GARCH refit frequency (in test observations). Defaults to 20.

    Returns:
        Tuple of (forecasts DataFrame, metrics dictionary).

    Raises:
        ValueError: If parameters are invalid or data is insufficient.
    """
    ewma_lambda, rolling_window = _validate_backtest_params(ewma_lambda, rolling_window, var_alphas)

    fore_garch, metr_garch = _run_garch_forecasts(df, var_alphas, refit_every=refit_every)

    # Use provided df when available; otherwise, load default GARCH dataset
    # to extract residuals for baselines and metrics alignment.
    data_for_residuals: pd.DataFrame
    if df is None:
        logger.info("No DataFrame provided; loading default GARCH dataset: %s", C.GARCH_DATASET_FILE)
        data_for_residuals = load_garch_dataset()
    else:
        data_for_residuals = df

    e_all, dates_all, test_mask = _select_residual_column(data_for_residuals)
    valid, pos_test = _pos_test_valid(e_all, test_mask)
    if pos_test.size == 0:
        msg = "No valid test observations found"
        raise ValueError(msg)

    e_valid = e_all[valid]
    dates_valid = dates_all[valid]
    e_test = e_valid[pos_test]
    dates_test = dates_valid[pos_test]

    s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3 = _compute_baseline_forecasts(
        e_valid, pos_test, ewma_lambda, rolling_window
    )

    s2_garch_raw = np.asarray(fore_garch.loc[:, "sigma2_forecast"].to_numpy(), dtype=np.float64)
    s2_garch = _align_garch_forecasts(s2_garch_raw, dates_test.size)

    forecasts = _merge_outputs(
        dates_test, e_test, s2_garch, s2_ewma, s2_roll_var, s2_roll_std, s2_arch1, s2_har3
    )

    metrics = _compute_metrics(
        e_test,
        s2_garch,
        s2_ewma,
        s2_roll_var,
        s2_roll_std,
        s2_arch1,
        s2_har3,
    )

    return forecasts, metrics


def _plot_volatility_forecasts(
    dates: np.ndarray,
    e_test: np.ndarray,
    forecasts: pd.DataFrame,
) -> None:
    """Plot volatility forecasts comparison for all models.

    Args:
        dates: Test dates array.
        e_test: Test residuals.
        forecasts: DataFrame with all model forecasts.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
        return

    realized_var = e_test**2

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    models_config = [
        ("arima_garch", "ARIMA-GARCH", axes[0, 0]),
        ("har3", "HAR(3)", axes[0, 1]),
        ("ewma", "EWMA", axes[1, 0]),
        ("arch1", "ARCH(1)", axes[1, 1]),
        ("roll_var", "Rolling Variance", axes[2, 0]),
        ("roll_std", "Rolling Std", axes[2, 1]),
    ]

    for model_key, model_name, ax in models_config:
        s2_col = f"s2_{model_key}" if model_key != "arima_garch" else "s2_arima_garch"
        if s2_col not in forecasts.columns:
            continue

        s2_forecast = forecasts[s2_col].to_numpy()
        m = np.isfinite(realized_var) & np.isfinite(s2_forecast)

        ax.plot(
            dates[m],
            realized_var[m],
            label="Realized (e²)",
            alpha=0.6,
            linewidth=1.5,
            color="#1f77b4",
        )
        ax.plot(
            dates[m],
            s2_forecast[m],
            label="Forecast (σ²)",
            alpha=0.8,
            linewidth=1.5,
            color="#ff7f0e",
        )
        ax.set_title(f"{model_name}", fontweight="bold")
        ax.set_ylabel("Variance")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Volatility Forecasts Comparison: Realized vs Forecasted Variance",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    C.VOL_BACKTEST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.VOL_BACKTEST_VOLATILITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved volatility forecasts plot: %s", C.VOL_BACKTEST_VOLATILITY_PLOT)


def save_vol_backtest_outputs(forecasts: pd.DataFrame, metrics: dict[str, Any]) -> None:
    """Persist volatility backtest forecasts, metrics, and plots to disk.

    Creates directories as needed under constants paths.
    """
    C.VOL_BACKTEST_FORECASTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(C.VOL_BACKTEST_FORECASTS_FILE, index=False)
    C.VOL_BACKTEST_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with C.VOL_BACKTEST_METRICS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)

    dates = pd.to_datetime(forecasts["date"]).to_numpy()
    e_test = forecasts["e"].to_numpy()
    _plot_volatility_forecasts(dates, e_test, forecasts)

    logger.info(
        "Saved volatility backtest outputs: %s, %s, %s",
        C.VOL_BACKTEST_FORECASTS_FILE,
        C.VOL_BACKTEST_METRICS_FILE,
        C.VOL_BACKTEST_VOLATILITY_PLOT,
    )


__all__ = [
    "run_vol_backtest",
    "save_vol_backtest_outputs",
    "run_rolling_garch_from_artifacts",
]
