"""Classic evaluation metrics for GARCH volatility forecasts.

Includes:
- Variance forecast losses: QLIKE, MSE, MAE
- Mincer-Zarnowitz regression on e_t^2 vs sigma_t^2
- VaR backtests: Kupiec POF and Christoffersen independence + combined

All functions are small, typed, and dependency-light (SciPy optional for p-values).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_CALIBRATION_EPS,
    GARCH_DATASET_FILE,
    GARCH_EVAL_DEFAULT_ALPHAS,
    GARCH_EVAL_DEFAULT_SLOPE,
    GARCH_EVAL_EPSILON,
    GARCH_EVAL_HALF,
    GARCH_EVAL_METRICS_FILE,
    GARCH_EVAL_MIN_ALPHA,
    GARCH_STUDENT_NU_MIN,
    GARCH_VARIANCE_OUTPUTS_FILE,
)
from src.garch.garch_params.estimation import egarch11_variance
from src.garch.garch_eval.distributions import skewt_ppf
from src.utils import get_logger

logger = get_logger(__name__)


def _chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function; returns NaN if SciPy unavailable."""
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


def qlike_loss(e: np.ndarray, sigma2: np.ndarray) -> float:
    """Return average QLIKE loss: log(sigma2) + e^2 / sigma2.

    Args:
        e: Residuals aligned to sigma2 (length n)
        sigma2: Conditional variance sequence (positive; length n)

    Returns:
        Mean QLIKE over finite pairs.
    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.log(s2[m]) + (e[m] ** 2) / s2[m]))


def mse_mae_variance(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Compute MSE and MAE between realized e^2 and forecast sigma^2."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    if not np.any(m):
        return {"mse": float("nan"), "mae": float("nan")}
    y = e[m] ** 2
    f = s2[m]
    return {
        "mse": float(np.mean((y - f) ** 2)),
        "mae": float(np.mean(np.abs(y - f))),
    }


def _compute_mz_pvalues(
    beta: np.ndarray,
    x_mat: np.ndarray,
    ss_res: float,
) -> dict[str, float]:
    """Compute p-values for Mincer-Zarnowitz regression coefficients.

    Args:
        beta: Regression coefficients.
        x_mat: Design matrix.
        ss_res: Sum of squared residuals.

    Returns:
        Dictionary with t-statistics and p-values.
    """
    try:
        from scipy.stats import t as student_t  # type: ignore

        n, k = x_mat.shape
        s2_err = ss_res / max(1, n - k)
        xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
        se = np.sqrt(np.diag(xtx_inv) * s2_err)
        t_intercept = float(beta[0] / se[0]) if se[0] > 0 else float("nan")
        t_slope = float(beta[1] / se[1]) if se[1] > 0 else float("nan")
        dof = max(1, n - k)
        p_intercept = float(2.0 * (1.0 - student_t.cdf(abs(t_intercept), df=dof)))
        p_slope = float(2.0 * (1.0 - student_t.cdf(abs(t_slope), df=dof)))
        return {
            "t_intercept": t_intercept,
            "t_slope": t_slope,
            "p_intercept": p_intercept,
            "p_slope": p_slope,
        }
    except Exception as ex:
        logger.debug("SciPy unavailable for MZ p-values; continuing without: %s", ex)
        return {}


def mincer_zarnowitz(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Run Mincer-Zarnowitz regression: e^2 = c + b * sigma^2 + u.

    Returns intercept c, slope b, R^2 and (optionally) p-values when SciPy is available.
    """
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2)
    y = (e[m] ** 2).astype(float)
    x = s2[m].astype(float)
    min_obs = 2
    if y.size < min_obs:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}

    x_mat = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
    y_hat = x_mat @ beta
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_hat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    out: dict[str, float] = {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(r2),
    }
    out.update(_compute_mz_pvalues(beta, x_mat, ss_res))
    return out


def mz_calibration_params(e: np.ndarray, sigma2: np.ndarray) -> dict[str, float]:
    """Return intercept and slope (a,b) from MZ regression to calibrate variance.

    Why: If b>1 (like in your diagnostics), the raw sigma² underestimates
    realized variance. Typically use multiplicative calibration: h_adj = b * h
    (intercept usually not significant and can cause instability).
    """
    mz = mincer_zarnowitz(e, sigma2)
    return {
        "intercept": float(mz.get("intercept", np.nan)),
        "slope": float(mz.get("slope", np.nan)),
        "p_intercept": float(mz.get("p_intercept", np.nan)),
        "p_slope": float(mz.get("p_slope", np.nan)),
    }


def apply_mz_calibration(
    sigma2: np.ndarray,
    intercept: float,
    slope: float,
    *,
    eps: float = GARCH_CALIBRATION_EPS,
    use_intercept: bool = False,
) -> np.ndarray:
    """Apply MZ calibration: h_adj = max(eps, a + b * h) or h_adj = b * h.

    Args:
        sigma2: Variance array to calibrate.
        intercept: MZ regression intercept.
        slope: MZ regression slope.
        eps: Minimum variance threshold.
        use_intercept: If False, use multiplicative calibration only (slope * h).
                      If True, use full additive calibration (intercept + slope * h).

    Returns:
        Calibrated variance array.
    """
    s2 = np.asarray(sigma2, dtype=float)
    if use_intercept:
        h_adj = intercept + slope * s2
    else:
        # Multiplicative calibration only (more stable)
        h_adj = slope * s2
    return np.asarray(np.maximum(eps, h_adj), dtype=float)


def _var_quantile(alpha: float, dist: str, nu: float | None, lambda_skew: float | None = None) -> float:
    """Quantile for VaR under Normal/Student/Skew-t innovations (left tail)."""
    try:
        from scipy.stats import norm, t  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is expected in project
        msg = "SciPy required for VaR backtests"
        raise RuntimeError(msg) from exc

    if dist.lower() == "student":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN:
            msg = "Student-t requires nu>2 for VaR"
            raise ValueError(msg)
        return float(t.ppf(alpha, df=nu))
    if dist.lower() == "skewt":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN or lambda_skew is None:
            msg = "Skew-t requires nu>2 and lambda for VaR"
            raise ValueError(msg)
        return float(skewt_ppf(alpha, nu, lambda_skew))
    return float(norm.ppf(alpha))


def _build_var_series(
    sigma2: np.ndarray, alpha: float, dist: str, nu: float | None, lambda_skew: float | None = None
) -> np.ndarray:
    """Return VaR_t series at level alpha with zero mean: VaR = q_alpha * sigma_t."""
    s2 = np.asarray(sigma2, dtype=float).ravel()
    q = _var_quantile(float(alpha), dist, nu, lambda_skew)
    return q * np.sqrt(s2)


def empirical_quantiles(z: np.ndarray, alphas: list[float]) -> dict[float, float]:
    """Return empirical left-tail quantiles of standardized residuals.

    Args:
        z: Standardized residuals (train) ~ iid under well-specified model.
        alphas: Tail probabilities (e.g., [0.01, 0.05]).
    """
    zz = np.asarray(z, dtype=float)
    zz = zz[np.isfinite(zz)]
    out: dict[float, float] = {}
    for a in alphas:
        a_f = float(a)
        a_f = min(max(a_f, GARCH_EVAL_MIN_ALPHA), GARCH_EVAL_HALF)
        out[a] = float(np.quantile(zz, a_f)) if zz.size else float("nan")
    return out


def kupiec_pof_test(hits: np.ndarray, alpha: float) -> dict[str, float]:
    """Kupiec Proportion-of-Failures (POF) test.

    Args:
        hits: 1 if return < VaR, else 0.
        alpha: Target tail probability (e.g. 0.01).
    Returns:
        Dict with n, x, hit_rate, lr_uc, p_value.
    """
    h = np.asarray(hits, dtype=float).ravel()
    m = np.isfinite(h)
    h = h[m]
    n = int(h.size)
    x = int(np.sum(h > GARCH_EVAL_HALF))
    if n == 0:
        return {
            "n": 0.0,
            "x": 0.0,
            "hit_rate": float("nan"),
            "lr_uc": float("nan"),
            "p_value": float("nan"),
        }
    phat = x / max(1, n)

    # Likelihood ratio for unconditional coverage
    def _lnp(p: float) -> float:
        p = min(max(p, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        return (n - x) * np.log(1 - p) + x * np.log(p)

    lr_uc = -2.0 * (_lnp(alpha) - _lnp(phat))
    p_val = _chi2_sf(float(lr_uc), df=1)
    return {
        "n": float(n),
        "x": float(x),
        "hit_rate": float(phat),
        "lr_uc": float(lr_uc),
        "p_value": float(p_val),
    }


def christoffersen_ind_test(hits: np.ndarray) -> dict[str, float]:
    """Christoffersen independence test (first-order Markov).

    Returns LR_ind and p-value (df=1) and the transition counts.
    """
    h = (np.asarray(hits, dtype=float).ravel() > GARCH_EVAL_HALF).astype(int)
    if h.size <= 1:
        return {
            "lr_ind": float("nan"),
            "p_value": float("nan"),
            "n00": 0.0,
            "n01": 0.0,
            "n10": 0.0,
            "n11": 0.0,
        }
    n00 = np.sum((h[1:] == 0) & (h[:-1] == 0))
    n01 = np.sum((h[1:] == 1) & (h[:-1] == 0))
    n10 = np.sum((h[1:] == 0) & (h[:-1] == 1))
    n11 = np.sum((h[1:] == 1) & (h[:-1] == 1))
    n0 = max(1, n00 + n01)
    n1 = max(1, n10 + n11)
    p01 = n01 / n0
    p11 = n11 / n1
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    # Likelihood ratio statistic for independence
    def _ll(p0: float, p1: float) -> float:
        p0 = min(max(p0, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        p1 = min(max(p1, GARCH_EVAL_EPSILON), 1 - GARCH_EVAL_EPSILON)
        return n00 * np.log(1 - p0) + n01 * np.log(p0) + n10 * np.log(1 - p1) + n11 * np.log(p1)

    l1 = _ll(p01, p11)
    l0 = _ll(p, p)
    lr_ind = -2.0 * (l0 - l1)
    p_val = _chi2_sf(float(lr_ind), df=1)
    return {
        "lr_ind": float(lr_ind),
        "p_value": float(p_val),
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
    }


def var_backtest_metrics(
    e: np.ndarray,
    sigma2: np.ndarray,
    *,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float],
) -> dict[str, dict[str, float]]:
    """Compute VaR backtests (Kupiec/Christoffersen) for multiple alphas."""
    e = np.asarray(e, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    m = np.isfinite(e) & np.isfinite(s2) & (s2 > 0)
    out: dict[str, dict[str, float]] = {}
    for a in alphas:
        var_t = _build_var_series(s2[m], a, dist, nu, lambda_skew)
        hits = (e[m] < var_t).astype(int)
        kup = kupiec_pof_test(hits, a)
        ind = christoffersen_ind_test(hits)
        out[str(a)] = {
            "n": kup["n"],
            "violations": kup["x"],
            "hit_rate": kup["hit_rate"],
            "lr_uc": kup["lr_uc"],
            "p_uc": kup["p_value"],
            "lr_ind": ind["lr_ind"],
            "p_ind": ind["p_value"],
            "lr_cc": float(kup["lr_uc"] + ind["lr_ind"]),
            "p_cc": _chi2_sf(float(kup["lr_uc"] + ind["lr_ind"]), df=2),
        }
    return out


def _prepare_residuals_from_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare and filter residuals from dataset, preserving index alignment.

    Args:
        dataset: Input dataset DataFrame.

    Returns:
        Tuple of (sorted_dataframe, all_residuals, valid_mask, filtered_residuals).
        Returns empty arrays if no valid residuals found.
    """
    df_sorted = dataset.sort_values("date").reset_index(drop=True)

    # Build residual series and preserve index alignment
    series = pd.to_numeric(df_sorted.get("arima_residual_return"), errors="coerce")
    resid = np.asarray(series, dtype=float)
    valid_mask = np.isfinite(resid)
    if not np.any(valid_mask):
        return (
            df_sorted,
            np.array([], dtype=float),
            np.array([], dtype=bool),
            np.array([], dtype=float),
        )

    # Filtered contiguous residuals for variance recursion
    resid_f = resid[valid_mask]
    return df_sorted, resid, valid_mask, resid_f


def _compute_variance_path_for_test(
    resid_f: np.ndarray,
    model_name: str,
    params: dict[str, float],
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Compute variance path for filtered residuals using EGARCH(1,1).

    Args:
        resid_f: Filtered residual series.
        model_name: Model name (e.g., 'egarch_normal', 'egarch_skewt').
        params: Model parameters dictionary.
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom (for Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
        Variance path array, or empty array if computation failed.
    """
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha", np.nan))
    beta = float(params.get("beta", np.nan))
    gamma_val = params.get("gamma")
    gamma = float(gamma_val) if gamma_val is not None else 0.0

    s2_f = egarch11_variance(
        resid_f, omega, alpha, gamma, beta, dist=dist, nu=nu, lambda_skew=lambda_skew
    )

    # If recursion failed, return empty to signal no valid metrics
    if not (np.all(np.isfinite(s2_f)) and np.all(s2_f > 0)):
        return np.array([], dtype=float)
    return s2_f


def _extract_aligned_test_indices(
    df_sorted: pd.DataFrame,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Extract aligned test indices from sorted dataset.

    Args:
        df_sorted: Sorted dataset DataFrame.
        valid_mask: Boolean mask for valid residuals.

    Returns:
        Array of positions in filtered arrays for test data, or empty array if none.
    """
    # Extract aligned test block using masks on the original index
    test_mask = (df_sorted["split"].astype(str) == "test").to_numpy()
    idx_all = np.arange(df_sorted.shape[0])
    idx_valid = idx_all[valid_mask]
    idx_test = idx_all[test_mask]
    idx_test_valid = np.intersect1d(idx_valid, idx_test, assume_unique=False)
    if idx_test_valid.size == 0:
        return np.array([], dtype=int)

    # Map original indices -> positions in filtered arrays
    pos_in_valid = -np.ones(df_sorted.shape[0], dtype=int)
    pos_in_valid[idx_valid] = np.arange(idx_valid.size)
    pos_test = pos_in_valid[idx_test_valid]
    # Keep order of time by sorting positions
    pos_test.sort()
    return pos_test


def _load_test_resid_sigma2(
    params: dict[str, float],
    dataset: pd.DataFrame,
    *,
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned residuals and sigma² on the test split for the chosen model.

    Why: Previous implementation dropped NaNs before slicing, which broke
    alignment. This version uses EGARCH(1,1) parameters correctly.

    Args:
        params: Model parameters dictionary.
        dataset: Input dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
        Tuple of (test_residuals, test_variance).
    """
    # Prepare residuals
    df_sorted, resid, valid_mask, resid_f = _prepare_residuals_from_dataset(dataset)
    if resid_f.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Compute variance path
    s2_f = _compute_variance_path_for_test(resid_f, model_name, params, dist, nu, lambda_skew)
    if s2_f.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Extract aligned test indices
    pos_test = _extract_aligned_test_indices(df_sorted, valid_mask)
    if pos_test.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Extract test data
    e_test = resid_f[pos_test]
    s2_test = s2_f[pos_test]
    return e_test.astype(float), s2_test.astype(float)


def _load_dataset_for_metrics() -> pd.DataFrame:
    """Load dataset for metrics computation, preferring variance outputs CSV.

    Returns:
        Dataset DataFrame.
    """
    try:
        dataset_df = pd.read_csv(GARCH_VARIANCE_OUTPUTS_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    except Exception:
        dataset_df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    return dataset_df


def _filter_test_data(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter test data to keep only finite and positive variance values.

    Args:
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
        Tuple of (filtered_residuals, filtered_variance).
    """
    m = np.isfinite(e_test) & np.isfinite(s2_test) & (s2_test > 0)
    return e_test[m], s2_test[m]


def _compute_variance_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> dict[str, float]:
    """Compute variance forecast metrics (QLIKE, MSE, MAE).

    Args:
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
        Dictionary with variance metrics.
    """
    out_losses = mse_mae_variance(e_test, s2_test)
    return {
        "n_test": int(e_test.size),
        "qlike": qlike_loss(e_test, s2_test),
        "mse_var": out_losses["mse"],
        "mae_var": out_losses["mae"],
    }


def _apply_mz_calibration_if_requested(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    use_mz_calibration: bool,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    """Apply MZ calibration to test variances if requested.

    Args:
        e_test: Test residuals.
        s2_test: Test variance.
        use_mz_calibration: Whether to apply calibration.

    Returns:
        Tuple of (calibrated_variance, mz_results, intercept, slope).
    """
    mz_results = mincer_zarnowitz(e_test, s2_test)
    mz_intercept = mz_results.get("intercept", 0.0)
    mz_slope = mz_results.get("slope", GARCH_EVAL_DEFAULT_SLOPE)

    if use_mz_calibration:
        s2_calibrated = apply_mz_calibration(s2_test, mz_intercept, mz_slope, use_intercept=False)
        logger.info(
            f"Applied MZ calibration (multiplicative): slope={mz_slope:.3f} "
            f"(intercept={mz_intercept:.6f} ignored for stability)"
        )
    else:
        s2_calibrated = s2_test

    return s2_calibrated, mz_results, mz_intercept, mz_slope


def _add_comparison_metrics(
    out: dict[str, object],
    e_test: np.ndarray,
    s2_test: np.ndarray,
    s2_calibrated: np.ndarray,
    use_mz_calibration: bool,
) -> None:
    """Add comparison metrics between original and calibrated variances.

    Args:
        out: Output dictionary to update.
        e_test: Test residuals.
        s2_test: Original test variance.
        s2_calibrated: Calibrated test variance.
        use_mz_calibration: Whether calibration was applied.
    """
    if use_mz_calibration:
        variance_metrics_original = _compute_variance_metrics(e_test, s2_test)
        out["variance_metrics_original"] = variance_metrics_original
        mz_calibrated = mincer_zarnowitz(e_test, s2_calibrated)
        out["mz_calibrated"] = {f"mz_{k}": v for k, v in mz_calibrated.items()}


def _compute_all_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    dist: str,
    nu: float | None,
    alphas: list[float],
    *,
    lambda_skew: float | None = None,
    use_mz_calibration: bool = True,
) -> dict[str, object]:
    """Compute all GARCH evaluation metrics.

    Args:
        e_test: Test residuals.
        s2_test: Test variance.
        dist: Distribution type.
        nu: Degrees of freedom.
        alphas: VaR alpha levels.
        use_mz_calibration: Whether to apply MZ calibration.

    Returns:
        Dictionary with all metrics.
    """
    out: dict[str, object] = {}

    # Apply MZ calibration if requested
    s2_calibrated, mz_results, mz_intercept, mz_slope = _apply_mz_calibration_if_requested(
        e_test, s2_test, use_mz_calibration
    )

    # Add variance metrics (use calibrated variance if calibration is applied)
    s2_for_metrics = s2_calibrated if use_mz_calibration else s2_test
    variance_metrics = _compute_variance_metrics(e_test, s2_for_metrics)
    out.update(variance_metrics)

    # Add Mincer-Zarnowitz metrics (on original variances for diagnostic)
    out.update({f"mz_{k}": v for k, v in mz_results.items()})

    # Add MZ calibration parameters
    out["mz_calibration"] = {
        "intercept": float(mz_intercept),
        "slope": float(mz_slope),
        "applied": use_mz_calibration,
    }

    # Add VaR backtests on original variances (never MZ before VaR)
    out["var_backtests"] = var_backtest_metrics(
        e_test, s2_test, dist=dist, nu=nu, lambda_skew=lambda_skew, alphas=alphas
    )

    # Add comparison metrics (original vs calibrated)
    _add_comparison_metrics(out, e_test, s2_test, s2_calibrated, use_mz_calibration)

    return out


def compute_classic_metrics_from_artifacts(
    *,
    params: dict[str, float],
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
    alphas: list[float] | None = None,
    apply_mz_calibration: bool = False,
) -> dict[str, object]:
    """Compute classic GARCH metrics on the test split and return a summary dict.

    Args:
        params: Model parameters dictionary.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).
        alphas: VaR alpha levels (default: [0.01, 0.05]).
        apply_mz_calibration: Whether to apply MZ calibration to variances.

    Returns:
        Dictionary with all computed metrics.
    """
    if alphas is None:
        alphas = list(GARCH_EVAL_DEFAULT_ALPHAS)

    # Load dataset
    dataset_df = _load_dataset_for_metrics()

    # Load test residuals and variance
    e_test, s2_test = _load_test_resid_sigma2(
        params, dataset_df, model_name=model_name, dist=dist, nu=nu, lambda_skew=lambda_skew
    )

    # Filter test data
    e_test, s2_test = _filter_test_data(e_test, s2_test)

    # Compute all metrics with optional MZ calibration
    return _compute_all_metrics(
        e_test, s2_test, dist, nu, alphas, lambda_skew=lambda_skew, use_mz_calibration=apply_mz_calibration
    )


def save_metrics_json(payload: dict[str, object]) -> None:
    """Persist metrics to JSON path in constants."""
    GARCH_EVAL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GARCH_EVAL_METRICS_FILE.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved GARCH evaluation metrics to: %s", GARCH_EVAL_METRICS_FILE)


__all__ = [
    "qlike_loss",
    "mse_mae_variance",
    "mincer_zarnowitz",
    "kupiec_pof_test",
    "christoffersen_ind_test",
    "var_backtest_metrics",
    "compute_classic_metrics_from_artifacts",
    "save_metrics_json",
]
