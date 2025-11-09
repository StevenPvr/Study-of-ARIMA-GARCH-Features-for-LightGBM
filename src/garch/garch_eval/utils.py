"""Utility functions for GARCH evaluation.

Contains helper functions for:
- Quantile and VaR calculations
- AIC computation
- Data loading and preparation
- Variance path computation
- MZ calibration
- Statistical utilities
- File operations
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_AIC_MULTIPLIER,
    GARCH_EVAL_DEFAULT_SLOPE,
    GARCH_FORECASTS_FILE,
    GARCH_STUDENT_NU_MIN,
    GARCH_VARIANCE_OUTPUTS_FILE,
)
from src.garch.garch_eval.distributions import skewt_ppf
from src.garch.garch_params.estimation import egarch11_variance
from src.garch.structure_garch.utils import prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Quantile and VaR Utilities
# ============================================================================


def quantile(dist: str, p: float, nu: float | None, lambda_skew: float | None = None) -> float:
    """Return left-tail quantile for Normal, Student-t, or Hansen skew-t.

    Args:
    ----
        dist: Distribution type ('normal', 'student', or 'skewt').
        p: Probability level.
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Quantile value.

    """
    from scipy.stats import norm, t  # type: ignore

    dist_l = dist.lower()
    if dist_l == "skewt":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN or lambda_skew is None:
            msg = "Skew-t requires nu>2 and lambda for quantiles"
            raise ValueError(msg)
        return float(skewt_ppf(float(p), float(nu), float(lambda_skew)))
    if dist_l == "student":
        if nu is None or nu <= GARCH_STUDENT_NU_MIN:
            msg = "Student-t requires nu>2 for quantiles"
            raise ValueError(msg)
        return float(t.ppf(p, df=float(nu)))
    return float(norm.ppf(p))


def var_quantile(
    alpha: float, dist: str, nu: float | None, lambda_skew: float | None = None
) -> float:
    """Quantile for VaR under Normal/Student/Skew-t innovations (left tail).

    Args:
    ----
        alpha: Tail probability level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        VaR quantile value.

    """
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


def build_var_series(
    sigma2: np.ndarray,
    alpha: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Return VaR_t series at level alpha with zero mean: VaR = q_alpha * sigma_t.

    Args:
    ----
        sigma2: Variance series.
        alpha: Tail probability level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        VaR series.

    """
    s2 = np.asarray(sigma2, dtype=float).ravel()
    q = var_quantile(float(alpha), dist, nu, lambda_skew)
    return q * np.sqrt(s2)


# ============================================================================
# AIC and Model Selection Utilities
# ============================================================================


def aic(ll: float, k: int) -> float:
    """Calculate AIC: 2k - 2*loglik.

    Args:
    ----
        ll: Log-likelihood value.
        k: Number of parameters.

    Returns:
    -------
        AIC score.

    """
    return GARCH_EVAL_AIC_MULTIPLIER * k - GARCH_EVAL_AIC_MULTIPLIER * float(ll)


def collect_converged_candidates(
    payload: dict,
    keys: list[str],
    k_params: dict[str, int],
) -> list[tuple[str, dict, float]]:
    """Collect converged model candidates with their AIC scores.

    Args:
    ----
        payload: Estimation payload dictionary.
        keys: List of model keys to check.
        k_params: Dictionary mapping model names to parameter counts.

    Returns:
    -------
        List of tuples (name, params_dict, aic_score).

    """
    cand: list[tuple[str, dict, float]] = []
    for name in keys:
        d = payload.get(name)
        if isinstance(d, dict) and d.get("converged"):
            k = k_params[name]
            cand.append((name, d, aic(float(d["loglik"]), k)))
    return cand


def choose_best_from_estimation(
    payload: dict,
) -> tuple[dict[str, float], str, float | None, float | None]:
    """Pick best model from estimation JSON using AIC and preference order.

    Preference order on ties: skew-t → student → normal.

    Args:
    ----
        payload: Estimation payload dictionary.

    Returns:
    -------
        Tuple of (params_dict, model_name, nu, lambda_skew).

    """
    keys = ["egarch_skewt", "egarch_student", "egarch_normal"]
    k_params = {"egarch_normal": 4, "egarch_student": 5, "egarch_skewt": 6}

    cand = collect_converged_candidates(payload, keys, k_params)
    if not cand:
        msg = "No converged volatility model found in estimation file"
        raise RuntimeError(msg)

    # Sort by AIC then by preference order
    order = {k: i for i, k in enumerate(keys)}
    cand.sort(key=lambda t: (t[2], order.get(t[0], 999)))
    name, params, _ = cand[0]
    nu_val = params.get("nu")
    nu = float(nu_val) if nu_val is not None else None
    lambda_val = params.get("lambda")
    lambda_skew = float(lambda_val) if lambda_val is not None else None
    return params, name, nu, lambda_skew


# ============================================================================
# Data Loading Utilities
# ============================================================================


def load_model_params() -> tuple[
    dict[str, float],
    str,
    str,
    float | None,
    float | None,
    float | None,
]:
    """Load and extract model parameters from estimation file.

    Returns
    -------
        Tuple of (params_dict, model_name, dist, nu, gamma, lambda_skew).

    """
    with GARCH_ESTIMATION_FILE.open() as f:
        est = json.load(f)
    best, name, nu, lambda_skew = choose_best_from_estimation(est)

    # Map name to distribution (EGARCH-only)
    if "skewt" in name:
        dist = "skewt"
    elif "student" in name:
        dist = "student"
    else:
        dist = "normal"

    # Extract parameters
    omega = float(best["omega"])  # type: ignore[index]
    alpha = float(best["alpha"])  # type: ignore[index]
    beta = float(best["beta"])  # type: ignore[index]
    gamma_val = best.get("gamma")
    gamma = float(gamma_val) if gamma_val is not None else None

    params: dict[str, float] = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
    }
    return params, name, dist, nu, gamma, lambda_skew


def load_dataset_for_metrics() -> pd.DataFrame:
    """Load dataset for metrics computation, preferring variance outputs CSV.

    Returns
    -------
        Dataset DataFrame.

    """
    try:
        dataset_df = pd.read_csv(GARCH_VARIANCE_OUTPUTS_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    except Exception:
        dataset_df = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]
    return dataset_df


def load_and_prepare_residuals() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load dataset and prepare filtered residuals for train and all data.

    Returns
    -------
        Tuple of (dataframe, train_residuals, all_residuals).
        train_residuals: Only training residuals (for forecast initialization).
        all_residuals: All residuals (for variance path computation if needed).

    """
    data = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]

    # Get train residuals for forecast initialization (no data leakage)
    df_train = data.loc[data["split"] == "train"].copy()
    resid_train = prepare_residuals(df_train, use_test_only=False)
    resid_train = resid_train[np.isfinite(resid_train)]

    # Get all residuals (for potential future use, but not for forecast init)
    resid_all = prepare_residuals(data, use_test_only=False)
    resid_all = resid_all[np.isfinite(resid_all)]

    return data, resid_train, resid_all


def prepare_residuals_from_dataset(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare and filter residuals from dataset, preserving index alignment.

    Args:
    ----
        dataset: Input dataset DataFrame.

    Returns:
    -------
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


def extract_aligned_test_indices(
    df_sorted: pd.DataFrame,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Extract aligned test indices from sorted dataset.

    Args:
    ----
        df_sorted: Sorted dataset DataFrame.
        valid_mask: Boolean mask for valid residuals.

    Returns:
    -------
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


def load_test_resid_sigma2(
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
    ----
        params: Model parameters dictionary.
        dataset: Input dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (test_residuals, test_variance).

    """
    # Prepare residuals
    df_sorted, _resid, valid_mask, resid_f = prepare_residuals_from_dataset(dataset)
    if resid_f.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Compute variance path
    s2_f = compute_variance_path_for_test(resid_f, model_name, params, dist, nu, lambda_skew)
    if s2_f.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Extract aligned test indices
    pos_test = extract_aligned_test_indices(df_sorted, valid_mask)
    if pos_test.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Extract test data
    e_test = resid_f[pos_test]
    s2_test = s2_f[pos_test]
    return e_test.astype(float), s2_test.astype(float)


# ============================================================================
# Variance Path Computation Utilities
# ============================================================================


def compute_variance_path(
    resid_all: np.ndarray,
    model_name: str,  # noqa: ARG001
    omega: float,
    alpha: float,
    beta: float,
    gamma: float | None,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Compute variance path based on model type.

    Args:
    ----
        resid_all: Residual series.
        model_name: Model name (e.g., 'egarch_normal', 'egarch_skewt').
        omega: Omega parameter.
        alpha: Alpha parameter.
        beta: Beta parameter.
        gamma: Gamma parameter (for EGARCH/GJR).
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom (for Student-t/Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Variance path array.

    Raises:
    ------
        ValueError: If computed variance path is invalid.

    """
    sigma2_path = egarch11_variance(
        resid_all,
        omega,
        alpha,
        float(gamma or 0.0),
        beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    if not (np.all(np.isfinite(sigma2_path)) and np.all(sigma2_path > 0)):
        msg = "Invalid sigma^2 path computed from artifacts"
        raise ValueError(msg)
    return sigma2_path


def compute_variance_path_for_test(
    resid_f: np.ndarray,
    model_name: str,  # noqa: ARG001
    params: dict[str, float],
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Compute variance path for filtered residuals using EGARCH(1,1).

    Args:
    ----
        resid_f: Filtered residual series.
        model_name: Model name (e.g., 'egarch_normal', 'egarch_skewt').
        params: Model parameters dictionary.
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom (for Skew-t).
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Variance path array, or empty array if computation failed.

    """
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha", np.nan))
    beta = float(params.get("beta", np.nan))
    gamma_val = params.get("gamma")
    gamma = float(gamma_val) if gamma_val is not None else 0.0

    s2_f = egarch11_variance(
        resid_f,
        omega,
        alpha,
        gamma,
        beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    # If recursion failed, return empty to signal no valid metrics
    if not (np.all(np.isfinite(s2_f)) and np.all(s2_f > 0)):
        return np.array([], dtype=float)
    return s2_f


def compute_egarch_forecasts(
    e_last: float,
    s2_last: float,
    horizon: int,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[float, np.ndarray]:
    """Compute EGARCH one-step and multi-step variance forecasts.

    Args:
    ----
        e_last: Last residual.
        s2_last: Last variance.
        horizon: Forecast horizon.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (one-step forecast, multi-step forecasts array).

    """
    from src.garch.garch_params.utils import egarch_kappa as eg_kappa

    # One-step using EGARCH recursion with expected shock terms
    z_last = float(e_last / np.sqrt(s2_last))
    kappa = eg_kappa(dist, nu, lambda_skew)
    ln_next = omega + beta * np.log(s2_last) + alpha * (abs(z_last) - kappa) + gamma * z_last
    s2_1 = float(np.exp(ln_next))

    # Multi-step expectation: E(|z|-kappa)=0, E(z)=0 => log variance recursion
    s2_h = np.empty(horizon, dtype=float)
    log_s2 = float(np.log(s2_last))
    for i in range(horizon):
        log_s2 = omega + beta * log_s2
        s2_h[i] = float(np.exp(log_s2))

    return s2_1, s2_h


def compute_initial_forecasts(
    resid_train: np.ndarray,
    sigma2_path_train: np.ndarray,
    horizon: int,
    omega: float,
    alpha: float,
    gamma: float | None,
    beta: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[float, np.ndarray]:
    """Compute initial EGARCH variance forecasts from training data only.

    Uses only training residuals to initialize forecasts, preventing data leakage.

    Args:
    ----
        resid_train: Training residuals only (no test data).
        sigma2_path_train: Variance path computed on training data only.
        horizon: Forecast horizon.
        omega: Omega parameter.
        alpha: Alpha parameter.
        gamma: Gamma parameter.
        beta: Beta parameter.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (one_step_forecast, multi_step_forecasts).

    """
    if resid_train.size == 0 or sigma2_path_train.size == 0:
        msg = "Training residuals or variance path is empty"
        raise ValueError(msg)

    e_last = float(resid_train[-1])
    s2_last = float(sigma2_path_train[-1])
    s2_1, s2_h = compute_egarch_forecasts(
        e_last,
        s2_last,
        horizon,
        omega,
        alpha,
        float(gamma or 0.0),
        beta,
        dist,
        nu,
        lambda_skew,
    )
    return s2_1, s2_h


# ============================================================================
# MZ Calibration Utilities
# ============================================================================


def compute_mz_pvalues(
    beta: np.ndarray,
    x_mat: np.ndarray,
    ss_res: float,
) -> dict[str, float]:
    """Compute p-values for Mincer-Zarnowitz regression coefficients.

    Args:
    ----
        beta: Regression coefficients.
        x_mat: Design matrix.
        ss_res: Sum of squared residuals.

    Returns:
    -------
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


def filter_test_data(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter test data to keep only finite and positive variance values.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
    -------
        Tuple of (filtered_residuals, filtered_variance).

    """
    m = np.isfinite(e_test) & np.isfinite(s2_test) & (s2_test > 0)
    return e_test[m], s2_test[m]


def compute_variance_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
) -> dict[str, float]:
    """Compute variance forecast metrics (QLIKE, MSE, MAE).

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.

    Returns:
    -------
        Dictionary with variance metrics.

    """
    from src.garch.garch_eval.metrics import mse_mae_variance, qlike_loss

    out_losses = mse_mae_variance(e_test, s2_test)
    return {
        "n_test": int(e_test.size),
        "qlike": qlike_loss(e_test, s2_test),
        "mse_var": out_losses["mse"],
        "mae_var": out_losses["mae"],
    }


def apply_mz_calibration_if_requested(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    use_mz_calibration: bool,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    """Apply MZ calibration to test variances if requested.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.
        use_mz_calibration: Whether to apply calibration.

    Returns:
    -------
        Tuple of (calibrated_variance, mz_results, intercept, slope).

    """
    from src.garch.garch_eval.metrics import apply_mz_calibration, mincer_zarnowitz

    mz_results = mincer_zarnowitz(e_test, s2_test)
    mz_intercept = mz_results.get("intercept", 0.0)
    mz_slope = mz_results.get("slope", GARCH_EVAL_DEFAULT_SLOPE)

    if use_mz_calibration:
        s2_calibrated = apply_mz_calibration(s2_test, mz_intercept, mz_slope, use_intercept=False)
        logger.info(
            "Applied MZ calibration (multiplicative): slope=%.3f "
            "(intercept=%.6f ignored for stability)",
            mz_slope,
            mz_intercept,
        )
    else:
        s2_calibrated = s2_test

    return s2_calibrated, mz_results, mz_intercept, mz_slope


def add_comparison_metrics(
    out: dict[str, object],
    e_test: np.ndarray,
    s2_test: np.ndarray,
    s2_calibrated: np.ndarray,
    use_mz_calibration: bool,
) -> None:
    """Add comparison metrics between original and calibrated variances.

    Args:
    ----
        out: Output dictionary to update.
        e_test: Test residuals.
        s2_test: Original test variance.
        s2_calibrated: Calibrated test variance.
        use_mz_calibration: Whether calibration was applied.

    """
    if use_mz_calibration:
        variance_metrics_original = compute_variance_metrics(e_test, s2_test)
        out["variance_metrics_original"] = variance_metrics_original
        from src.garch.garch_eval.metrics import mincer_zarnowitz

        mz_calibrated = mincer_zarnowitz(e_test, s2_calibrated)
        out["mz_calibrated"] = {f"mz_{k}": v for k, v in mz_calibrated.items()}


def compute_all_metrics(
    e_test: np.ndarray,
    s2_test: np.ndarray,
    dist: str,
    nu: float | None,
    alphas: list[float],
    *,
    lambda_skew: float | None = None,
    use_mz_calibration: bool = False,
) -> dict[str, object]:
    """Compute all GARCH evaluation metrics.

    Args:
    ----
        e_test: Test residuals.
        s2_test: Test variance.
        dist: Distribution type.
        nu: Degrees of freedom.
        alphas: VaR alpha levels.
        use_mz_calibration: Whether to apply MZ calibration.
            Defaults to False to avoid data leakage. Set to True only for diagnostic purposes.

    Returns:
    -------
        Dictionary with all metrics.

    """
    from src.garch.garch_eval.metrics import var_backtest_metrics

    out: dict[str, object] = {}

    # Apply MZ calibration if requested
    s2_calibrated, mz_results, mz_intercept, mz_slope = apply_mz_calibration_if_requested(
        e_test,
        s2_test,
        use_mz_calibration,
    )

    # Add variance metrics (use calibrated variance if calibration is applied)
    s2_for_metrics = s2_calibrated if use_mz_calibration else s2_test
    variance_metrics = compute_variance_metrics(e_test, s2_for_metrics)
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
        e_test,
        s2_test,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        alphas=alphas,
    )

    # Add comparison metrics (original vs calibrated)
    add_comparison_metrics(out, e_test, s2_test, s2_calibrated, use_mz_calibration)

    return out


def apply_mz_calibration_to_forecasts(
    s2_1: float,
    s2_h: np.ndarray,
    params: dict[str, float],
    data: pd.DataFrame,
    model_name: str,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> tuple[float, np.ndarray, float, float]:
    """Apply MZ calibration to variance forecasts.

    WARNING: This function computes MZ calibration parameters using TEST data
    (via load_test_resid_sigma2). Using calibrated forecasts for evaluation
    introduces data leakage and invalidates out-of-sample performance metrics.
    This function should ONLY be used for diagnostic purposes, not for production
    scoring or model comparison.

    Args:
    ----
        s2_1: One-step variance forecast.
        s2_h: Multi-step variance forecasts.
        params: Model parameters.
        data: Dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        Tuple of (calibrated_s2_1, calibrated_s2_h, mz_intercept, mz_slope).

    """
    from src.garch.garch_eval.metrics import apply_mz_calibration, mz_calibration_params

    mz_intercept = 0.0
    mz_slope = GARCH_EVAL_DEFAULT_SLOPE

    try:
        e_test, s2_test = load_test_resid_sigma2(
            params,
            data,
            model_name=model_name,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
        )
        if e_test.size > 0 and s2_test.size > 0:
            mz_params = mz_calibration_params(e_test, s2_test)
            mz_intercept = mz_params.get("intercept", 0.0)
            mz_slope = mz_params.get("slope", GARCH_EVAL_DEFAULT_SLOPE)
            logger.info(
                "Computed MZ calibration: intercept=%.6f, slope=%.3f",
                mz_intercept,
                mz_slope,
            )
            s2_1 = float(
                apply_mz_calibration(np.array([s2_1]), mz_intercept, mz_slope, use_intercept=True)[
                    0
                ],
            )
            s2_h = apply_mz_calibration(s2_h, mz_intercept, mz_slope, use_intercept=True)
            logger.info(
                "Applied MZ calibration (multiplicative + additive with floor) to forecasts"
            )
    except Exception as e:
        logger.warning(
            "Failed to compute/apply MZ calibration: %s. Using uncalibrated forecasts.",
            e,
        )

    return s2_1, s2_h, mz_intercept, mz_slope


# ============================================================================
# Statistical Utilities
# ============================================================================


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function; returns NaN if SciPy unavailable.

    Args:
    ----
        x: Test statistic value.
        df: Degrees of freedom.

    Returns:
    -------
        P-value (survival function value).

    """
    try:
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, df))
    except Exception:
        return float("nan")


# ============================================================================
# File Operations Utilities
# ============================================================================


def save_forecast_results(
    out: pd.DataFrame,
    use_mz_calibration: bool,
    mz_intercept: float,
    mz_slope: float,
) -> None:
    """Save forecast results to CSV file.

    Args:
    ----
        out: Forecast results DataFrame.
        use_mz_calibration: Whether MZ calibration was applied.
        mz_intercept: MZ intercept value.
        mz_slope: MZ slope value.

    """
    out["mz_calibrated"] = use_mz_calibration
    if use_mz_calibration:
        out["mz_intercept"] = mz_intercept
        out["mz_slope"] = mz_slope

    GARCH_FORECASTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GARCH_FORECASTS_FILE, index=False)
    logger.info("Saved GARCH forecasts to: %s", GARCH_FORECASTS_FILE)


def ensure_parent(path: Path) -> None:
    """Ensure parent directory exists for a given path.

    Args:
    ----
        path: File path.

    """
    path.parent.mkdir(parents=True, exist_ok=True)


def to_numpy(series_like: list[float] | np.ndarray) -> np.ndarray:
    """Convert series-like object to numpy array.

    Args:
    ----
        series_like: Input series or array.

    Returns:
    -------
        Numpy array.

    """
    return np.asarray(list(series_like), dtype=float)


# ============================================================================
# Parsing Utilities
# ============================================================================


def parse_alphas(alphas_str: str) -> list[float]:
    """Parse comma-separated alpha values.

    Args:
    ----
        alphas_str: Comma-separated string of alpha values.

    Returns:
    -------
        List of parsed alpha values.

    """
    return [float(a) for a in str(alphas_str).split(",") if a]


def load_best_model() -> tuple[dict[str, float], str, str, float | None, float | None]:
    """Load best model from estimation file.

    Returns
    -------
        Tuple of (params, name, dist, nu, lambda_skew).

    """
    with GARCH_ESTIMATION_FILE.open() as f:
        est = json.load(f)
    params, name, nu, lambda_skew = choose_best_from_estimation(est)
    dist = "skewt" if "skewt" in name else "normal"
    return params, name, dist, nu, lambda_skew


# ============================================================================
# Assembly Utilities
# ============================================================================


def assemble_forecast_results(
    s2_h: np.ndarray,
    s2_1: float,
    level: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> pd.DataFrame:
    """Assemble forecast results into DataFrame with PI and VaR.

    Args:
    ----
        s2_h: Multi-step variance forecasts.
        s2_1: One-step variance forecast.
        level: Prediction interval level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
    -------
        DataFrame with forecast results.

    """
    from src.garch.garch_eval.eval import prediction_interval, value_at_risk

    rows = []
    for h, s2 in enumerate(s2_h, start=1):
        lo, hi = prediction_interval(
            0.0, s2, level=level, dist=dist, nu=nu, lambda_skew=lambda_skew
        )
        var_l = value_at_risk(
            1.0 - level, mean=0.0, variance=s2, dist=dist, nu=nu, lambda_skew=lambda_skew
        )
        rows.append(
            {
                "h": int(h),
                "sigma2_forecast": float(s2),
                "sigma_forecast": float(np.sqrt(s2)),
                "pi_level": float(level),
                "pi_lower": float(lo),
                "pi_upper": float(hi),
                "var_left_alpha": float(1.0 - level),
                "VaR": float(var_l),
                "dist": dist,
                "nu": float(nu) if nu is not None else np.nan,
                "lambda": float(lambda_skew) if lambda_skew is not None else np.nan,
            },
        )
    out = pd.DataFrame(rows)
    # Sanity check: include one-step as first row consistency
    if out.shape[0] >= 1:
        out.loc[out.index[0], "sigma2_one_step_check"] = s2_1
    return out
