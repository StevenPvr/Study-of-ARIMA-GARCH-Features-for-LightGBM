"""GARCH evaluation: variance forecasts, VaR and prediction intervals.

Implements Step 5 (prévision et utilisation):
- One-step and multi-step conditional variance forecasts
- Asymmetric prediction intervals for returns
- Short-horizon Value-at-Risk based on the innovation distribution
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_DATASET_FILE,
    GARCH_ESTIMATION_FILE,
    GARCH_EVAL_AIC_MULTIPLIER,
    GARCH_EVAL_DEFAULT_HORIZON,
    GARCH_EVAL_DEFAULT_LEVEL,
    GARCH_EVAL_DEFAULT_SLOPE,
    GARCH_FORECASTS_FILE,
    GARCH_STUDENT_NU_MIN,
)
from src.garch.garch_eval.distributions import skewt_ppf
from src.garch.garch_params.estimation import egarch11_variance
from src.garch.structure_garch.detection import prepare_residuals
from src.utils import get_logger

logger = get_logger(__name__)

## EGARCH-only mode


def _quantile(dist: str, p: float, nu: float | None, lambda_skew: float | None = None) -> float:
    """Return left-tail quantile for Normal, Student-t, or Hansen skew-t."""
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


def prediction_interval(
    mean: float,
    variance: float,
    *,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> tuple[float, float]:
    """Two-sided prediction interval for returns under chosen innovation dist.

    Args:
        mean: Mean of the return distribution.
        variance: Conditional variance.
        level: Prediction interval level (default: 0.95).
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t distribution.
        lambda_skew: Skewness parameter for Skew-t distribution.

    Returns:
        Tuple of (lower_bound, upper_bound).

    For level=0.95, returns (lo, hi) with tail probability 2.5% each.
    """
    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)
    alpha = (1.0 - float(level)) / 2.0
    sigma = float(np.sqrt(variance))
    q_lo = _quantile(dist, alpha, nu, lambda_skew)
    q_hi = _quantile(dist, 1.0 - alpha, nu, lambda_skew)
    return float(mean + sigma * q_lo), float(mean + sigma * q_hi)


def value_at_risk(
    alpha: float,
    *,
    mean: float = 0.0,
    variance: float,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """Left-tail Value-at-Risk at level alpha (e.g., 0.01 or 0.05).

    Returns VaR_alpha such that P(R < VaR_alpha) = alpha.
    """
    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)
    sigma = float(np.sqrt(variance))
    q = _quantile(dist, float(alpha), nu, lambda_skew)
    return float(mean + sigma * q)


def _aic(ll: float, k: int) -> float:
    """Calculate AIC: 2k - 2*loglik."""
    return GARCH_EVAL_AIC_MULTIPLIER * k - GARCH_EVAL_AIC_MULTIPLIER * float(ll)


def _collect_converged_candidates(
    payload: dict,
    keys: list[str],
    k_params: dict[str, int],
) -> list[tuple[str, dict, float]]:
    """Collect converged model candidates with their AIC scores.

    Args:
        payload: Estimation payload dictionary.
        keys: List of model keys to check.
        k_params: Dictionary mapping model names to parameter counts.

    Returns:
        List of tuples (name, params_dict, aic_score).
    """
    cand: list[tuple[str, dict, float]] = []
    for name in keys:
        d = payload.get(name)
        if isinstance(d, dict) and d.get("converged"):
            k = k_params[name]
            cand.append((name, d, _aic(float(d["loglik"]), k)))
    return cand


def _choose_best_from_estimation(
    payload: dict,
) -> tuple[dict[str, float], str, float | None, float | None]:
    """Pick best model from estimation JSON using AIC and preference order.

    Preference order on ties: skew-t → student → normal.
    """
    keys = ["egarch_skewt", "egarch_student", "egarch_normal"]
    k_params = {"egarch_normal": 4, "egarch_student": 5, "egarch_skewt": 6}

    cand = _collect_converged_candidates(payload, keys, k_params)
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


def _load_model_params() -> tuple[
    dict[str, float], str, str, float | None, float | None, float | None
]:
    """Load and extract model parameters from estimation file.

    Returns:
        Tuple of (params_dict, model_name, dist, nu, gamma, lambda_skew).
    """
    with GARCH_ESTIMATION_FILE.open() as f:
        est = json.load(f)
    best, name, nu, lambda_skew = _choose_best_from_estimation(est)

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


def _compute_variance_path(
    resid_all: np.ndarray,
    model_name: str,
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
        Variance path array.

    Raises:
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


def _compute_egarch_forecasts(
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
        Tuple of (one-step forecast, multi-step forecasts array).
    """
    from src.garch.garch_params.estimation import _egarch_kappa as eg_kappa

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


def egarch_one_step_variance_forecast(
    e_last: float,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
) -> float:
    """One-step variance forecast for EGARCH(1,1).

    Uses ln(sigma2_{t+1}) = omega + beta*ln(sigma2_t) + alpha*(|z_t|-kappa) + gamma*z_t,
    where z_t = e_t / sigma_t and kappa depends on the distribution.
    """
    from src.garch.garch_params.estimation import _egarch_kappa as eg_kappa

    z_last = float(e_last / np.sqrt(s2_last))
    kappa = eg_kappa(dist, nu, lambda_skew)
    ln_next = (
        float(omega)
        + float(beta) * float(np.log(s2_last))
        + float(alpha) * (abs(z_last) - float(kappa))
        + float(gamma) * z_last
    )
    return float(np.exp(ln_next))


def egarch_multi_step_variance_forecast(
    horizon: int,
    s2_last: float,
    *,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    dist: str = "normal",
    nu: float | None = None,
) -> np.ndarray:
    """Multi-step variance path for EGARCH(1,1) under zero-mean shock expectations.

    With E(|z|-kappa)=0 and E(z)=0 for h>=2, the recursion reduces to
    ln(sigma2_{t+k}) = omega + beta * ln(sigma2_{t+k-1}).

    Args:
        horizon: Forecast horizon.
        s2_last: Last variance value.
        omega: Omega parameter.
        alpha: Alpha parameter (not used in multi-step under expectations).
        gamma: Gamma parameter (not used in multi-step under expectations).
        beta: Beta parameter.
        dist: Distribution type (not used in multi-step under expectations).
        nu: Degrees of freedom (not used in multi-step under expectations).

    Returns:
        Array of variance forecasts.
    """
    # Parameters alpha, gamma, dist, nu are not needed beyond expectations
    h = int(max(0, horizon))
    out = np.empty(h, dtype=float)
    log_s2 = float(np.log(s2_last))
    for i in range(h):
        log_s2 = float(omega) + float(beta) * log_s2
        out[i] = float(np.exp(log_s2))
    return out


def _assemble_forecast_results(
    s2_h: np.ndarray,
    s2_1: float,
    level: float,
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> pd.DataFrame:
    """Assemble forecast results into DataFrame with PI and VaR.

    Args:
        s2_h: Multi-step variance forecasts.
        s2_1: One-step variance forecast.
        level: Prediction interval level.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
        DataFrame with forecast results.
    """
    rows = []
    for h, s2 in enumerate(s2_h, start=1):
        lo, hi = prediction_interval(0.0, s2, level=level, dist=dist, nu=nu, lambda_skew=lambda_skew)
        var_l = value_at_risk(1.0 - level, mean=0.0, variance=s2, dist=dist, nu=nu, lambda_skew=lambda_skew)
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
            }
        )
    out = pd.DataFrame(rows)
    # Sanity check: include one-step as first row consistency
    if out.shape[0] >= 1:
        out.loc[out.index[0], "sigma2_one_step_check"] = s2_1
    return out


def _load_and_prepare_residuals() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load dataset and prepare filtered residuals for train and all data.

    Returns:
        Tuple of (dataframe, train_residuals, all_residuals).
        train_residuals: Only training residuals (for forecast initialization).
        all_residuals: All residuals (for variance path computation if needed).
    """
    data = pd.read_csv(GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]

    # Get train residuals for forecast initialization (no data leakage)
    df_train = data[data["split"] == "train"].copy()
    resid_train = prepare_residuals(df_train, use_test_only=False)
    resid_train = resid_train[np.isfinite(resid_train)]

    # Get all residuals (for potential future use, but not for forecast init)
    resid_all = prepare_residuals(data, use_test_only=False)
    resid_all = resid_all[np.isfinite(resid_all)]

    return data, resid_train, resid_all


def _compute_initial_forecasts(
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
        Tuple of (one_step_forecast, multi_step_forecasts).
    """
    if resid_train.size == 0 or sigma2_path_train.size == 0:
        msg = "Training residuals or variance path is empty"
        raise ValueError(msg)

    e_last = float(resid_train[-1])
    s2_last = float(sigma2_path_train[-1])
    s2_1, s2_h = _compute_egarch_forecasts(
        e_last, s2_last, horizon, omega, alpha, float(gamma or 0.0), beta, dist, nu, lambda_skew
    )
    return s2_1, s2_h


def _save_forecast_results(
    out: pd.DataFrame,
    use_mz_calibration: bool,
    mz_intercept: float,
    mz_slope: float,
) -> None:
    """Save forecast results to CSV file.

    Args:
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


def _apply_mz_calibration_to_forecasts(
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

    Args:
        s2_1: One-step variance forecast.
        s2_h: Multi-step variance forecasts.
        params: Model parameters.
        data: Dataset DataFrame.
        model_name: Model name.
        dist: Distribution type.
        nu: Degrees of freedom.
        lambda_skew: Skewness parameter (for Skew-t).

    Returns:
        Tuple of (calibrated_s2_1, calibrated_s2_h, mz_intercept, mz_slope).
    """
    from src.garch.garch_eval.metrics import (
        _load_test_resid_sigma2,
        apply_mz_calibration,
        mz_calibration_params,
    )

    mz_intercept = 0.0
    mz_slope = GARCH_EVAL_DEFAULT_SLOPE

    try:
        e_test, s2_test = _load_test_resid_sigma2(
            params, data, model_name=model_name, dist=dist, nu=nu, lambda_skew=lambda_skew
        )
        if e_test.size > 0 and s2_test.size > 0:
            mz_params = mz_calibration_params(e_test, s2_test)
            mz_intercept = mz_params.get("intercept", 0.0)
            mz_slope = mz_params.get("slope", GARCH_EVAL_DEFAULT_SLOPE)
            logger.info(
                f"Computed MZ calibration: intercept={mz_intercept:.6f}, slope={mz_slope:.3f}"
            )
            s2_1 = float(
                apply_mz_calibration(np.array([s2_1]), mz_intercept, mz_slope, use_intercept=True)[
                    0
                ]
            )
            s2_h = apply_mz_calibration(s2_h, mz_intercept, mz_slope, use_intercept=True)
            logger.info("Applied MZ calibration (multiplicative + additive with floor) to forecasts")
    except Exception as e:
        logger.warning(
            f"Failed to compute/apply MZ calibration: {e}. Using uncalibrated forecasts."
        )

    return s2_1, s2_h, mz_intercept, mz_slope


def forecast_from_artifacts(
    *,
    horizon: int = GARCH_EVAL_DEFAULT_HORIZON,
    level: float = GARCH_EVAL_DEFAULT_LEVEL,
    use_mz_calibration: bool = False,
) -> pd.DataFrame:
    """Build forecasts from saved estimation outputs and dataset.

    Steps:
    - Load best GARCH params from estimation JSON (normal vs student)
    - Recompute sigma^2 path on full residual series
    - Compute MZ calibration parameters from test data
    - Produce one-step and multi-step variance forecasts up to horizon
    - Apply MZ calibration to forecasts if requested (default: off)
    - Compute VaR_alpha (left tail) and two-sided prediction intervals
    - Save CSV to `GARCH_FORECASTS_FILE`

    Args:
        horizon: Forecast horizon (default: 5).
        level: Prediction interval level (default: 0.95).
        use_mz_calibration: Whether to apply MZ calibration to forecasts (default: False).

    Returns:
        DataFrame with forecast results.
    """
    # Load model parameters
    params, model_name, dist, nu, gamma, lambda_skew = _load_model_params()
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Load and prepare residuals (separate train and all)
    data, resid_train, resid_all = _load_and_prepare_residuals()

    # Compute variance path on TRAINING DATA ONLY (prevent data leakage)
    sigma2_path_train = _compute_variance_path(
        resid_train, model_name, omega, alpha, beta, gamma, dist, nu, lambda_skew
    )

    # Compute initial forecasts from training data only
    s2_1, s2_h = _compute_initial_forecasts(
        resid_train, sigma2_path_train, horizon, omega, alpha, gamma, beta, dist, nu, lambda_skew
    )

    # Apply MZ calibration if requested
    mz_intercept = 0.0
    mz_slope = GARCH_EVAL_DEFAULT_SLOPE
    if use_mz_calibration:
        s2_1, s2_h, mz_intercept, mz_slope = _apply_mz_calibration_to_forecasts(
            s2_1, s2_h, params, data, model_name, dist, nu, lambda_skew
        )

    # Assemble and save results
    out = _assemble_forecast_results(s2_h, s2_1, level, dist, nu, lambda_skew)
    _save_forecast_results(out, use_mz_calibration, mz_intercept, mz_slope)
    return out


__all__ = [
    "egarch_one_step_variance_forecast",
    "egarch_multi_step_variance_forecast",
    "prediction_interval",
    "value_at_risk",
    "forecast_from_artifacts",
]
