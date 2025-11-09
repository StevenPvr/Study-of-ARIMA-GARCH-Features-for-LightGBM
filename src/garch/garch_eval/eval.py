"""GARCH evaluation: variance forecasts, VaR and prediction intervals.

Implements Step 5 (prévision et utilisation):
- One-step and multi-step conditional variance forecasts
- Asymmetric prediction intervals for returns
- Short-horizon Value-at-Risk based on the innovation distribution
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    GARCH_EVAL_DEFAULT_HORIZON,
    GARCH_EVAL_DEFAULT_LEVEL,
    GARCH_EVAL_DEFAULT_SLOPE,
)
from src.garch.garch_eval.utils import (
    apply_mz_calibration_to_forecasts,
    assemble_forecast_results,
    compute_initial_forecasts,
    compute_variance_path,
    load_and_prepare_residuals,
    load_model_params,
    quantile,
    save_forecast_results,
)
from src.utils import get_logger

logger = get_logger(__name__)

## EGARCH-only mode


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
    ----
        mean: Mean of the return distribution.
        variance: Conditional variance.
        level: Prediction interval level (default: 0.95).
        dist: Distribution type ('normal' or 'skewt').
        nu: Degrees of freedom for Student-t/Skew-t distribution.
        lambda_skew: Skewness parameter for Skew-t distribution.

    Returns:
    -------
        Tuple of (lower_bound, upper_bound).

    For level=0.95, returns (lo, hi) with tail probability 2.5% each.

    """
    if variance <= 0.0:
        msg = "variance must be positive"
        raise ValueError(msg)
    alpha = (1.0 - float(level)) / 2.0
    sigma = float(np.sqrt(variance))
    q_lo = quantile(dist, alpha, nu, lambda_skew)
    q_hi = quantile(dist, 1.0 - alpha, nu, lambda_skew)
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
    q = quantile(dist, float(alpha), nu, lambda_skew)
    return float(mean + sigma * q)


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
    from src.garch.garch_params.utils import egarch_kappa as eg_kappa

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
    alpha: float,  # noqa: ARG001
    gamma: float,  # noqa: ARG001
    beta: float,
    dist: str = "normal",  # noqa: ARG001
    nu: float | None = None,  # noqa: ARG001
) -> np.ndarray:
    """Multi-step variance path for EGARCH(1,1) under zero-mean shock expectations.

    With E(|z|-kappa)=0 and E(z)=0 for h>=2, the recursion reduces to
    ln(sigma2_{t+k}) = omega + beta * ln(sigma2_{t+k-1}).

    Args:
    ----
        horizon: Forecast horizon.
        s2_last: Last variance value.
        omega: Omega parameter.
        alpha: Alpha parameter (not used in multi-step under expectations).
        gamma: Gamma parameter (not used in multi-step under expectations).
        beta: Beta parameter.
        dist: Distribution type (not used in multi-step under expectations).
        nu: Degrees of freedom (not used in multi-step under expectations).

    Returns:
    -------
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
    ----
        horizon: Forecast horizon (default: 5).
        level: Prediction interval level (default: 0.95).
        use_mz_calibration: Whether to apply MZ calibration to forecasts (default: False).

    Returns:
    -------
        DataFrame with forecast results.

    """
    # Load model parameters
    params, model_name, dist, nu, gamma, lambda_skew = load_model_params()
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]

    # Load and prepare residuals (separate train and all)
    data, resid_train, _resid_all = load_and_prepare_residuals()

    # Compute variance path on TRAINING DATA ONLY (prevent data leakage)
    sigma2_path_train = compute_variance_path(
        resid_train,
        model_name,
        omega,
        alpha,
        beta,
        gamma,
        dist,
        nu,
        lambda_skew,
    )

    # Compute initial forecasts from training data only
    s2_1, s2_h = compute_initial_forecasts(
        resid_train,
        sigma2_path_train,
        horizon,
        omega,
        alpha,
        gamma,
        beta,
        dist,
        nu,
        lambda_skew,
    )

    # Apply MZ calibration if requested
    mz_intercept = 0.0
    mz_slope = GARCH_EVAL_DEFAULT_SLOPE
    if use_mz_calibration:
        s2_1, s2_h, mz_intercept, mz_slope = apply_mz_calibration_to_forecasts(
            s2_1,
            s2_h,
            params,
            data,
            model_name,
            dist,
            nu,
            lambda_skew,
        )

    # Assemble and save results
    out = assemble_forecast_results(s2_h, s2_1, level, dist, nu, lambda_skew)
    save_forecast_results(out, use_mz_calibration, mz_intercept, mz_slope)
    return out


__all__ = [
    "egarch_one_step_variance_forecast",
    "egarch_multi_step_variance_forecast",
    "prediction_interval",
    "value_at_risk",
    "forecast_from_artifacts",
]
