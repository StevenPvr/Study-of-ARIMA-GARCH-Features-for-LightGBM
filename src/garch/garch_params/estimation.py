"""EGARCH(1,1) parameter estimation via conditional MLE.

Implements conditional maximum likelihood estimation:
- Assumes parametric distribution for innovations zt (Normal, Student-t, Skew-t)
- Maximizes conditional log-likelihood by recursing conditional variance σt²
- Numerical optimization performed by software libraries

Intended for ARIMA residuals (mean ~ 0).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.constants import (
    GARCH_MIN_INIT_VAR,
    GARCH_SKEWT_LAMBDA_INIT,
    GARCH_SKEWT_LAMBDA_MAX,
    GARCH_SKEWT_LAMBDA_MIN,
    GARCH_STUDENT_NU_INIT,
    GARCH_STUDENT_NU_MAX,
    GARCH_STUDENT_NU_MIN,
)
from src.utils import get_logger

logger = get_logger(__name__)


def _validate_series(e: np.ndarray) -> np.ndarray:
    """Return residual series as 1D float array and validate length.

    Raises ValueError if fewer than 10 observations are provided.
    """
    arr = np.asarray(e, dtype=float).ravel()
    if arr.size < 10:
        msg = "Need at least 10 observations to estimate EGARCH(1,1)."
        raise ValueError(msg)
    return arr


# ---------------------- EGARCH(1,1) ----------------------


def _egarch_kappa(dist: str, nu: float | None, lambda_skew: float | None = None) -> float:
    """Return E[|Z|] for standardized innovations used in EGARCH.

    For Normal: sqrt(2/pi)
    For Student-t (variance=1 standardized): sqrt(nu-2) * Gamma((nu-1)/2) / (sqrt(pi) * Gamma(nu/2))
    For Skew-t (Hansen): computed numerically or approximated
    """
    dist_l = dist.lower()
    if dist_l == "skewt" and nu is not None and lambda_skew is not None and nu > 2.0:
        try:
            from scipy.special import gammaln  # type: ignore

            # Skew-t kappa: E[|Z|] for standardized Skew-t
            # Approximation using numerical integration or closed form when available
            # For now, use a reasonable approximation based on nu and lambda
            lambda_val = float(lambda_skew)
            nu_val = float(nu)
            
            # Base Student-t kappa
            ln_num = 0.5 * np.log(max(nu_val - 2.0, 1e-12)) + gammaln(0.5 * (nu_val - 1.0))
            ln_den = 0.5 * np.log(np.pi) + gammaln(0.5 * nu_val)
            kappa_t = float(np.exp(ln_num - ln_den))
            
            # Adjust for skewness (lambda affects asymmetry)
            # When lambda < 0 (left skew), E[|Z|] tends to be slightly higher
            # Simple approximation: kappa_skewt ≈ kappa_t * (1 + 0.1 * |lambda|)
            kappa_adj = kappa_t * (1.0 + 0.1 * abs(lambda_val))
            return float(kappa_adj)
        except Exception as ex:
            logger.debug("Falling back to Student-t kappa; scipy unavailable or failed: %s", ex)
            # Fall through to Student-t
    if dist_l == "student" and nu is not None and nu > 2.0:
        try:
            from scipy.special import gammaln  # type: ignore

            ln_num = 0.5 * np.log(max(nu - 2.0, 1e-12)) + gammaln(0.5 * (nu - 1.0))
            ln_den = 0.5 * np.log(np.pi) + gammaln(0.5 * nu)
            return float(np.exp(ln_num - ln_den))
        except Exception as ex:
            logger.debug("Falling back to Normal kappa; scipy unavailable or failed: %s", ex)
    # Default to Normal constant
    return float(np.sqrt(2.0 / np.pi))


def _initialize_variance(ee: np.ndarray, init: float | None) -> float:
    """Initialize variance for EGARCH recursion."""
    if init is not None and init > 0:
        return float(init)
    v = float(np.var(ee))
    return max(v, GARCH_MIN_INIT_VAR)


def _compute_variance_step(
    ee: np.ndarray,
    s2_prev: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    kappa: float,
) -> float:
    """Compute next variance step in EGARCH recursion."""
    z_prev = float(ee / np.sqrt(max(s2_prev, GARCH_MIN_INIT_VAR)))
    ln_next = (
        omega
        + beta * np.log(max(s2_prev, GARCH_MIN_INIT_VAR))
        + alpha * (abs(z_prev) - kappa)
        + gamma * z_prev
    )
    # Clip ln_next to prevent overflow in exp (ln(700) ≈ 6.55 is safe)
    ln_next_clipped = min(ln_next, 700.0)
    return float(np.exp(ln_next_clipped))


def egarch11_variance(
    e: np.ndarray,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    *,
    dist: str = "normal",
    nu: float | None = None,
    lambda_skew: float | None = None,
    init: float | None = None,
) -> np.ndarray:
    """Compute EGARCH(1,1) variance path.

    log(sigma_t^2) = omega + beta * log(sigma_{t-1}^2)
    + alpha * (|z_{t-1}| - kappa) + gamma * z_{t-1}
    where z_{t-1} = e_{t-1}/sigma_{t-1} and kappa = E|Z|
    under the chosen innovation distribution.
    """
    ee = np.asarray(e, dtype=float).ravel()
    n = ee.size
    s2 = np.empty(n, dtype=float)
    s2[0] = _initialize_variance(ee, init)
    kappa = _egarch_kappa(dist, nu, lambda_skew)
    for t in range(1, n):
        s2[t] = _compute_variance_step(
            ee[t - 1], s2[t - 1], omega, alpha, gamma, beta, kappa
        )
        if not np.isfinite(s2[t]) or s2[t] <= 0:
            return np.full(n, np.nan)
    return s2


def _negloglik_egarch_normal(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Normal innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta = params
    # |beta|<1 for stationarity in log variance; keep broad bounds for others
    if not (-0.999 < beta < 0.999):
        return 1e50
    s2 = egarch11_variance(e, omega, alpha, gamma, beta, dist="normal", nu=None)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    # Suppress overflow warnings but keep exact calculations
    # If real overflow occurs (inf values), we'll detect it below
    with np.errstate(divide="ignore", over="ignore"):
        z2 = (e**2) / s2
    # Only reject if actual overflow occurred (infinite values)
    if not np.all(np.isfinite(z2)):
        return 1e50
    ll = -0.5 * (np.log(2.0 * np.pi) + np.log(s2) + z2).sum()
    if not np.isfinite(ll):
        return 1e50
    return -float(ll)


def _validate_student_params(beta: float, nu: float) -> bool:
    """Validate Student-t distribution parameters."""
    return (-0.999 < beta < 0.999) and nu > 2.0


def _validate_skewt_params(beta: float, nu: float, lambda_skew: float) -> bool:
    """Validate Skew-t distribution parameters."""
    return (
        (-0.999 < beta < 0.999)
        and nu > 2.0
        and GARCH_SKEWT_LAMBDA_MIN < lambda_skew < GARCH_SKEWT_LAMBDA_MAX
    )


def _compute_student_loglikelihood(
    e: np.ndarray, s2: np.ndarray, nu: float
) -> float:
    """Compute Student-t log-likelihood given variance and degrees of freedom."""
    from scipy.special import gammaln  # type: ignore

    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (
        np.log(np.pi) + np.log(nu - 2.0)
    )
    # Suppress overflow warnings but keep exact calculations
    # If real overflow occurs (inf values), we'll detect it below
    with np.errstate(divide="ignore", over="ignore"):
        z2_scaled = (e**2) / (s2 * (nu - 2.0))
    # Only reject if actual overflow occurred (infinite values)
    if not np.all(np.isfinite(z2_scaled)):
        raise ValueError("Overflow in z2_scaled computation")
    ll_terms = c_log - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(z2_scaled)
    return np.sum(ll_terms)


def _compute_skewt_loglikelihood(
    e: np.ndarray, s2: np.ndarray, nu: float, lambda_skew: float
) -> float:
    """Compute Skew-t (Hansen) log-likelihood given variance, nu, and lambda.

    Skew-t standardized (variance=1) density:
    f(z) = bc * (1 + 1/(nu-2) * ((b*z+a)/(1-lambda))^2)^(-(nu+1)/2)  if z < -a/b
    f(z) = bc * (1 + 1/(nu-2) * ((b*z+a)/(1+lambda))^2)^(-(nu+1)/2)  if z >= -a/b

    where:
    - c = Gamma((nu+1)/2) / (sqrt(pi*(nu-2)) * Gamma(nu/2))
    - a = 4*lambda*c*(nu-2)/(nu-1)
    - b = sqrt(1 + 3*lambda^2 - a^2)
    """
    from scipy.special import gammaln  # type: ignore

    z = e / np.sqrt(s2)
    lambda_val = float(lambda_skew)
    nu_val = float(nu)

    # Compute constants
    c_log = gammaln(0.5 * (nu_val + 1.0)) - gammaln(0.5 * nu_val) - 0.5 * (
        np.log(np.pi) + np.log(nu_val - 2.0)
    )
    c = np.exp(c_log)
    a = 4.0 * lambda_val * c * (nu_val - 2.0) / (nu_val - 1.0)
    b_sq = 1.0 + 3.0 * lambda_val**2 - a**2
    if b_sq <= 0:
        raise ValueError("Invalid Skew-t parameters: b^2 <= 0")
    b = np.sqrt(b_sq)
    threshold = -a / b

    # Compute log-likelihood terms
    ll_terms = np.empty_like(z)
    mask_left = z < threshold
    mask_right = ~mask_left

    # Left tail: z < -a/b
    if np.any(mask_left):
        z_left = z[mask_left]
        denom = 1.0 - lambda_val
        z_scaled = (b * z_left + a) / denom
        z2_scaled = z_scaled**2 / (nu_val - 2.0)
        ll_terms[mask_left] = (
            c_log + np.log(b)
            - 0.5 * np.log(s2[mask_left])
            - 0.5 * (nu_val + 1.0) * np.log1p(z2_scaled)
        )

    # Right tail: z >= -a/b
    if np.any(mask_right):
        z_right = z[mask_right]
        denom = 1.0 + lambda_val
        z_scaled = (b * z_right + a) / denom
        z2_scaled = z_scaled**2 / (nu_val - 2.0)
        ll_terms[mask_right] = (
            c_log + np.log(b)
            - 0.5 * np.log(s2[mask_right])
            - 0.5 * (nu_val + 1.0) * np.log1p(z2_scaled)
        )

    return float(np.sum(ll_terms))


def _negloglik_egarch_student(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Student-t innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta, nu = params
    if not _validate_student_params(beta, nu):
        return 1e50
    s2 = egarch11_variance(e, omega, alpha, gamma, beta, dist="student", nu=nu)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    try:
        ll = _compute_student_loglikelihood(e, s2, nu)
        return -float(ll)
    except Exception:
        return 1e50


def _negloglik_egarch_skewt(params: np.ndarray, e: np.ndarray) -> float:
    """Compute negative log-likelihood for EGARCH(1,1) with Skew-t innovations.

    Returns large penalty (1e50) for invalid parameters or numerical issues.
    """
    omega, alpha, gamma, beta, nu, lambda_skew = params
    if not _validate_skewt_params(beta, nu, lambda_skew):
        return 1e50
    s2 = egarch11_variance(
        e, omega, alpha, gamma, beta, dist="skewt", nu=nu, lambda_skew=lambda_skew
    )
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e50
    try:
        ll = _compute_skewt_loglikelihood(e, s2, nu, lambda_skew)
        return -float(ll)
    except Exception:
        return 1e50


def _minimize_slsqp(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    constraints: Sequence[dict] | None = None,
) -> Any:
    """Run SciPy SLSQP minimize with local import to keep optional dep isolated."""
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        msg = "SciPy required for MLE estimation"
        raise RuntimeError(msg) from exc
    return minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)


def _optimize_with_fallback(
    fun: Callable[[np.ndarray], float],
    primary_x0: np.ndarray,
    fallback_x0: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    constraints: Sequence[dict] | None = None,
) -> Any:
    """Try optimization with primary x0, fallback to alternative on failure."""
    try:
        return _minimize_slsqp(fun, primary_x0, bounds, constraints)
    except Exception as exc:
        logger.info("Warm-start failed, retrying with defaults: %s", exc)
        return _minimize_slsqp(fun, fallback_x0, bounds, constraints)


def _egarch_setup(e: np.ndarray, dist: str, x0: Iterable[float] | None, v: float) -> tuple[
    np.ndarray,
    Sequence[tuple[float | None, float | None]],
    Callable[[np.ndarray], float],
    np.ndarray,
]:
    """Build init vector, bounds, objective and fallback x0 for EGARCH(1,1)."""
    b0, a0, g0 = 0.95, 0.1, 0.0
    w0 = (1.0 - b0) * np.log(max(v, GARCH_MIN_INIT_VAR))
    if dist.lower() == "normal":
        x0_arr = (
            np.array(list(x0), dtype=float)
            if x0 is not None
            else np.array([w0, a0, g0, b0], dtype=float)
        )
        bounds = [(-50.0, 50.0), (-5.0, 5.0), (-5.0, 5.0), (-0.999, 0.999)]

        def fun(p):
            return _negloglik_egarch_normal(p, e)

        fallback_x0 = np.array([w0, a0, g0, b0], dtype=float)
    elif dist.lower() == "student":
        x0_arr = (
            np.array(list(x0), dtype=float)
            if x0 is not None
            else np.array([w0, a0, g0, b0, GARCH_STUDENT_NU_INIT], dtype=float)
        )
        bounds = [
            (-50.0, 50.0),
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-0.999, 0.999),
            (GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX),
        ]

        def fun(p):
            return _negloglik_egarch_student(p, e)

        fallback_x0 = np.array([w0, a0, g0, b0, GARCH_STUDENT_NU_INIT], dtype=float)
    elif dist.lower() == "skewt":
        x0_arr = (
            np.array(list(x0), dtype=float)
            if x0 is not None
            else np.array(
                [w0, a0, g0, b0, GARCH_STUDENT_NU_INIT, GARCH_SKEWT_LAMBDA_INIT],
                dtype=float,
            )
        )
        bounds = [
            (-50.0, 50.0),
            (-5.0, 5.0),
            (-5.0, 5.0),
            (-0.999, 0.999),
            (GARCH_STUDENT_NU_MIN, GARCH_STUDENT_NU_MAX),
            (GARCH_SKEWT_LAMBDA_MIN, GARCH_SKEWT_LAMBDA_MAX),
        ]

        def fun(p):
            return _negloglik_egarch_skewt(p, e)

        fallback_x0 = np.array(
            [w0, a0, g0, b0, GARCH_STUDENT_NU_INIT, GARCH_SKEWT_LAMBDA_INIT], dtype=float
        )
    else:
        msg = "dist must be 'normal', 'student', or 'skewt'."
        raise ValueError(msg)
    return x0_arr, bounds, fun, fallback_x0


def _egarch_finalize(dist: str, res: Any) -> dict[str, float]:
    out: dict[str, float] = {
        "omega": float(res.x[0]),
        "alpha": float(res.x[1]),
        "gamma": float(res.x[2]),
        "beta": float(res.x[3]),
        "loglik": float(-res.fun),
        "converged": bool(res.success),
    }
    dist_l = dist.lower()
    if dist_l == "student":
        out["nu"] = float(res.x[4])
    elif dist_l == "skewt":
        out["nu"] = float(res.x[4])
        out["lambda"] = float(res.x[5])
    return out


def estimate_egarch_mle(
    residuals: np.ndarray,
    *,
    dist: str = "normal",
    x0: Iterable[float] | None = None,
) -> dict[str, float]:
    """Estimate EGARCH(1,1) parameters via conditional maximum likelihood.

    Assumes parametric distribution for innovations zt (Normal, Student-t, or Skew-t)
    and maximizes the conditional log-likelihood by recursing the conditional
    variance σt². Numerical optimization is performed using SciPy.

    Args:
        residuals: Residual series εt from mean model (SARIMA).
        dist: Distribution for innovations: 'normal', 'student', or 'skewt'.
        x0: Optional initial parameter vector.

    Returns:
        Dictionary with estimated parameters (omega, alpha, gamma, beta, nu, lambda)
        and optimization results (loglik, converged).
    """
    e = _validate_series(residuals)
    v = float(np.var(e))
    x0_arr, bounds, fun, fallback_x0 = _egarch_setup(e, dist, x0, v)
    logger.info("Starting EGARCH(1,1) MLE: dist=%s", dist)
    res = _optimize_with_fallback(fun, x0_arr, fallback_x0, bounds)
    out = _egarch_finalize(dist, res)
    extra_params = ""
    if "nu" in out:
        extra_params = f", nu={out['nu']:.2f}"
    if "lambda" in out:
        extra_params += f", lambda={out['lambda']:.4f}"
    logger.info(
        "Finished EGARCH MLE (success=%s): omega=%.6g, alpha=%.4f, gamma=%.4f, beta=%.4f%s",
        out["converged"],
        out["omega"],
        out["alpha"],
        out["gamma"],
        out["beta"],
        extra_params,
    )
    return out


__all__ = [
    "egarch11_variance",
    "estimate_egarch_mle",
    "_egarch_kappa",
]
