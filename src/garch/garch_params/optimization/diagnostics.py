"""Optimization diagnostics and composite objective for EGARCH models.

This module provides:
- Diagnostic penalties via ARCH-LM tests on standardized residuals
- AIC computation and normalization
- Composite objective combining QLIKE, AIC, and diagnostics
"""

from __future__ import annotations

import numpy as np

from src.constants import (
    GARCH_OPTIMIZATION_AIC_WEIGHT,
    GARCH_OPTIMIZATION_ARCH_LM_LAGS,
    GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD,
    GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT,
    GARCH_OPTIMIZATION_QLIKE_WEIGHT,
)
from src.garch.garch_params.core import egarch_variance
from src.garch.garch_params.models import EGARCHParams
from src.utils import get_logger

logger = get_logger(__name__)


def _compute_arch_lm_statistic(residuals_squared: np.ndarray, lags: int) -> tuple[float, float]:
    """Compute ARCH-LM test statistic and p-value.

    Args:
        residuals_squared: Squared standardized residuals.
        lags: Number of lags for ARCH-LM test.

    Returns:
        Tuple of (lm_statistic, p_value).
    """
    try:
        from scipy.stats import chi2  # type: ignore
    except ImportError:
        logger.warning("SciPy not available for ARCH-LM test, returning penalty")
        return float("inf"), 0.0

    n = len(residuals_squared)
    if n < lags + 1:
        return float("inf"), 0.0

    # Regress squared residuals on lagged squared residuals
    # y_t = c + b1*y_{t-1} + ... + bp*y_{t-p} + e_t
    y = residuals_squared[lags:]
    X = np.ones((n - lags, lags + 1))
    for i in range(lags):
        X[:, i + 1] = residuals_squared[lags - i - 1 : n - i - 1]

    # Compute R-squared from regression
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fitted = X @ beta
        ss_res = np.sum((y - y_fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # LM statistic = n * R^2 ~ chi2(lags)
        lm_stat = float((n - lags) * r_squared)
        p_value = float(1.0 - chi2.cdf(lm_stat, df=lags))

        return lm_stat, p_value
    except np.linalg.LinAlgError:
        return float("inf"), 0.0


def _standardize_residuals(
    residuals: np.ndarray,
    sigma2: np.ndarray,
) -> np.ndarray:
    """Standardize residuals by conditional standard deviation.

    Args:
        residuals: Raw residuals.
        sigma2: Conditional variance.

    Returns:
        Standardized residuals.
    """
    # Filter valid values
    mask = np.isfinite(residuals) & np.isfinite(sigma2) & (sigma2 > 0)
    if not np.any(mask):
        return np.array([])

    std_residuals = residuals[mask] / np.sqrt(sigma2[mask])
    return std_residuals


def _compute_egarch_sigma2(
    residuals: np.ndarray,
    params: dict[str, float],
    o: int,
    p: int,
    dist: str,
) -> np.ndarray | None:
    """Compute EGARCH conditional variance.

    Args:
        residuals: Raw residuals.
        params: EGARCH parameters.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Conditional variance array or None if computation fails.
    """
    egarch_params = EGARCHParams.from_dict(params, o=o, p=p, dist=dist)
    omega = egarch_params.omega
    alpha, gamma, beta = egarch_params.extract_for_variance()
    nu = egarch_params.nu
    lambda_skew = egarch_params.lambda_skew

    try:
        return egarch_variance(
            residuals,
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            dist=dist,
            nu=nu,
            lambda_skew=lambda_skew,
            init=None,
            o=o,
            p=p,
        )
    except Exception as ex:
        logger.debug("Failed to compute variance for diagnostics: %s", ex)
        return None


def _compute_penalty_from_pvalue(p_value: float) -> float:
    """Compute penalty based on ARCH-LM p-value.

    Args:
        p_value: P-value from ARCH-LM test.

    Returns:
        Penalty in [0, 1] range.
    """
    if p_value > GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD:
        return 0.0
    return 1.0 - (p_value / GARCH_OPTIMIZATION_DIAGNOSTIC_PVALUE_THRESHOLD)


def compute_diagnostic_penalty(
    residuals: np.ndarray,
    params: dict[str, float],
    o: int,
    p: int,
    dist: str,
) -> float:
    """Compute diagnostic penalty for model quality.

    Args:
        residuals: Raw residuals.
        params: Estimated EGARCH parameters.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Diagnostic penalty value (0 = perfect, higher = worse).
    """
    sigma2 = _compute_egarch_sigma2(residuals, params, o, p, dist)
    if sigma2 is None:
        return 1.0

    # Check if variance computation produced reasonable results
    valid_variance_mask = np.isfinite(sigma2) & (sigma2 > 0)
    if not np.any(valid_variance_mask):
        return 1.0

    # Check for extreme values that indicate numerical instability
    finite_sigma2 = sigma2[valid_variance_mask]
    if len(finite_sigma2) == 0 or np.any(finite_sigma2 > 1e10) or np.any(finite_sigma2 < 1e-10):
        return 1.0

    std_residuals = _standardize_residuals(residuals, sigma2)
    if len(std_residuals) < GARCH_OPTIMIZATION_ARCH_LM_LAGS + 1:
        return 1.0

    std_residuals_squared = std_residuals**2
    lm_stat, p_value = _compute_arch_lm_statistic(
        std_residuals_squared, GARCH_OPTIMIZATION_ARCH_LM_LAGS
    )

    penalty = _compute_penalty_from_pvalue(p_value)
    logger.debug("Diagnostic: ARCH-LM p-value=%.4f, penalty=%.4f", p_value, penalty)

    return float(penalty)


def compute_aic_penalty(n_obs: int, loglik: float, n_params: int) -> float:
    """Compute AIC (Akaike Information Criterion).

    Args:
        n_obs: Number of observations.
        loglik: Log-likelihood value.
        n_params: Number of model parameters.

    Returns:
        AIC value (lower is better).
    """
    # AIC = -2*log(L) + 2*k
    # But we have negative log-likelihood, so: AIC = 2*(-loglik) + 2*k
    aic = -2.0 * loglik + 2.0 * n_params
    return float(aic)


def normalize_aic_penalty(aic: float, n_obs: int) -> float:
    """Normalize AIC to [0, 1] range for composite objective.

    Args:
        aic: Raw AIC value.
        n_obs: Number of observations.

    Returns:
        Normalized AIC penalty in [0, 1].
    """
    # Normalize by number of observations to make comparable across datasets
    # Use sigmoid-like transformation to bound to [0, 1]
    normalized = aic / n_obs
    # Apply tanh for soft bounding
    penalty = float(np.tanh(normalized / 10.0))  # Scale factor of 10 for reasonable range
    return max(0.0, min(1.0, penalty))


# ==================== Composite Objective ====================


def _count_model_parameters(o: int, p: int, dist: str) -> int:
    """Count number of parameters in EGARCH(o,p) model.

    Args:
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Number of parameters.
    """
    # Base parameters: omega (1) + alpha (o) + gamma (o) + beta (p)
    n_params = 1 + o + o + p

    # Distribution parameters
    if dist == "student":
        n_params += 1  # nu
    elif dist == "skewt":
        n_params += 2  # nu + lambda

    return n_params


def _compute_weighted_sum(
    qlike_normalized: float, aic_normalized: float, diagnostic_penalty: float
) -> float:
    """Compute weighted sum of objective components."""
    return (
        GARCH_OPTIMIZATION_QLIKE_WEIGHT * qlike_normalized
        + GARCH_OPTIMIZATION_AIC_WEIGHT * aic_normalized
        + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT * diagnostic_penalty
    )


def _validate_weights() -> None:  # pragma: no cover - logging only
    total_weight = (
        GARCH_OPTIMIZATION_QLIKE_WEIGHT
        + GARCH_OPTIMIZATION_AIC_WEIGHT
        + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT
    )
    if not np.isclose(total_weight, 1.0):
        logger.warning("Composite objective weights do not sum to 1.0: %.4f", total_weight)


def compute_composite_objective(
    qlike: float,
    residuals: np.ndarray,
    params: dict[str, float],
    loglik: float,
    o: int,
    p: int,
    dist: str,
) -> tuple[float, dict[str, float]]:
    """Compute composite objective function value.

    Args:
        qlike: QLIKE loss value.
        residuals: Residuals for diagnostic tests.
        params: Estimated parameters.
        loglik: Log-likelihood value.
        o: ARCH order.
        p: GARCH order.
        dist: Distribution name.

    Returns:
        Tuple of (composite_objective, components_dict).
    """
    qlike_normalized = float(qlike)

    n_params = _count_model_parameters(o, p, dist)
    n_obs = len(residuals)
    aic_raw = compute_aic_penalty(n_obs, loglik, n_params)
    aic_normalized = normalize_aic_penalty(aic_raw, n_obs)

    diagnostic_penalty = compute_diagnostic_penalty(residuals, params, o, p, dist)
    composite = _compute_weighted_sum(qlike_normalized, aic_normalized, diagnostic_penalty)

    _validate_weights()

    components = {
        "qlike": qlike_normalized,
        "aic": aic_normalized,
        "aic_raw": aic_raw,
        "diagnostic": diagnostic_penalty,
        "composite": composite,
    }

    logger.debug(
        "Composite objective: QLIKE=%.4f, AIC=%.4f, Diagnostic=%.4f => Total=%.4f",
        qlike_normalized,
        aic_normalized,
        diagnostic_penalty,
        composite,
    )

    return float(composite), components


__all__ = [
    "compute_diagnostic_penalty",
    "compute_aic_penalty",
    "normalize_aic_penalty",
    # Composite objective
    "_count_model_parameters",
    "compute_composite_objective",
]
