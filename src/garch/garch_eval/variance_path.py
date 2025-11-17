"""Variance path computation utilities for GARCH evaluation."""

from __future__ import annotations

import numpy as np

from src.garch.garch_params.core import egarch11_variance


def _validate_input_residuals(resid_all: np.ndarray) -> None:
    """Validate input residuals are non-empty and have finite values."""
    if resid_all.size == 0:
        msg = "Cannot compute variance path: residual array is empty"
        raise ValueError(msg)

    if not np.any(np.isfinite(resid_all)):
        msg = "Cannot compute variance path: no finite values in residual array"
        raise ValueError(msg)


def _validate_variance_finiteness(
    sigma2_path: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float,
    resid_all: np.ndarray,
) -> None:
    """Validate that computed variance path has finite values."""
    finite_mask = np.isfinite(sigma2_path)
    n_invalid = np.sum(~finite_mask)

    if not np.all(finite_mask):
        if n_invalid == sigma2_path.size:
            msg = (
                f"Invalid sigma^2 path: all {sigma2_path.size} values are non-finite (NaN). "
                f"This may indicate invalid parameters (omega={omega:.6f}, alpha={alpha:.6f}, "
                f"beta={beta:.6f}, gamma={gamma:.6f}) or invalid residuals "
                f"(size={resid_all.size}, finite={np.sum(np.isfinite(resid_all))})."
            )
        else:
            finite_values = sigma2_path[finite_mask]
            msg = (
                f"Invalid sigma^2 path: {n_invalid}/{sigma2_path.size} values are non-finite. "
                f"Finite values: min={np.min(finite_values):.6e}, max={np.max(finite_values):.6e}"
            )
        raise ValueError(msg)


def _validate_variance_positivity(sigma2_path: np.ndarray) -> None:
    """Validate that variance path values are positive."""
    positive_mask = sigma2_path > 0
    if not np.all(positive_mask):
        n_non_positive = np.sum(~positive_mask)
        min_val = np.min(sigma2_path[positive_mask]) if np.any(positive_mask) else 0.0
        msg = (
            f"Invalid sigma^2 path: {n_non_positive}/{sigma2_path.size} values are non-positive "
            f"(min={np.min(sigma2_path):.6e}, min_positive={min_val:.6e})"
        )
        raise ValueError(msg)


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
    _validate_input_residuals(resid_all)

    gamma_val = float(gamma or 0.0)
    sigma2_path = egarch11_variance(
        resid_all,
        omega,
        alpha,
        gamma_val,
        beta,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
    )

    if sigma2_path.size == 0:
        msg = "Invalid sigma^2 path: computed path is empty"
        raise ValueError(msg)

    _validate_variance_finiteness(sigma2_path, omega, alpha, beta, gamma_val, resid_all)
    _validate_variance_positivity(sigma2_path)

    return sigma2_path


def _extract_params_from_dict(params: dict[str, float]) -> tuple[float, float, float, float]:
    """Extract and validate GARCH parameters from dictionary."""
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha", np.nan))
    beta = float(params.get("beta", np.nan))
    gamma_val = params.get("gamma")
    gamma = float(gamma_val) if gamma_val is not None else 0.0
    return omega, alpha, beta, gamma


def _determine_garch_orders(params: dict[str, float]) -> tuple[int, int]:
    """Determine GARCH orders (o, p) from parameter dictionary."""
    # Determine p (beta order)
    if "beta3" in params:
        p = 3
    elif "beta2" in params:
        p = 2
    elif "beta1" in params:
        p = 1
    else:
        # Fallback to legacy 'beta' parameter
        beta_fb = float(params.get("beta", np.nan))
        if np.isfinite(beta_fb):
            p = 1
        else:
            raise ValueError("Missing GARCH beta parameters (beta/beta1/beta2/beta3)")

    # Determine o (alpha/gamma order)
    o = 2 if ("alpha2" in params or "gamma2" in params) else 1

    return o, p


def _build_alpha_params(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Build alpha parameters based on GARCH order o."""
    if o == 1:
        alpha = float(params.get("alpha", np.nan))
        if not np.isfinite(alpha):
            raise ValueError("Missing alpha for o=1")
        return alpha
    else:  # o == 2
        try:
            return (float(params["alpha1"]), float(params["alpha2"]))
        except KeyError as exc:
            raise ValueError("Missing alpha1/alpha2 for o=2") from exc


def _build_gamma_params(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Build gamma parameters based on GARCH order o."""
    if o == 1:
        return float(params.get("gamma", 0.0) or 0.0)
    else:  # o == 2
        try:
            return (float(params["gamma1"]), float(params["gamma2"]))
        except KeyError as exc:
            raise ValueError("Missing gamma1/gamma2 for o=2") from exc


def _build_beta_params(
    params: dict[str, float],
    p: int,
    beta_fallback: float,
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Build beta parameters based on GARCH order p."""
    if p == 1:
        beta = float(params.get("beta1", beta_fallback))
        if not np.isfinite(beta):
            raise ValueError("Missing beta for p=1")
        return beta
    elif p == 2:
        try:
            return (float(params["beta1"]), float(params["beta2"]))
        except KeyError as exc:
            raise ValueError("Missing beta1/beta2 for p=2") from exc
    else:  # p == 3
        try:
            return (float(params["beta1"]), float(params["beta2"]), float(params["beta3"]))
        except KeyError as exc:
            raise ValueError("Missing beta1/beta2/beta3 for p=3") from exc


def _validate_computed_variance(s2_f: np.ndarray) -> None:
    """Validate computed variance has finite and positive values."""
    if not np.all(np.isfinite(s2_f)):
        msg = (
            f"Variance computation failed: non-finite values detected. "
            f"Size={s2_f.size}, finite={np.sum(np.isfinite(s2_f))}"
        )
        raise ValueError(msg)

    if not np.all(s2_f > 0):
        msg = (
            f"Variance computation failed: non-positive values detected. "
            f"Min={np.min(s2_f):.6e}, positive={np.sum(s2_f > 0)}/{s2_f.size}"
        )
        raise ValueError(msg)


def compute_variance_path_for_test(
    resid_f: np.ndarray,
    model_name: str,  # noqa: ARG001
    params: dict[str, float],
    dist: str,
    nu: float | None,
    lambda_skew: float | None = None,
) -> np.ndarray:
    """Compute variance path for filtered residuals using EGARCH(o,p).

    Infers (o,p) from provided params without silent defaults and raises
    explicit errors if required keys are missing.
    """
    # Extract basic parameters
    omega, _, beta_fb, _ = _extract_params_from_dict(params)

    # Determine GARCH orders
    o, p = _determine_garch_orders(params)

    # Build parameter tuples based on orders
    alpha_param = _build_alpha_params(params, o)
    gamma_param = _build_gamma_params(params, o)
    beta_param = _build_beta_params(params, p, beta_fb)

    # Compute variance path using general EGARCH(o,p)
    from src.garch.garch_params.core import egarch_variance

    s2_f = egarch_variance(
        resid_f,
        float(omega),
        alpha_param,
        gamma_param,
        beta_param,
        dist=dist,
        nu=nu,
        lambda_skew=lambda_skew,
        init=None,
        o=o,
        p=p,
    )
    _validate_computed_variance(s2_f)
    return s2_f


def _compute_one_step_forecast(
    e_last: float,
    s2_last: float,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    kappa: float,
) -> float:
    """Compute one-step EGARCH variance forecast."""
    z_last = float(e_last / np.sqrt(s2_last))
    ln_next = omega + beta * np.log(s2_last) + alpha * (abs(z_last) - kappa) + gamma * z_last
    return float(np.exp(ln_next))


def _compute_multi_step_forecasts(
    s2_last: float,
    horizon: int,
    omega: float,
    beta: float,
) -> np.ndarray:
    """Compute multi-step EGARCH variance forecasts."""
    s2_h = np.empty(horizon, dtype=float)
    log_s2 = float(np.log(s2_last))
    for i in range(horizon):
        log_s2 = omega + beta * log_s2
        s2_h[i] = float(np.exp(log_s2))
    return s2_h


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
    from src.garch.garch_params.core import egarch_kappa as eg_kappa

    kappa = eg_kappa(dist, nu, lambda_skew)
    s2_1 = _compute_one_step_forecast(e_last, s2_last, omega, alpha, gamma, beta, kappa)
    s2_h = _compute_multi_step_forecasts(s2_last, horizon, omega, beta)
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


__all__ = [
    "compute_variance_path",
    "compute_variance_path_for_test",
    "compute_egarch_forecasts",
    "compute_initial_forecasts",
]
