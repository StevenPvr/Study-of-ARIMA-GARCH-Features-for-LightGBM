"""EGARCH model parameter representations.

This module provides a unified dataclass for EGARCH parameters,
eliminating duplication across estimation and optimization modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.garch.garch_params.core import (
    validate_beta,
    validate_skewt_params,
    validate_student_params,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Private helper functions for parameter extraction and building
# ============================================================================


def _extract_alpha_from_array(
    params: np.ndarray, idx: int, o: int
) -> tuple[float | tuple[float, float], int]:
    """Extract alpha parameter(s) from array.

    Args:
        params: Parameter array.
        idx: Current index in array.
        o: ARCH order (1 or 2).

    Returns:
        Tuple of (alpha, new_index) where alpha is float for o=1 or tuple for o=2.
    """
    if o == 1:
        return float(params[idx]), idx + 1
    # o == 2
    return (float(params[idx]), float(params[idx + 1])), idx + 2


def _extract_gamma_from_array(
    params: np.ndarray, idx: int, o: int
) -> tuple[float | tuple[float, float], int]:
    """Extract gamma parameter(s) from array.

    Args:
        params: Parameter array.
        idx: Current index in array.
        o: ARCH order (1 or 2).

    Returns:
        Tuple of (gamma, new_index) where gamma is float for o=1 or tuple for o=2.
    """
    if o == 1:
        return float(params[idx]), idx + 1
    # o == 2
    return (float(params[idx]), float(params[idx + 1])), idx + 2


def _extract_beta_from_array(
    params: np.ndarray, idx: int, p: int
) -> tuple[float | tuple[float, float] | tuple[float, float, float], int]:
    """Extract beta parameter(s) from array.

    Args:
        params: Parameter array.
        idx: Current index in array.
        p: GARCH order (1, 2, or 3).

    Returns:
        Tuple of (beta, new_index) where beta matches the GARCH order.
    """
    if p == 1:
        return float(params[idx]), idx + 1
    if p == 2:
        return (float(params[idx]), float(params[idx + 1])), idx + 2
    # p == 3
    return (float(params[idx]), float(params[idx + 1]), float(params[idx + 2])), idx + 3


def _extract_alpha_from_dict(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Extract alpha parameter(s) from dictionary.

    Args:
        params: Parameter dictionary.
        o: ARCH order (1 or 2).

    Returns:
        Alpha value(s) matching the order.

    Raises:
        KeyError: If required keys are missing.
    """
    if o == 1:
        return params["alpha"]
    # o == 2
    if "alpha1" in params and "alpha2" in params:
        return (params["alpha1"], params["alpha2"])
    if "alpha" in params and isinstance(params["alpha"], tuple):
        return params["alpha"]
    msg = f"EGARCH(2,p) requires alpha1 and alpha2, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_gamma_from_dict(params: dict[str, float], o: int) -> float | tuple[float, float]:
    """Extract gamma parameter(s) from dictionary.

    Args:
        params: Parameter dictionary.
        o: ARCH order (1 or 2).

    Returns:
        Gamma value(s) matching the order.

    Raises:
        KeyError: If required keys are missing.
    """
    if o == 1:
        return params["gamma"]
    # o == 2
    if "gamma1" in params and "gamma2" in params:
        return (params["gamma1"], params["gamma2"])
    if "gamma" in params and isinstance(params["gamma"], tuple):
        return params["gamma"]
    msg = f"EGARCH(o,2) requires gamma1 and gamma2, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_beta_p2(params: dict[str, float]) -> tuple[float, float]:
    """Extract beta parameters for p=2."""
    if "beta1" in params and "beta2" in params:
        return (params["beta1"], params["beta2"])
    if "beta" in params and isinstance(params["beta"], tuple):
        return params["beta"]
    msg = f"EGARCH(o,2) requires beta1 and beta2, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_beta_p3(params: dict[str, float]) -> tuple[float, float, float]:
    """Extract beta parameters for p=3."""
    if "beta1" in params and "beta2" in params and "beta3" in params:
        return (params["beta1"], params["beta2"], params["beta3"])
    if "beta" in params and isinstance(params["beta"], tuple):
        return params["beta"]
    msg = f"EGARCH(o,3) requires beta1, beta2, and beta3, got {list(params.keys())}"
    raise KeyError(msg)


def _extract_beta_from_dict(
    params: dict[str, float], p: int
) -> float | tuple[float, float] | tuple[float, float, float]:
    """Extract beta parameter(s) from dictionary.

    Args:
        params: Parameter dictionary.
        p: GARCH order (1, 2, or 3).

    Returns:
        Beta value(s) matching the order.

    Raises:
        KeyError: If required keys are missing.
    """
    if p == 1:
        return params["beta"]
    if p == 2:
        return _extract_beta_p2(params)
    # p == 3
    return _extract_beta_p3(params)


def _build_alpha_dict_entries(alpha: float | tuple[float, float], o: int) -> dict[str, float]:
    """Build dictionary entries for alpha parameter(s).

    Args:
        alpha: Alpha value(s).
        o: ARCH order (1 or 2).

    Returns:
        Dictionary with alpha entries.

    Raises:
        TypeError: If alpha type doesn't match order.
    """
    if o == 1:
        if isinstance(alpha, tuple):
            msg = "alpha should be float for o=1"
            raise TypeError(msg)
        return {"alpha": float(alpha)}
    # o == 2
    if not isinstance(alpha, tuple):
        msg = "alpha should be tuple for o=2"
        raise TypeError(msg)
    return {"alpha1": float(alpha[0]), "alpha2": float(alpha[1])}


def _build_gamma_dict_entries(gamma: float | tuple[float, float], o: int) -> dict[str, float]:
    """Build dictionary entries for gamma parameter(s).

    Args:
        gamma: Gamma value(s).
        o: ARCH order (1 or 2).

    Returns:
        Dictionary with gamma entries.

    Raises:
        TypeError: If gamma type doesn't match order.
    """
    if o == 1:
        if isinstance(gamma, tuple):
            msg = "gamma should be float for o=1"
            raise TypeError(msg)
        return {"gamma": float(gamma)}
    # o == 2
    if not isinstance(gamma, tuple):
        msg = "gamma should be tuple for o=2"
        raise TypeError(msg)
    return {"gamma1": float(gamma[0]), "gamma2": float(gamma[1])}


def _build_beta_dict_entries(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> dict[str, float]:
    """Build dictionary entries for beta parameter(s).

    Args:
        beta: Beta value(s).
        p: GARCH order (1, 2, or 3).

    Returns:
        Dictionary with beta entries.

    Raises:
        TypeError: If beta type doesn't match order.
    """
    if p == 1:
        if isinstance(beta, tuple):
            msg = "beta should be float for p=1"
            raise TypeError(msg)
        return {"beta": float(beta)}
    if p == 2:
        if not isinstance(beta, tuple) or len(beta) != 2:
            msg = "beta should be tuple of 2 floats for p=2"
            raise TypeError(msg)
        return {"beta1": float(beta[0]), "beta2": float(beta[1])}
    # p == 3
    if not isinstance(beta, tuple) or len(beta) != 3:
        msg = "beta should be tuple of 3 floats for p=3"
        raise TypeError(msg)
    return {
        "beta1": float(beta[0]),
        "beta2": float(beta[1]),
        "beta3": float(beta[2]),
    }


def _append_alpha_to_list(
    params_list: list[float], alpha: float | tuple[float, float], o: int
) -> None:
    """Append alpha parameter(s) to list.

    Args:
        params_list: List to append to.
        alpha: Alpha value(s).
        o: ARCH order (1 or 2).

    Raises:
        TypeError: If alpha type doesn't match order.
    """
    if o == 1:
        if isinstance(alpha, tuple):
            msg = "alpha should be float for o=1"
            raise TypeError(msg)
        params_list.append(float(alpha))
    else:  # o == 2
        if not isinstance(alpha, tuple):
            msg = "alpha should be tuple for o=2"
            raise TypeError(msg)
        params_list.extend([float(alpha[0]), float(alpha[1])])


def _append_gamma_to_list(
    params_list: list[float], gamma: float | tuple[float, float], o: int
) -> None:
    """Append gamma parameter(s) to list.

    Args:
        params_list: List to append to.
        gamma: Gamma value(s).
        o: ARCH order (1 or 2).

    Raises:
        TypeError: If gamma type doesn't match order.
    """
    if o == 1:
        if isinstance(gamma, tuple):
            msg = "gamma should be float for o=1"
            raise TypeError(msg)
        params_list.append(float(gamma))
    else:  # o == 2
        if not isinstance(gamma, tuple):
            msg = "gamma should be tuple for o=2"
            raise TypeError(msg)
        params_list.extend([float(gamma[0]), float(gamma[1])])


def _append_beta_to_list(
    params_list: list[float],
    beta: float | tuple[float, float] | tuple[float, float, float],
    p: int,
) -> None:
    """Append beta parameter(s) to list.

    Args:
        params_list: List to append to.
        beta: Beta value(s).
        p: GARCH order (1, 2, or 3).

    Raises:
        TypeError: If beta type doesn't match order.
    """
    if p == 1:
        if isinstance(beta, tuple):
            msg = "beta should be float for p=1"
            raise TypeError(msg)
        params_list.append(float(beta))
    elif p == 2:
        if not isinstance(beta, tuple) or len(beta) != 2:
            msg = "beta should be tuple of 2 floats for p=2"
            raise TypeError(msg)
        params_list.extend([float(beta[0]), float(beta[1])])
    else:  # p == 3
        if not isinstance(beta, tuple) or len(beta) != 3:
            msg = "beta should be tuple of 3 floats for p=3"
            raise TypeError(msg)
        params_list.extend([float(beta[0]), float(beta[1]), float(beta[2])])


def _validate_order_constraints(o: int, p: int) -> None:
    """Validate ARCH and GARCH order constraints.

    Args:
        o: ARCH order.
        p: GARCH order.

    Raises:
        ValueError: If orders are not supported.
    """
    if o not in (1, 2):
        msg = f"ARCH order o={o} not supported (only o=1 or 2)"
        raise ValueError(msg)
    if p not in (1, 2, 3):
        msg = f"GARCH order p={p} not supported (only p=1, 2, or 3)"
        raise ValueError(msg)


def _validate_alpha_type(alpha: float | tuple[float, float], o: int) -> None:
    """Validate alpha parameter type matches ARCH order.

    Args:
        alpha: Alpha parameter(s).
        o: ARCH order.

    Raises:
        TypeError: If alpha type doesn't match order.
    """
    if o == 1:
        if not isinstance(alpha, (int, float)):
            msg = f"alpha must be float for o=1, got {type(alpha)}"
            raise TypeError(msg)
    else:  # o == 2
        if not isinstance(alpha, tuple) or len(alpha) != 2:
            msg = f"alpha must be tuple of 2 floats for o=2, got {alpha}"
            raise TypeError(msg)


def _validate_gamma_type(gamma: float | tuple[float, float], o: int) -> None:
    """Validate gamma parameter type matches ARCH order.

    Args:
        gamma: Gamma parameter(s).
        o: ARCH order.

    Raises:
        TypeError: If gamma type doesn't match order.
    """
    if o == 1:
        if not isinstance(gamma, (int, float)):
            msg = f"gamma must be float for o=1, got {type(gamma)}"
            raise TypeError(msg)
    else:  # o == 2
        if not isinstance(gamma, tuple) or len(gamma) != 2:
            msg = f"gamma must be tuple of 2 floats for o=2, got {gamma}"
            raise TypeError(msg)


def _validate_beta_type(
    beta: float | tuple[float, float] | tuple[float, float, float], p: int
) -> None:
    """Validate beta parameter type matches GARCH order.

    Args:
        beta: Beta parameter(s).
        p: GARCH order.

    Raises:
        TypeError: If beta type doesn't match order.
    """
    if p == 1:
        if not isinstance(beta, (int, float)):
            msg = f"beta must be float for p=1, got {type(beta)}"
            raise TypeError(msg)
    elif p == 2:
        if not isinstance(beta, tuple) or len(beta) != 2:
            msg = f"beta must be tuple of 2 floats for p=2, got {beta}"
            raise TypeError(msg)
    else:  # p == 3
        if not isinstance(beta, tuple) or len(beta) != 3:
            msg = f"beta must be tuple of 3 floats for p=3, got {beta}"
            raise TypeError(msg)


def _validate_normal_params(nu: float | None, lambda_skew: float | None) -> None:
    """Validate Normal distribution parameters."""
    if nu is not None or lambda_skew is not None:
        logger.warning("Normal distribution should not have nu or lambda_skew")


def _validate_distribution_params(dist: str, nu: float | None, lambda_skew: float | None) -> None:
    """Validate distribution-specific parameters.

    Args:
        dist: Distribution name.
        nu: Degrees of freedom parameter.
        lambda_skew: Skewness parameter.

    Raises:
        ValueError: If distribution is not supported.
    """
    dist_lower = dist.lower()
    validators = {
        "normal": _validate_normal_params,
        "student": lambda nu, ls: None,  # Student validation handled elsewhere
        "skewt": lambda nu, ls: None,  # Skew-t validation handled elsewhere
    }

    if dist_lower not in validators:
        msg = f"Distribution '{dist}' not supported"
        raise ValueError(msg)

    validator = validators[dist_lower]
    validator(nu, lambda_skew)


def _extract_dist_params_from_array(
    params: np.ndarray, idx: int, dist: str
) -> tuple[float | None, float | None]:
    """Extract distribution-specific parameters from array.

    Args:
        params: Parameter array.
        idx: Current index in array.
        dist: Distribution name.

    Returns:
        Tuple of (nu, lambda_skew).
    """
    nu = None
    lambda_skew = None
    dist_lower = dist.lower()

    if dist_lower == "student":
        nu = float(params[idx])
    elif dist_lower == "skewt":
        nu = float(params[idx])
        lambda_skew = float(params[idx + 1])

    return nu, lambda_skew


@dataclass(frozen=True)
class EGARCHParams:
    """EGARCH model parameters.

    Attributes:
        omega: Constant term in log-variance equation.
        alpha: ARCH coefficient(s) - float for o=1, tuple for o=2.
        gamma: Asymmetry coefficient(s) - float for o=1, tuple for o=2.
        beta: GARCH coefficient(s) - float for p=1, tuple for p=2 or p=3.
        nu: Degrees of freedom parameter (Student-t, Skew-t).
        lambda_skew: Skewness parameter (Skew-t).
        o: ARCH order (1 or 2).
        p: GARCH order (1, 2, or 3).
        dist: Distribution name ('normal', 'student', 'skewt').
        loglik: Log-likelihood value (optional).
        converged: Convergence status (optional).
    """

    omega: float
    alpha: float | tuple[float, float]
    gamma: float | tuple[float, float]
    beta: float | tuple[float, float] | tuple[float, float, float]
    nu: float | None
    lambda_skew: float | None
    o: int
    p: int
    dist: str
    loglik: float | None = None
    converged: bool | None = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        _validate_order_constraints(self.o, self.p)
        _validate_alpha_type(self.alpha, self.o)
        _validate_gamma_type(self.gamma, self.o)
        _validate_beta_type(self.beta, self.p)
        _validate_distribution_params(self.dist, self.nu, self.lambda_skew)

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to dictionary representation.

        Returns:
            Dictionary with parameter names as keys.
        """
        logger.debug(f"Converting EGARCH({self.o},{self.p}) parameters to dict")

        out: dict[str, float] = {"omega": self.omega}

        # Add alpha/gamma/beta using helpers
        out.update(_build_alpha_dict_entries(self.alpha, self.o))
        out.update(_build_gamma_dict_entries(self.gamma, self.o))
        out.update(_build_beta_dict_entries(self.beta, self.p))

        # Add distribution parameters
        if self.nu is not None:
            out["nu"] = float(self.nu)
        if self.lambda_skew is not None:
            out["lambda"] = float(self.lambda_skew)

        # Add metadata
        if self.loglik is not None:
            out["loglik"] = float(self.loglik)
        if self.converged is not None:
            out["converged"] = float(self.converged)

        return out

    def to_array(self) -> np.ndarray:
        """Convert parameters to array representation for optimization.

        Returns:
            1D numpy array with parameters in standard order.
        """
        logger.debug(f"Converting EGARCH({self.o},{self.p}) parameters to array")

        params_list: list[float] = [self.omega]

        # Add alpha/gamma/beta using helpers
        _append_alpha_to_list(params_list, self.alpha, self.o)
        _append_gamma_to_list(params_list, self.gamma, self.o)
        _append_beta_to_list(params_list, self.beta, self.p)

        # Add distribution parameters
        if self.nu is not None:
            params_list.append(float(self.nu))
        if self.lambda_skew is not None:
            params_list.append(float(self.lambda_skew))

        return np.array(params_list, dtype=float)

    @classmethod
    def from_array(
        cls,
        params: np.ndarray,
        o: int,
        p: int,
        dist: str,
        loglik: float | None = None,
        converged: bool | None = None,
    ) -> EGARCHParams:
        """Create EGARCHParams from parameter array.

        Args:
            params: Parameter vector.
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.
            loglik: Optional log-likelihood value.
            converged: Optional convergence status.

        Returns:
            EGARCHParams instance.
        """
        logger.debug(f"Creating EGARCH({o},{p}) parameters from array")

        idx = 0
        omega = float(params[idx])
        idx += 1

        alpha, idx = _extract_alpha_from_array(params, idx, o)
        gamma, idx = _extract_gamma_from_array(params, idx, o)
        beta, idx = _extract_beta_from_array(params, idx, p)
        nu, lambda_skew = _extract_dist_params_from_array(params, idx, dist)

        return cls(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            nu=nu,
            lambda_skew=lambda_skew,
            o=o,
            p=p,
            dist=dist,
            loglik=loglik,
            converged=converged,
        )

    @classmethod
    def from_dict(
        cls,
        params: dict[str, float],
        o: int,
        p: int,
        dist: str,
    ) -> EGARCHParams:
        """Create EGARCHParams from parameter dictionary.

        Args:
            params: Parameter dictionary.
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.

        Returns:
            EGARCHParams instance.

        Raises:
            KeyError: If required parameters are missing.
            ValueError: If parameter types don't match orders.
        """
        logger.debug(f"Creating EGARCH({o},{p}) parameters from dict")

        omega = params["omega"]

        # Extract alpha, gamma, beta using helpers
        alpha = _extract_alpha_from_dict(params, o)
        gamma = _extract_gamma_from_dict(params, o)
        beta = _extract_beta_from_dict(params, p)

        # Extract distribution parameters
        nu = params.get("nu")
        lambda_skew = params.get("lambda")

        # Extract metadata
        loglik = params.get("loglik")
        converged_val = params.get("converged")
        converged = bool(converged_val) if converged_val is not None else None

        return cls(
            omega=omega,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            nu=nu,
            lambda_skew=lambda_skew,
            o=o,
            p=p,
            dist=dist,
            loglik=loglik,
            converged=converged,
        )

    @classmethod
    def from_optimization_result(
        cls,
        result: Any,
        o: int,
        p: int,
        dist: str,
    ) -> EGARCHParams:
        """Create EGARCHParams from scipy optimization result.

        Args:
            result: Optimization result object with x, fun, success attributes.
            o: ARCH order.
            p: GARCH order.
            dist: Distribution name.

        Returns:
            EGARCHParams instance.
        """
        loglik = float(-result.fun) if hasattr(result, "fun") else None
        converged = bool(result.success) if hasattr(result, "success") else None

        return cls.from_array(
            params=result.x,
            o=o,
            p=p,
            dist=dist,
            loglik=loglik,
            converged=converged,
        )

    def get_beta_representative(self) -> float:
        """Get representative beta value for stationarity checks.

        Returns:
            Single beta value (for p=1) or sum of betas (for p=2).
        """
        if self.p == 1:
            if isinstance(self.beta, tuple):
                msg = "beta should be float for p=1"
                raise TypeError(msg)
            return float(self.beta)
        # p == 2
        if not isinstance(self.beta, tuple):
            msg = "beta should be tuple for p=2"
            raise TypeError(msg)
        return float(self.beta[0] + self.beta[1])

    def validate_stationarity(self) -> bool:
        """Validate parameter stationarity conditions.

        Returns:
            True if parameters satisfy stationarity constraints.
        """
        beta_val = self.get_beta_representative()

        dist_lower = self.dist.lower()
        if dist_lower == "normal":
            return validate_beta(beta_val)
        if dist_lower == "student":
            if self.nu is None:
                return False
            return validate_student_params(beta_val, self.nu)
        if dist_lower == "skewt":
            if self.nu is None or self.lambda_skew is None:
                return False
            return validate_skewt_params(beta_val, self.nu, self.lambda_skew)
        return False

    def extract_for_variance(
        self,
    ) -> tuple[
        float | tuple[float, float],
        float | tuple[float, float],
        float | tuple[float, float] | tuple[float, float, float],
    ]:
        """Extract (alpha, gamma, beta) tuple for variance computation.

        This method provides the parameters in the exact format required by
        variance filtering and forecasting functions.

        Returns:
            Tuple of (alpha, gamma, beta) where:
            - alpha/gamma: float for o=1, tuple[float, float] for o=2
            - beta: float for p=1, tuple[float, float] for p=2, tuple[float, float, float] for p=3

        Example:
            >>> params = EGARCHParams(omega=0.1, alpha=0.2, gamma=-0.1, beta=0.9,
            ...                       nu=None, lambda_skew=None, o=1, p=1, dist="normal")
            >>> alpha, gamma, beta = params.extract_for_variance()
            >>> # alpha=0.2, gamma=-0.1, beta=0.9 (all floats)

            >>> params = EGARCHParams(omega=0.1, alpha=(0.2, 0.15), gamma=(-0.1, -0.05),
            ...                       beta=(0.8, 0.1), nu=None, lambda_skew=None,
            ...                       o=2, p=2, dist="normal")
            >>> alpha, gamma, beta = params.extract_for_variance()
            >>> # alpha=(0.2, 0.15), gamma=(-0.1, -0.05), beta=(0.8, 0.1)
        """
        return self.alpha, self.gamma, self.beta

    def __str__(self) -> str:
        """Return string representation of parameters."""
        parts = [f"EGARCH({self.o},{self.p})", f"dist={self.dist}"]

        params_dict = self.to_dict()
        if self.o == 1:
            parts.append(f"alpha={params_dict['alpha']:.4f}")
            parts.append(f"gamma={params_dict['gamma']:.4f}")
        else:
            parts.append(f"alpha=({params_dict['alpha1']:.4f}, {params_dict['alpha2']:.4f})")
            parts.append(f"gamma=({params_dict['gamma1']:.4f}, {params_dict['gamma2']:.4f})")

        if self.p == 1:
            parts.append(f"beta={params_dict['beta']:.4f}")
        else:
            parts.append(f"beta=({params_dict['beta1']:.4f}, {params_dict['beta2']:.4f})")

        if self.nu is not None:
            parts.append(f"nu={self.nu:.2f}")
        if self.lambda_skew is not None:
            parts.append(f"lambda={self.lambda_skew:.4f}")

        if self.loglik is not None:
            parts.append(f"loglik={self.loglik:.2f}")

        return ", ".join(parts)
