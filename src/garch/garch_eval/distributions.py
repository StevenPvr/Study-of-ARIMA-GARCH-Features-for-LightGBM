"""Distribution utilities for EGARCH evaluation (Hansen skew-t).

Implements closed-form CDF/PPF for the standardized Hansen (1994) skewed
Student-t distribution used in EGARCH variance models. These functions are
used for accurate VaR and prediction intervals when the selected innovation
distribution is skew-t.

Formulas follow Hansen (1994): define constants
    c = Gamma((nu+1)/2) / (sqrt(pi*(nu-2)) * Gamma(nu/2))
    a = 4 * lambda * c * (nu-2) / (nu-1)
    b = sqrt(1 + 3*lambda^2 - a^2)

Let T_nu be the CDF of the standard Student-t with nu degrees of freedom, and
z0 = -a/b. For z < z0,
    F(z) = (1 - lambda) * T_nu( sqrt(nu/(nu-2)) * (b*z + a) / (1 - lambda) )
Else (z >= z0),
    F(z) = (1 + lambda) * T_nu( sqrt(nu/(nu-2)) * (b*z + a) / (1 + lambda) ) - lambda

The inverse (PPF) is obtained by inverting the piecewise CDF.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.constants import GARCH_STUDENT_NU_MIN
from src.utils import get_logger

logger = get_logger(__name__)


def _skewt_abc(nu: float, lam: float) -> Tuple[float, float, float]:
    """Return Hansen skew-t constants (a, b, c) for nu>2 and |lam|<1.

    Parameters
    ----------
    nu : float
        Degrees of freedom (must be > 2 for finite variance).
    lam : float
        Skewness parameter in (-1, 1).

    Returns
    -------
    (a, b, c) : tuple of floats
        Hansen skew-t constants.
    """
    if not (nu is not None and float(nu) > GARCH_STUDENT_NU_MIN):
        raise ValueError("Skew-t requires nu > 2.0")
    lam = float(lam)
    try:
        from scipy.special import gammaln  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is required in this project
        raise RuntimeError("SciPy required for skew-t constants") from exc

    c_log = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * (
        np.log(np.pi) + np.log(nu - 2.0)
    )
    c = float(np.exp(c_log))
    a = float(4.0 * lam * c * (nu - 2.0) / (nu - 1.0))
    b_sq = float(1.0 + 3.0 * lam * lam - a * a)
    if b_sq <= 0.0:
        raise ValueError("Invalid (nu, lambda) for skew-t: b^2 <= 0")
    b = float(np.sqrt(b_sq))
    return a, b, c


def skewt_cdf(z: np.ndarray | float, nu: float, lam: float) -> np.ndarray | float:
    """CDF of the standardized Hansen skew-t at z.

    Parameters
    ----------
    z : array-like or float
        Evaluation point(s).
    nu : float
        Degrees of freedom (>2).
    lam : float
        Skewness parameter (-1 < lam < 1).

    Returns
    -------
    F : array-like or float
        CDF value(s) at z.
    """
    try:
        from scipy.stats import t as student_t  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is required
        raise RuntimeError("SciPy required for skew-t CDF") from exc

    a, b, _ = _skewt_abc(float(nu), float(lam))
    z_arr = np.asarray(z, dtype=float)
    s = float(np.sqrt(nu / (nu - 2.0)))
    thr = float(-a / b)

    x_left = s * (b * z_arr + a) / (1.0 - lam)
    x_right = s * (b * z_arr + a) / (1.0 + lam)
    mask_left = z_arr < thr

    F = np.empty_like(z_arr, dtype=float)
    # Left branch
    if np.any(mask_left):
        F[mask_left] = (1.0 - lam) * student_t.cdf(x_left[mask_left], df=nu)
    # Right branch
    if np.any(~mask_left):
        F[~mask_left] = (1.0 + lam) * student_t.cdf(x_right[~mask_left], df=nu) - lam

    return float(F.item()) if np.isscalar(z) else F


def skewt_ppf(q: np.ndarray | float, nu: float, lam: float) -> np.ndarray | float:
    """PPF (inverse CDF) of the standardized Hansen skew-t at probability q.

    Piecewise inversion using the closed-form CDF mapping to Student-t.

    Parameters
    ----------
    q : array-like or float
        Probability level(s) in (0,1).
    nu : float
        Degrees of freedom (>2).
    lam : float
        Skewness parameter (-1 < lam < 1).

    Returns
    -------
    z_q : array-like or float
        Quantile(s) such that P(Z <= z_q) = q.
    """
    try:
        from scipy.stats import t as student_t  # type: ignore
    except Exception as exc:  # pragma: no cover - SciPy is required
        raise RuntimeError("SciPy required for skew-t PPF") from exc

    a, b, _ = _skewt_abc(float(nu), float(lam))
    q_arr = np.asarray(q, dtype=float)
    s_inv = float(np.sqrt((nu - 2.0) / nu))
    p0 = 0.5 * (1.0 - lam)

    z_out = np.empty_like(q_arr, dtype=float)

    # Left branch: q < (1 - lambda)/2
    mask_left = q_arr < p0
    if np.any(mask_left):
        qq = q_arr[mask_left] / (1.0 - lam)
        x = student_t.ppf(qq, df=nu)
        z_out[mask_left] = ((1.0 - lam) * s_inv * x - a) / b

    # Right branch: q >= (1 - lambda)/2
    mask_right = ~mask_left
    if np.any(mask_right):
        qq = (q_arr[mask_right] + lam) / (1.0 + lam)
        x = student_t.ppf(qq, df=nu)
        z_out[mask_right] = ((1.0 + lam) * s_inv * x - a) / b

    return float(z_out.item()) if np.isscalar(q) else z_out


__all__ = ["skewt_cdf", "skewt_ppf"]

