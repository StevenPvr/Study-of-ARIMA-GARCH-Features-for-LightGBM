"""Unit tests for GARCH core modules (validation, variance, distributions)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import (
    GARCH_ESTIMATION_BETA_MAX,
    GARCH_ESTIMATION_NU_MIN_THRESHOLD,
    GARCH_MIN_INIT_VAR,
    GARCH_SKEWT_LAMBDA_MAX,
)
from src.garch.garch_params.core.distributions import (
    compute_kappa,
    compute_kappa_normal,
    compute_kappa_skewt,
    compute_kappa_student,
    compute_loglik_normal,
    compute_loglik_skewt,
    compute_loglik_student,
)
from src.garch.garch_params.core.validation import (
    validate_beta,
    validate_residuals,
    validate_skewt_params,
    validate_student_params,
)
from src.garch.garch_params.core.variance import (
    clip_and_exp_log_variance,
    compute_variance_path,
    compute_variance_path_egarch11,
    compute_variance_step_egarch11,
    initialize_variance,
    safe_variance,
    validate_param_types,
)


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 0.01, size=200)


# ==================== Validation Tests ====================


def test_validate_residuals_valid(sample_residuals: np.ndarray) -> None:
    """Test validate_residuals with valid input."""
    result = validate_residuals(sample_residuals)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_residuals.shape


def test_validate_residuals_insufficient() -> None:
    """Test validate_residuals raises ValueError for insufficient data."""
    resid = np.array([1.0, 2.0])  # Too few observations
    with pytest.raises(ValueError, match="at least"):
        validate_residuals(resid)


def test_validate_beta_valid() -> None:
    """Test validate_beta with valid beta."""
    beta = 0.9
    assert validate_beta(beta) is True


def test_validate_beta_invalid() -> None:
    """Test validate_beta with invalid beta."""
    beta = GARCH_ESTIMATION_BETA_MAX + 0.1
    assert validate_beta(beta) is False


def test_validate_student_params_valid() -> None:
    """Test validate_student_params with valid parameters."""
    beta = 0.9
    nu = 5.0
    assert validate_student_params(beta, nu) is True


def test_validate_student_params_invalid_nu() -> None:
    """Test validate_student_params with invalid nu."""
    beta = 0.9
    nu = GARCH_ESTIMATION_NU_MIN_THRESHOLD - 0.1
    assert validate_student_params(beta, nu) is False


def test_validate_skewt_params_valid() -> None:
    """Test validate_skewt_params with valid parameters."""
    beta = 0.9
    nu = 5.0
    lambda_skew = 0.1
    assert validate_skewt_params(beta, nu, lambda_skew) is True


def test_validate_skewt_params_invalid_lambda() -> None:
    """Test validate_skewt_params with invalid lambda."""
    beta = 0.9
    nu = 5.0
    lambda_skew = GARCH_SKEWT_LAMBDA_MAX + 0.1
    assert validate_skewt_params(beta, nu, lambda_skew) is False


# ==================== Variance Tests ====================


def test_initialize_variance_with_init() -> None:
    """Test initialize_variance with provided initial value."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=100)
    init = 0.001
    result = initialize_variance(residuals, init)
    assert result == init


def test_initialize_variance_without_init() -> None:
    """Test initialize_variance computes sample variance."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=100)
    result = initialize_variance(residuals, None)
    assert result >= GARCH_MIN_INIT_VAR
    assert np.isfinite(result)


def test_safe_variance() -> None:
    """Test safe_variance enforces minimum threshold."""
    result = safe_variance(1e-10)
    assert result >= GARCH_MIN_INIT_VAR


def test_clip_and_exp_log_variance_valid() -> None:
    """Test clip_and_exp_log_variance with valid input."""
    log_var = -5.0
    result = clip_and_exp_log_variance(log_var)
    assert result > 0
    assert np.isfinite(result)


def test_clip_and_exp_log_variance_invalid() -> None:
    """Test clip_and_exp_log_variance with invalid input."""
    log_var = float("inf")
    result = clip_and_exp_log_variance(log_var)
    assert np.isnan(result)


def test_compute_variance_step_egarch11(sample_residuals: np.ndarray) -> None:
    """Test compute_variance_step_egarch11."""
    residual = sample_residuals[0]
    variance_prev = 0.0001
    omega = -5.0
    alpha = 0.1
    gamma = 0.0
    beta = 0.95
    kappa = np.sqrt(2.0 / np.pi)

    result = compute_variance_step_egarch11(
        residual, variance_prev, omega, alpha, gamma, beta, kappa
    )

    assert result > 0
    assert np.isfinite(result)


def test_compute_variance_path_egarch11(sample_residuals: np.ndarray) -> None:
    """Test compute_variance_path_egarch11."""
    omega = -5.0
    alpha = 0.1
    gamma = 0.0
    beta = 0.95
    kappa = np.sqrt(2.0 / np.pi)

    variances = compute_variance_path_egarch11(sample_residuals, omega, alpha, gamma, beta, kappa)

    assert variances.shape == sample_residuals.shape
    assert np.all(variances > 0)
    assert np.all(np.isfinite(variances))


def test_compute_variance_path_egarch11_with_init(sample_residuals: np.ndarray) -> None:
    """Test compute_variance_path_egarch11 with custom initialization."""
    omega = -5.0
    alpha = 0.1
    gamma = 0.0
    beta = 0.95
    kappa = np.sqrt(2.0 / np.pi)
    init = 0.001

    variances = compute_variance_path_egarch11(
        sample_residuals, omega, alpha, gamma, beta, kappa, init=init
    )

    assert variances[0] == init
    assert np.all(variances > 0)


def test_compute_variance_path_egarch11_normal(sample_residuals: np.ndarray) -> None:
    """Test compute_variance_path for EGARCH(1,1) with normal distribution."""
    omega = -5.0
    alpha = 0.1
    gamma = 0.0
    beta = 0.95
    kappa = compute_kappa("normal")

    variances = compute_variance_path(
        sample_residuals,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        kappa=kappa,
        o=1,
        p=1,
    )

    assert variances.shape == sample_residuals.shape
    assert np.all(variances > 0)
    assert np.all(np.isfinite(variances))


def test_compute_variance_path_egarch12_normal(sample_residuals: np.ndarray) -> None:
    """Test compute_variance_path for EGARCH(1,2) with normal distribution."""
    omega = -5.0
    alpha = 0.1
    gamma = 0.0
    beta = (0.5, 0.4)
    kappa = compute_kappa("normal")

    variances = compute_variance_path(
        sample_residuals,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        kappa=kappa,
        o=1,
        p=2,
    )

    assert variances.shape == sample_residuals.shape
    assert np.all(variances > 0)
    assert np.all(np.isfinite(variances))


def test_validate_param_types_11() -> None:
    """Test validate_param_types for o=1, p=1."""
    alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = validate_param_types(
        alpha=0.1, gamma=0.0, beta=0.95, o=1, p=1
    )
    assert alpha1 == 0.1
    assert alpha2 == 0.0
    assert gamma1 == 0.0
    assert gamma2 == 0.0
    assert beta1 == 0.95
    assert beta2 == 0.0
    assert beta3 == 0.0


def test_validate_param_types_12() -> None:
    """Test validate_param_types for o=1, p=2."""
    alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = validate_param_types(
        alpha=0.1, gamma=0.0, beta=(0.5, 0.4), o=1, p=2
    )
    assert alpha1 == 0.1
    assert beta1 == 0.5
    assert beta2 == 0.4
    assert beta3 == 0.0


def test_validate_param_types_21() -> None:
    """Test validate_param_types for o=2, p=1."""
    alpha1, alpha2, gamma1, gamma2, beta1, beta2, beta3 = validate_param_types(
        alpha=(0.05, 0.05), gamma=(0.0, 0.0), beta=0.9, o=2, p=1
    )
    assert alpha1 == 0.05
    assert alpha2 == 0.05
    assert beta1 == 0.9
    assert beta2 == 0.0
    assert beta3 == 0.0


# ==================== Distribution Tests ====================


def test_compute_kappa_normal() -> None:
    """Test compute_kappa_normal."""
    kappa = compute_kappa_normal()
    expected = np.sqrt(2.0 / np.pi)
    assert np.isclose(kappa, expected, rtol=1e-10)


def test_compute_kappa_student() -> None:
    """Test compute_kappa_student."""
    nu = 5.0
    kappa = compute_kappa_student(nu)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_compute_kappa_student_invalid_nu() -> None:
    """Test compute_kappa_student raises ValueError for invalid nu."""
    nu = GARCH_ESTIMATION_NU_MIN_THRESHOLD - 0.1
    with pytest.raises(ValueError, match="Invalid Student-t parameter"):
        compute_kappa_student(nu)


def test_compute_kappa_skewt() -> None:
    """Test compute_kappa_skewt."""
    nu = 5.0
    lambda_skew = 0.1
    kappa = compute_kappa_skewt(nu, lambda_skew)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_compute_kappa_skewt_invalid_nu() -> None:
    """Test compute_kappa_skewt raises ValueError for invalid nu."""
    nu = GARCH_ESTIMATION_NU_MIN_THRESHOLD - 0.1
    lambda_skew = 0.1
    with pytest.raises(ValueError, match="Invalid Skew-t parameter"):
        compute_kappa_skewt(nu, lambda_skew)


def test_compute_kappa_skewt_invalid_lambda() -> None:
    """Test compute_kappa_skewt raises ValueError for invalid lambda."""
    nu = 5.0
    lambda_skew = GARCH_SKEWT_LAMBDA_MAX + 0.1
    with pytest.raises(ValueError, match="Invalid Skew-t parameter"):
        compute_kappa_skewt(nu, lambda_skew)


def test_compute_kappa_normal_wrapper() -> None:
    """Test compute_kappa for normal distribution."""
    kappa = compute_kappa("normal", None, None)
    expected = np.sqrt(2.0 / np.pi)
    assert np.isclose(kappa, expected, rtol=1e-10)


def test_compute_kappa_student_wrapper() -> None:
    """Test compute_kappa for student distribution."""
    nu = 5.0
    kappa = compute_kappa("student", nu, None)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_compute_kappa_skewt_wrapper() -> None:
    """Test compute_kappa for skewt distribution."""
    nu = 5.0
    lambda_skew = 0.1
    kappa = compute_kappa("skewt", nu, lambda_skew)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_compute_kappa_invalid_distribution() -> None:
    """Test compute_kappa raises ValueError for invalid distribution."""
    with pytest.raises(ValueError, match="Unsupported distribution"):
        compute_kappa("invalid", None, None)


def test_compute_loglik_normal(sample_residuals: np.ndarray) -> None:
    """Test compute_loglik_normal."""
    variances = np.ones_like(sample_residuals) * 0.0001
    loglik = compute_loglik_normal(sample_residuals, variances)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


def test_compute_loglik_student(sample_residuals: np.ndarray) -> None:
    """Test compute_loglik_student."""
    variances = np.ones_like(sample_residuals) * 0.0001
    nu = 5.0
    loglik = compute_loglik_student(sample_residuals, variances, nu)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


def test_compute_loglik_skewt(sample_residuals: np.ndarray) -> None:
    """Test compute_loglik_skewt."""
    variances = np.ones_like(sample_residuals) * 0.0001
    nu = 5.0
    lambda_skew = 0.1
    loglik = compute_loglik_skewt(sample_residuals, variances, nu, lambda_skew)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
