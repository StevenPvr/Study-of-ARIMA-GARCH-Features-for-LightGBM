"""Unit tests for GARCH computation functions."""

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
    GARCH_ESTIMATION_BETA_MIN,
    GARCH_ESTIMATION_NU_MIN_THRESHOLD,
    GARCH_MIN_INIT_VAR,
    GARCH_SKEWT_LAMBDA_MAX,
)
from src.garch.garch_params.core import (
    compute_normal_loglikelihood,
    compute_skewt_loglikelihood,
    compute_student_loglikelihood,
    egarch11_variance,
    egarch_kappa,
    egarch_variance,
    initialize_variance,
    validate_beta,
    validate_series,
    validate_skewt_params,
    validate_student_params,
)


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 0.01, size=200)


@pytest.fixture
def large_sample_residuals() -> np.ndarray:
    """Generate larger sample residuals for testing."""
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 0.01, size=600)


# ==================== Validation Tests ====================


def test_validate_series_valid(sample_residuals: np.ndarray) -> None:
    """Test validate_series with valid input."""
    result = validate_series(sample_residuals)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_residuals.shape
    assert result.dtype == float


def test_validate_series_insufficient_observations() -> None:
    """Test validate_series raises ValueError for insufficient observations."""
    e = np.array([1.0, 2.0])  # Too few observations
    with pytest.raises(ValueError, match="at least"):
        validate_series(e)


def test_validate_series_2d_array() -> None:
    """Test validate_series converts 2D array to 1D."""
    rng = np.random.default_rng(42)
    e_2d = rng.normal(0.0, 0.01, size=(100, 1))
    result = validate_series(e_2d)
    assert result.ndim == 1
    assert result.size == 100


def test_validate_beta_valid() -> None:
    """Test validate_beta with valid beta."""
    beta = 0.9
    assert validate_beta(beta) is True


def test_validate_beta_too_low() -> None:
    """Test validate_beta with beta too low."""
    beta = GARCH_ESTIMATION_BETA_MIN - 0.1
    assert validate_beta(beta) is False


def test_validate_beta_too_high() -> None:
    """Test validate_beta with beta too high."""
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


# ==================== Variance Initialization Tests ====================


def test_initialize_variance_with_init() -> None:
    """Test initialize_variance with provided initial value."""
    rng = np.random.default_rng(42)
    ee = rng.normal(0.0, 0.01, size=100)
    init = 0.001
    result = initialize_variance(ee, init)
    assert result == init


def test_initialize_variance_without_init() -> None:
    """Test initialize_variance computes sample variance."""
    rng = np.random.default_rng(42)
    ee = rng.normal(0.0, 0.01, size=100)
    result = initialize_variance(ee, None)
    assert result >= GARCH_MIN_INIT_VAR
    assert np.isfinite(result)


def test_initialize_variance_minimum_threshold() -> None:
    """Test initialize_variance enforces minimum threshold."""
    rng = np.random.default_rng(42)
    ee = rng.normal(0.0, 1e-10, size=100)  # Very small variance
    result = initialize_variance(ee, None)
    assert result >= GARCH_MIN_INIT_VAR


# ==================== Kappa Tests ====================


def test_egarch_kappa_normal() -> None:
    """Test egarch_kappa for normal distribution."""
    kappa = egarch_kappa("normal", None, None)
    expected = np.sqrt(2.0 / np.pi)
    assert np.isclose(kappa, expected, rtol=1e-10)


def test_egarch_kappa_student() -> None:
    """Test egarch_kappa for student distribution."""
    nu = 5.0
    kappa = egarch_kappa("student", nu, None)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_egarch_kappa_skewt() -> None:
    """Test egarch_kappa for skewt distribution."""
    nu = 5.0
    lambda_skew = 0.1
    kappa = egarch_kappa("skewt", nu, lambda_skew)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_egarch_kappa_invalid_distribution() -> None:
    """Test egarch_kappa raises ValueError for invalid distribution."""
    with pytest.raises(ValueError, match="Unsupported distribution"):
        egarch_kappa("invalid", None, None)


def test_egarch_kappa_student_missing_nu() -> None:
    """Test egarch_kappa raises ValueError for student without nu."""
    with pytest.raises(ValueError, match="requires nu"):
        egarch_kappa("student", None, None)


def test_egarch_kappa_skewt_missing_params() -> None:
    """Test egarch_kappa raises ValueError for skewt without params."""
    with pytest.raises(ValueError, match="requires both nu and lambda"):
        egarch_kappa("skewt", None, None)


# ==================== Variance Computation Tests ====================


def test_egarch11_variance_positive(sample_residuals: np.ndarray) -> None:
    """Test that EGARCH(1,1) variance is always positive and finite."""
    s2 = egarch11_variance(
        sample_residuals, omega=-5.0, alpha=0.1, gamma=0.0, beta=0.95, dist="normal"
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch11_variance_with_init(sample_residuals: np.ndarray) -> None:
    """Test EGARCH(1,1) variance with custom initialization."""
    init_var = 0.001
    s2 = egarch11_variance(
        sample_residuals,
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        dist="normal",
        init=init_var,
    )
    assert s2[0] == init_var
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch11_variance_skewt_dist(sample_residuals: np.ndarray) -> None:
    """Test EGARCH(1,1) variance with Skew-t distribution."""
    s2 = egarch11_variance(
        sample_residuals,
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        dist="skewt",
        nu=5.0,
        lambda_skew=-0.1,
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch_variance_egarch11(sample_residuals: np.ndarray) -> None:
    """Test egarch_variance for EGARCH(1,1)."""
    s2 = egarch_variance(
        sample_residuals,
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=0.95,
        o=1,
        p=1,
        dist="normal",
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch_variance_egarch12(sample_residuals: np.ndarray) -> None:
    """Test egarch_variance for EGARCH(1,2)."""
    s2 = egarch_variance(
        sample_residuals,
        omega=-5.0,
        alpha=0.1,
        gamma=0.0,
        beta=(0.5, 0.4),
        o=1,
        p=2,
        dist="normal",
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch_variance_egarch21(sample_residuals: np.ndarray) -> None:
    """Test egarch_variance for EGARCH(2,1)."""
    s2 = egarch_variance(
        sample_residuals,
        omega=-5.0,
        alpha=(0.05, 0.05),
        gamma=(0.0, 0.0),
        beta=0.9,
        o=2,
        p=1,
        dist="normal",
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch_variance_egarch22(sample_residuals: np.ndarray) -> None:
    """Test egarch_variance for EGARCH(2,2)."""
    s2 = egarch_variance(
        sample_residuals,
        omega=-5.0,
        alpha=(0.05, 0.05),
        gamma=(0.0, 0.0),
        beta=(0.5, 0.4),
        o=2,
        p=2,
        dist="normal",
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


# ==================== Log-Likelihood Tests ====================


def test_compute_normal_loglikelihood(sample_residuals: np.ndarray) -> None:
    """Test compute_normal_loglikelihood."""
    variances = np.ones_like(sample_residuals) * 0.0001
    loglik = compute_normal_loglikelihood(sample_residuals, variances)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


def test_compute_student_loglikelihood(sample_residuals: np.ndarray) -> None:
    """Test compute_student_loglikelihood."""
    variances = np.ones_like(sample_residuals) * 0.0001
    nu = 5.0
    loglik = compute_student_loglikelihood(sample_residuals, variances, nu)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


def test_compute_skewt_loglikelihood(sample_residuals: np.ndarray) -> None:
    """Test compute_skewt_loglikelihood."""
    variances = np.ones_like(sample_residuals) * 0.0001
    nu = 5.0
    lambda_skew = 0.1
    loglik = compute_skewt_loglikelihood(sample_residuals, variances, nu, lambda_skew)
    assert np.isfinite(loglik)
    assert isinstance(loglik, float)


def test_compute_normal_loglikelihood_overflow() -> None:
    """Test compute_normal_loglikelihood handles overflow."""
    residuals = np.array([1e308, -1e308])
    variances = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="Overflow"):
        compute_normal_loglikelihood(residuals, variances)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
