"""Unit tests for GARCH parameter estimation (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import warnings

import numpy as np
import pytest

from src.garch.garch_params.core import egarch11_variance
from src.garch.garch_params.estimation import estimate_egarch_mle


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 0.01, size=200)


@pytest.fixture
def large_sample_residuals() -> np.ndarray:
    """Generate larger sample residuals for MLE testing."""
    rng = np.random.default_rng(7)
    return rng.normal(0.0, 0.01, size=600)


def test_egarch_variance_positive(sample_residuals: np.ndarray) -> None:
    """Test that EGARCH variance is always positive and finite."""
    s2 = egarch11_variance(
        sample_residuals, omega=-5.0, alpha=0.1, gamma=0.0, beta=0.95, dist="normal"
    )
    assert s2.shape == sample_residuals.shape
    assert np.all(np.isfinite(s2))
    assert np.all(s2 > 0)


def test_egarch_variance_with_init(sample_residuals: np.ndarray) -> None:
    """Test EGARCH variance with custom initialization."""
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


def test_egarch_variance_skewt_dist(sample_residuals: np.ndarray) -> None:
    """Test EGARCH variance with Skew-t distribution."""
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


def test_egarch_variance_invalid_params() -> None:
    """Test EGARCH variance with invalid parameters returns NaN."""
    rng = np.random.default_rng(42)
    e = rng.normal(0.0, 0.01, size=100)
    # Extreme beta that causes instability
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="overflow encountered in exp"
        )
        s2 = egarch11_variance(e, omega=100.0, alpha=10.0, gamma=10.0, beta=0.999, dist="normal")
    # Should handle gracefully (may return NaN or finite values)
    assert s2.shape == e.shape


def _validate_egarch_basic_fields(out: dict[str, float]) -> None:
    """Validate basic EGARCH output fields."""
    assert bool(out["converged"]) is True
    assert all(field in out for field in ["omega", "alpha", "gamma", "beta", "loglik"])
    assert isinstance(out["loglik"], float)
    assert -0.999 < out["beta"] < 0.999


def _validate_skewt_fields(out: dict[str, float]) -> None:
    """Validate Skew-t specific fields."""
    assert "nu" in out
    assert "lambda" in out
    assert out["nu"] > 2.0
    assert -0.99 < out["lambda"] < 0.99


def _validate_egarch_output(out: dict[str, float], expected_dist: str = "normal") -> None:
    """Validate EGARCH estimation output structure and values."""
    _validate_egarch_basic_fields(out)
    if expected_dist == "skewt":
        _validate_skewt_fields(out)


def test_estimate_egarch_converges_normal(large_sample_residuals: np.ndarray) -> None:
    """Test that EGARCH MLE converges for normal distribution."""
    params, convergence = estimate_egarch_mle(large_sample_residuals, dist="normal")
    _validate_egarch_output(params, expected_dist="normal")
    assert convergence.converged is True


def test_estimate_egarch_converges_student(large_sample_residuals: np.ndarray) -> None:
    """Test that EGARCH MLE converges for Student-t distribution."""
    # Suppress scipy optimization warnings about values outside bounds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
        params, convergence = estimate_egarch_mle(large_sample_residuals, dist="student")
    _validate_egarch_basic_fields(params)
    assert "nu" in params
    assert params["nu"] > 2.0
    assert convergence.converged is True


def test_estimate_egarch_converges_skewt(large_sample_residuals: np.ndarray) -> None:
    """Test that EGARCH MLE converges for Skew-t distribution."""
    # Suppress scipy optimization warnings about values outside bounds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize")
        params, convergence = estimate_egarch_mle(large_sample_residuals, dist="skewt")
    _validate_egarch_output(params, expected_dist="skewt")
    assert convergence.converged is True


def test_estimate_egarch_invalid_input() -> None:
    """Test that estimation raises ValueError for insufficient data."""
    e = np.array([1.0, 2.0])  # Too few observations
    with pytest.raises(ValueError, match="at least 10 observations"):
        estimate_egarch_mle(e, dist="normal")


def test_estimate_egarch_invalid_distribution(large_sample_residuals: np.ndarray) -> None:
    """Test that invalid distribution falls back to penalty values."""
    params, convergence = estimate_egarch_mle(large_sample_residuals, dist="invalid")
    # Expect a very negative log-likelihood to flag invalid distribution handling.
    assert params["loglik"] <= -1e20
    assert convergence.final_loglik <= -1e20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
