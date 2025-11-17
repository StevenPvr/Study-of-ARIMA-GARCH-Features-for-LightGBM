"""Unit tests for GARCH estimation modules (initialization, convergence, mle)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_params.estimation.common import (
    ConvergenceResult,
    ConvergenceTracker,
    extract_convergence_info,
)
from src.garch.garch_params.estimation.initialization import (
    build_bounds,
    build_initial_params,
    build_result_dict,
    compute_initial_omega,
    count_params,
    extract_params_from_array,
    get_default_params,
)

# ==================== Initialization Tests ====================


def test_compute_initial_omega() -> None:
    """Test compute_initial_omega."""
    variance_estimate = 0.0001
    omega = compute_initial_omega(variance_estimate)
    assert np.isfinite(omega)
    assert isinstance(omega, float)


def test_get_default_params() -> None:
    """Test get_default_params."""
    params = get_default_params()
    assert "beta" in params
    assert "alpha" in params
    assert "gamma" in params
    assert "nu" in params
    assert "lambda" in params


def test_build_bounds_normal_11() -> None:
    """Test build_bounds for normal, o=1, p=1."""
    bounds = build_bounds(o=1, p=1, dist="normal")
    assert len(bounds) == 4  # omega, alpha, gamma, beta
    assert all(isinstance(b, tuple) and len(b) == 2 for b in bounds)


def test_build_bounds_student_11() -> None:
    """Test build_bounds for student, o=1, p=1."""
    bounds = build_bounds(o=1, p=1, dist="student")
    assert len(bounds) == 5  # omega, alpha, gamma, beta, nu


def test_build_bounds_skewt_11() -> None:
    """Test build_bounds for skewt, o=1, p=1."""
    bounds = build_bounds(o=1, p=1, dist="skewt")
    assert len(bounds) == 6  # omega, alpha, gamma, beta, nu, lambda


def test_build_bounds_12() -> None:
    """Test build_bounds for o=1, p=2."""
    bounds = build_bounds(o=1, p=2, dist="normal")
    assert len(bounds) == 5  # omega, alpha, gamma, beta1, beta2


def test_build_bounds_21() -> None:
    """Test build_bounds for o=2, p=1."""
    bounds = build_bounds(o=2, p=1, dist="normal")
    assert len(bounds) == 6  # omega, alpha1, alpha2, gamma1, gamma2, beta


def test_build_bounds_22() -> None:
    """Test build_bounds for o=2, p=2."""
    bounds = build_bounds(o=2, p=2, dist="normal")
    assert len(bounds) == 7  # omega, alpha1, alpha2, gamma1, gamma2, beta1, beta2


def test_count_params_normal_11() -> None:
    """Test count_params for normal, o=1, p=1."""
    n = count_params(o=1, p=1, dist="normal")
    assert n == 4


def test_count_params_student_11() -> None:
    """Test count_params for student, o=1, p=1."""
    n = count_params(o=1, p=1, dist="student")
    assert n == 5


def test_count_params_skewt_11() -> None:
    """Test count_params for skewt, o=1, p=1."""
    n = count_params(o=1, p=1, dist="skewt")
    assert n == 6


def test_build_initial_params_normal_11() -> None:
    """Test build_initial_params for normal, o=1, p=1."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=200)
    variance_estimate = float(np.var(residuals))
    omega_init = compute_initial_omega(variance_estimate)
    params = build_initial_params(omega_init, o=1, p=1, dist="normal")
    assert len(params) == 4
    assert all(np.isfinite(p) for p in params)


def test_build_initial_params_student_11() -> None:
    """Test build_initial_params for student, o=1, p=1."""
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=200)
    variance_estimate = float(np.var(residuals))
    omega_init = compute_initial_omega(variance_estimate)
    params = build_initial_params(omega_init, o=1, p=1, dist="student")
    assert len(params) == 5


def test_extract_params_from_array_normal_11() -> None:
    """Test extract_params_from_array for normal, o=1, p=1."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95])
    omega, alpha, gamma, beta, nu, lambda_skew = extract_params_from_array(
        arr, o=1, p=1, dist="normal"
    )
    assert omega == -5.0
    assert alpha == 0.1
    assert gamma == 0.0
    assert beta == 0.95
    assert nu is None
    assert lambda_skew is None


def test_extract_params_from_array_student_11() -> None:
    """Test extract_params_from_array for student, o=1, p=1."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95, 5.0])
    omega, alpha, gamma, beta, nu, lambda_skew = extract_params_from_array(
        arr, o=1, p=1, dist="student"
    )
    assert omega == -5.0
    assert alpha == 0.1
    assert nu == 5.0
    assert lambda_skew is None


def test_build_result_dict_normal_11() -> None:
    """Test build_result_dict for normal, o=1, p=1."""
    arr = np.array([-5.0, 0.1, 0.0, 0.95])
    loglik = -100.0
    converged = True
    result = build_result_dict(arr, o=1, p=1, dist="normal", loglik=loglik, converged=converged)
    assert result["omega"] == -5.0
    assert result["alpha"] == 0.1
    assert result["beta"] == 0.95
    assert result["loglik"] == -100.0
    assert bool(result["converged"]) is True


# ==================== Convergence Tests ====================


def test_convergence_result() -> None:
    """Test ConvergenceResult creation."""
    result = ConvergenceResult(
        converged=True, n_iterations=10, final_loglik=-100.0, message="Success"
    )
    assert result.converged is True
    assert result.n_iterations == 10
    assert result.final_loglik == -100.0
    assert result.message == "Success"


def test_convergence_result_str() -> None:
    """Test ConvergenceResult string representation."""
    result = ConvergenceResult(
        converged=True, n_iterations=10, final_loglik=-100.0, message="Success"
    )
    s = str(result)
    assert "converged=True" in s
    assert "n_iter=10" in s
    assert "loglik=-100.00" in s


def test_convergence_tracker() -> None:
    """Test ConvergenceTracker."""
    tracker = ConvergenceTracker()
    result1 = ConvergenceResult(converged=True, n_iterations=10, final_loglik=-100.0)
    result2 = ConvergenceResult(converged=False, n_iterations=5, final_loglik=-200.0)

    tracker.add_result(result1)
    tracker.add_result(result2)

    assert len(tracker.results) == 2
    assert tracker.compute_convergence_rate() == 0.5


def test_convergence_tracker_from_optimizer() -> None:
    """Test ConvergenceTracker.add_from_optimizer_result."""
    tracker = ConvergenceTracker()
    mock_opt_result = MagicMock()
    mock_opt_result.success = True
    mock_opt_result.nit = 10
    mock_opt_result.fun = 100.0  # Negative log-likelihood
    mock_opt_result.message = "Optimization terminated successfully"

    tracker.add_from_optimizer_result(mock_opt_result)

    assert len(tracker.results) == 1
    assert tracker.results[0].converged is True
    assert tracker.results[0].n_iterations == 10
    assert tracker.results[0].final_loglik == -100.0


def test_extract_convergence_info() -> None:
    """Test extract_convergence_info."""
    mock_opt_result = MagicMock()
    mock_opt_result.success = True
    mock_opt_result.nit = 10
    mock_opt_result.fun = 100.0
    mock_opt_result.message = "Success"

    result = extract_convergence_info(mock_opt_result)

    assert result.converged is True
    assert result.n_iterations == 10
    assert result.final_loglik == -100.0


def test_convergence_tracker_summary() -> None:
    """Test ConvergenceTracker.get_summary."""
    tracker = ConvergenceTracker()
    result1 = ConvergenceResult(converged=True, n_iterations=10, final_loglik=-100.0)
    result2 = ConvergenceResult(converged=True, n_iterations=5, final_loglik=-200.0)
    result3 = ConvergenceResult(converged=False, n_iterations=3, final_loglik=-300.0)

    tracker.add_result(result1)
    tracker.add_result(result2)
    tracker.add_result(result3)

    summary = tracker.get_summary()
    assert summary["n_estimations"] == 3
    assert summary["n_converged"] == 2
    assert summary["convergence_rate"] == pytest.approx(2.0 / 3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
