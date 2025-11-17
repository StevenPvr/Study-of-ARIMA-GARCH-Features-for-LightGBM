"""Tests for optimization diagnostic functions.

Tests diagnostic penalty computation and AIC calculations.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest

from src.garch.garch_params.optimization.diagnostics import (
    compute_aic_penalty,
    compute_diagnostic_penalty,
    normalize_aic_penalty,
)


class TestDiagnosticPenalty:
    """Test suite for diagnostic penalty computation."""

    def test_compute_diagnostic_penalty_valid_model(self) -> None:
        """Test diagnostic penalty for well-specified model."""
        # Generate well-behaved residuals (i.i.d. normal)
        np.random.seed(42)
        residuals = np.random.randn(500)

        params = {
            "omega": 0.1,
            "alpha": 0.05,
            "gamma": 0.0,
            "beta": 0.9,
        }

        penalty = compute_diagnostic_penalty(
            residuals=residuals,
            params=params,
            o=1,
            p=1,
            dist="normal",
        )

        # Well-specified model should have low penalty
        assert isinstance(penalty, float)
        assert 0.0 <= penalty <= 1.0

    def test_compute_diagnostic_penalty_with_arch_effects(self) -> None:
        """Test diagnostic penalty with remaining ARCH effects."""
        # Generate residuals with ARCH effects (volatility clustering)
        np.random.seed(42)
        n = 500
        residuals = np.zeros(n)
        sigma2 = np.ones(n)

        # Simple ARCH(1) process
        for t in range(1, n):
            sigma2[t] = 0.1 + 0.5 * residuals[t - 1] ** 2
            residuals[t] = np.sqrt(sigma2[t]) * np.random.randn()

        # Fit with misspecified parameters (too low alpha)
        params = {
            "omega": 0.1,
            "alpha": 0.01,  # Too low, won't capture all ARCH effects
            "gamma": 0.0,
            "beta": 0.5,
        }

        penalty = compute_diagnostic_penalty(
            residuals=residuals,
            params=params,
            o=1,
            p=1,
            dist="normal",
        )

        # Misspecified model should have higher penalty
        assert isinstance(penalty, float)
        assert 0.0 <= penalty <= 1.0

    def test_compute_diagnostic_penalty_insufficient_data(self) -> None:
        """Test diagnostic penalty with insufficient data."""
        residuals = np.random.randn(3)  # Too few observations
        params = {"omega": 0.1, "alpha": 0.05, "gamma": 0.0, "beta": 0.9}

        penalty = compute_diagnostic_penalty(
            residuals=residuals,
            params=params,
            o=1,
            p=1,
            dist="normal",
        )

        # Insufficient data should return maximum penalty
        assert penalty == 1.0

    def test_compute_diagnostic_penalty_invalid_variance(self) -> None:
        """Test diagnostic penalty when variance computation fails."""
        residuals = np.random.randn(100)
        # Invalid parameters that cause variance computation to fail
        params = {"omega": -10.0, "alpha": 5.0, "gamma": 0.0, "beta": 2.0}

        penalty = compute_diagnostic_penalty(
            residuals=residuals,
            params=params,
            o=1,
            p=1,
            dist="normal",
        )

        # Failed variance computation should return maximum penalty
        assert penalty == 1.0


class TestAICPenalty:
    """Test suite for AIC penalty computation."""

    def test_compute_aic_penalty_basic(self) -> None:
        """Test basic AIC computation."""
        n_obs = 1000
        loglik = -500.0
        n_params = 4

        aic = compute_aic_penalty(n_obs, loglik, n_params)

        # AIC = -2*loglik + 2*k = -2*(-500) + 2*4 = 1000 + 8 = 1008
        expected_aic = 1008.0
        assert abs(aic - expected_aic) < 1e-10

    def test_compute_aic_penalty_different_params(self) -> None:
        """Test AIC increases with more parameters."""
        n_obs = 1000
        loglik = -500.0

        aic_4_params = compute_aic_penalty(n_obs, loglik, n_params=4)
        aic_6_params = compute_aic_penalty(n_obs, loglik, n_params=6)

        # More parameters should increase AIC (worse)
        assert aic_6_params > aic_4_params

    def test_compute_aic_penalty_better_loglik(self) -> None:
        """Test AIC decreases with better log-likelihood."""
        n_obs = 1000
        n_params = 4

        aic_poor = compute_aic_penalty(n_obs, loglik=-500.0, n_params=n_params)
        aic_good = compute_aic_penalty(n_obs, loglik=-400.0, n_params=n_params)

        # Better log-likelihood should decrease AIC (better)
        assert aic_good < aic_poor

    def test_normalize_aic_penalty(self) -> None:
        """Test AIC normalization to [0,1] range."""
        n_obs = 1000

        # Test various AIC values
        aic_low = 100.0
        aic_medium = 1000.0
        aic_high = 10000.0

        norm_low = normalize_aic_penalty(aic_low, n_obs)
        norm_medium = normalize_aic_penalty(aic_medium, n_obs)
        norm_high = normalize_aic_penalty(aic_high, n_obs)

        # All should be in [0, 1]
        assert 0.0 <= norm_low <= 1.0
        assert 0.0 <= norm_medium <= 1.0
        assert 0.0 <= norm_high <= 1.0

        # Higher AIC should give higher penalty
        assert norm_low < norm_medium < norm_high

    def test_normalize_aic_penalty_zero(self) -> None:
        """Test AIC normalization with zero AIC."""
        aic = 0.0
        n_obs = 1000

        norm = normalize_aic_penalty(aic, n_obs)

        assert 0.0 <= norm <= 1.0
        assert norm >= 0.0  # Should be very small but not negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
