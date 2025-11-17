"""Tests for composite objective function.

Tests composite objective combining QLIKE, AIC, and diagnostic penalties.
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

from src.constants import (
    GARCH_OPTIMIZATION_AIC_WEIGHT,
    GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT,
    GARCH_OPTIMIZATION_QLIKE_WEIGHT,
)
from src.garch.garch_params.optimization.diagnostics import _count_model_parameters, compute_composite_objective


class TestCountModelParameters:
    """Test suite for parameter counting."""

    def test_count_params_egarch11_normal(self) -> None:
        """Test parameter count for EGARCH(1,1) with normal distribution."""
        n_params = _count_model_parameters(o=1, p=1, dist="normal")
        # omega (1) + alpha (1) + gamma (1) + beta (1) = 4
        assert n_params == 4

    def test_count_params_egarch11_student(self) -> None:
        """Test parameter count for EGARCH(1,1) with Student-t distribution."""
        n_params = _count_model_parameters(o=1, p=1, dist="student")
        # omega (1) + alpha (1) + gamma (1) + beta (1) + nu (1) = 5
        assert n_params == 5

    def test_count_params_egarch11_skewt(self) -> None:
        """Test parameter count for EGARCH(1,1) with Skew-t distribution."""
        n_params = _count_model_parameters(o=1, p=1, dist="skewt")
        # omega (1) + alpha (1) + gamma (1) + beta (1) + nu (1) + lambda (1) = 6
        assert n_params == 6

    def test_count_params_egarch22_normal(self) -> None:
        """Test parameter count for EGARCH(2,2) with normal distribution."""
        n_params = _count_model_parameters(o=2, p=2, dist="normal")
        # omega (1) + alpha (2) + gamma (2) + beta (2) = 7
        assert n_params == 7

    def test_count_params_egarch33_skewt(self) -> None:
        """Test parameter count for EGARCH(3,3) with Skew-t distribution."""
        n_params = _count_model_parameters(o=3, p=3, dist="skewt")
        # omega (1) + alpha (3) + gamma (3) + beta (3) + nu (1) + lambda (1) = 12
        assert n_params == 12


class TestCompositeObjective:
    """Test suite for composite objective function."""

    def test_compute_composite_objective_basic(self) -> None:
        """Test basic composite objective computation."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        qlike = 2.5
        loglik = -250.0
        params = {
            "omega": 0.1,
            "alpha": 0.05,
            "gamma": 0.0,
            "beta": 0.9,
        }

        composite, components = compute_composite_objective(
            qlike=qlike,
            residuals=residuals,
            params=params,
            loglik=loglik,
            o=1,
            p=1,
            dist="normal",
        )

        # Check that composite is a weighted sum
        assert isinstance(composite, float)
        assert isinstance(components, dict)

        # Check components exist
        assert "qlike" in components
        assert "aic" in components
        assert "aic_raw" in components
        assert "diagnostic" in components
        assert "composite" in components

        # Check composite matches expected formula
        expected_composite = (
            GARCH_OPTIMIZATION_QLIKE_WEIGHT * components["qlike"]
            + GARCH_OPTIMIZATION_AIC_WEIGHT * components["aic"]
            + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT * components["diagnostic"]
        )
        assert abs(composite - expected_composite) < 1e-10

    def test_compute_composite_objective_components_normalized(self) -> None:
        """Test that all components are normalized to [0,1] range."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        qlike = 3.0
        loglik = -300.0
        params = {
            "omega": 0.1,
            "alpha": 0.05,
            "gamma": 0.0,
            "beta": 0.9,
        }

        _, components = compute_composite_objective(
            qlike=qlike,
            residuals=residuals,
            params=params,
            loglik=loglik,
            o=1,
            p=1,
            dist="normal",
        )

        # QLIKE should be passed through (assumed already normalized)
        assert components["qlike"] == qlike

        # AIC should be normalized to [0, 1]
        assert 0.0 <= components["aic"] <= 1.0

        # Diagnostic penalty should be in [0, 1]
        assert 0.0 <= components["diagnostic"] <= 1.0

    def test_compute_composite_objective_weights_sum_to_one(self) -> None:
        """Test that weights sum to 1.0."""
        total_weight = (
            GARCH_OPTIMIZATION_QLIKE_WEIGHT
            + GARCH_OPTIMIZATION_AIC_WEIGHT
            + GARCH_OPTIMIZATION_DIAGNOSTIC_WEIGHT
        )
        assert abs(total_weight - 1.0) < 1e-10

    def test_compute_composite_objective_different_distributions(self) -> None:
        """Test composite objective with different distributions."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        qlike = 2.5
        loglik = -250.0

        params_normal = {"omega": 0.1, "alpha": 0.05, "gamma": 0.0, "beta": 0.9}
        params_student = {"omega": 0.1, "alpha": 0.05, "gamma": 0.0, "beta": 0.9, "nu": 8.0}
        params_skewt = {
            "omega": 0.1,
            "alpha": 0.05,
            "gamma": 0.0,
            "beta": 0.9,
            "nu": 8.0,
            "lambda": -0.1,
        }

        comp_normal, comps_normal = compute_composite_objective(
            qlike, residuals, params_normal, loglik, o=1, p=1, dist="normal"
        )
        comp_student, comps_student = compute_composite_objective(
            qlike, residuals, params_student, loglik, o=1, p=1, dist="student"
        )
        comp_skewt, comps_skewt = compute_composite_objective(
            qlike, residuals, params_skewt, loglik, o=1, p=1, dist="skewt"
        )

        # All should produce valid results
        assert isinstance(comp_normal, float)
        assert isinstance(comp_student, float)
        assert isinstance(comp_skewt, float)

        # Student-t and Skew-t should have higher AIC due to more parameters
        # (assuming same log-likelihood)
        assert comps_student["aic_raw"] > comps_normal["aic_raw"]
        assert comps_skewt["aic_raw"] > comps_student["aic_raw"]

    def test_compute_composite_objective_low_qlike_preferred(self) -> None:
        """Test that lower QLIKE leads to lower composite objective."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        loglik = -250.0
        params = {"omega": 0.1, "alpha": 0.05, "gamma": 0.0, "beta": 0.9}

        comp_low_qlike, _ = compute_composite_objective(
            qlike=2.0, residuals=residuals, params=params, loglik=loglik, o=1, p=1, dist="normal"
        )
        comp_high_qlike, _ = compute_composite_objective(
            qlike=4.0, residuals=residuals, params=params, loglik=loglik, o=1, p=1, dist="normal"
        )

        # Lower QLIKE should lead to lower composite objective (better)
        assert comp_low_qlike < comp_high_qlike

    def test_compute_composite_objective_better_loglik_preferred(self) -> None:
        """Test that better log-likelihood leads to lower composite objective."""
        np.random.seed(42)
        residuals = np.random.randn(500)
        qlike = 2.5
        params = {"omega": 0.1, "alpha": 0.05, "gamma": 0.0, "beta": 0.9}

        comp_poor_loglik, _ = compute_composite_objective(
            qlike=qlike, residuals=residuals, params=params, loglik=-300.0, o=1, p=1, dist="normal"
        )
        comp_good_loglik, _ = compute_composite_objective(
            qlike=qlike, residuals=residuals, params=params, loglik=-200.0, o=1, p=1, dist="normal"
        )

        # Better log-likelihood should lead to lower composite objective (better)
        assert comp_good_loglik < comp_poor_loglik


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
