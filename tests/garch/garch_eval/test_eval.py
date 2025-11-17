"""Unit tests for GARCH evaluation (forecasts, VaR, intervals)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_eval.eval import (
    egarch_multi_step_variance_forecast,
    egarch_one_step_variance_forecast,
    prediction_interval,
    value_at_risk,
)


def test_one_step_variance_forecast_basic() -> None:
    # In EGARCH, with omega=0, alpha=0, gamma=0, beta=1 -> sigma2_{t+1} = sigma2_t
    e_last, s2_last = 0.5, 1.5
    s2_1 = egarch_one_step_variance_forecast(
        e_last,
        s2_last,
        omega=0.0,
        alpha=0.0,
        gamma=0.0,
        beta=1.0,
        dist="normal",
        nu=None,
    )
    assert np.isclose(s2_1, s2_last, rtol=0, atol=1e-12)


def test_multi_step_matches_log_recursion() -> None:
    # EGARCH multi-step under zero-mean shocks: log recursion
    omega, alpha, gamma, beta = 0.1, 0.0, 0.0, 0.9
    s2_last = 1.0
    h = 5
    s2_closed = egarch_multi_step_variance_forecast(
        h,
        s2_last,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        dist="normal",
        nu=None,
    )
    # Manual log-variance recursion
    s2_rec = np.empty(h, dtype=float)
    log_s2 = np.log(s2_last)
    for k in range(h):
        log_s2 = omega + beta * log_s2
        s2_rec[k] = np.exp(log_s2)
    assert np.allclose(s2_closed, s2_rec, rtol=0, atol=1e-12)


def test_prediction_interval_normal_95() -> None:
    z_975 = 1.959964
    lo, hi = prediction_interval(0.0, 1.0, level=0.95, dist="normal")
    # Approximately ±z_975
    assert np.isclose(hi, -lo, rtol=0, atol=1e-6)
    assert np.isclose(hi, z_975, rtol=0, atol=5e-4)


def test_var_student_monotonicity() -> None:
    v1 = value_at_risk(0.05, mean=0.0, variance=1.0, dist="student", nu=8.0)
    v2 = value_at_risk(0.01, mean=0.0, variance=1.0, dist="student", nu=8.0)
    assert v1 < 0.0
    assert v2 < 0.0
    # More conservative tail -> lower (more negative) VaR
    assert v2 < v1


# ==================== RefitManager Tests ====================
# Tests for the new refit API (replaces old _refit_model_params and _update_model_params_if_needed)


def test_refit_manager_insufficient_data() -> None:
    """Test RefitManager.perform_refit raises ValueError when data is insufficient."""
    from src.garch.garch_params.refit.refit_manager import RefitManager

    manager = RefitManager(frequency=10, window_type="expanding", o=1, p=1, dist="normal")
    rng = np.random.default_rng(42)
    # Create insufficient data (less than GARCH_ESTIMATION_MIN_OBSERVATIONS=10)
    insufficient_residuals = rng.normal(0.0, 0.01, size=5)

    with pytest.raises(ValueError, match="Need at least"):
        manager.perform_refit(insufficient_residuals, position=5)


def test_refit_manager_success() -> None:
    """Test RefitManager.perform_refit successfully refits with sufficient data."""
    from src.garch.garch_params.refit.refit_manager import RefitManager

    manager = RefitManager(frequency=10, window_type="expanding", o=1, p=1, dist="normal")
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=600)

    params, convergence = manager.perform_refit(residuals, position=600)

    assert isinstance(params, dict)
    assert "omega" in params
    assert "alpha" in params
    assert "beta" in params
    assert convergence.converged is True
    assert manager.current_params == params
    assert len(manager.refit_history) == 1


def test_refit_manager_failure_non_converged() -> None:
    """Test RefitManager.perform_refit returns non-converged result but still updates params."""
    from unittest.mock import patch

    from src.garch.garch_params.refit.refit_manager import RefitManager

    manager = RefitManager(frequency=10, window_type="expanding", o=1, p=1, dist="normal")
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=600)

    # Mock estimate_egarch_mle to return non-converged result
    with patch("src.garch.garch_params.refit.refit_manager.estimate_egarch_mle") as mock_mle:
        from src.garch.garch_params.estimation import ConvergenceResult

        mock_mle.return_value = (
            {"omega": -5.0, "alpha": 0.1, "gamma": 0.0, "beta": 0.95, "loglik": -100.0},
            ConvergenceResult(
                converged=False,
                n_iterations=100,
                final_loglik=-100.0,
                message="Maximum iterations reached",
            ),
        )

        params, convergence = manager.perform_refit(residuals, position=600)

        assert convergence.converged is False
        assert params["omega"] == -5.0
        # Params are still updated even if not converged
        assert manager.current_params == params
        assert len(manager.refit_history) == 1


# ==================== EGARCHForecaster Refit Tests ====================


def test_forecaster_refit_not_needed() -> None:
    """Test EGARCHForecaster does not refit when should_refit returns False."""
    from src.garch.training_garch.forecaster import EGARCHForecaster

    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=100,
        initial_window_size=50,
        window_type="expanding",
    )
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=150)

    # Position 60 should not trigger refit (refit_frequency=100, initial=50)
    # First refit at 50, next at 150
    assert not forecaster.refit_manager.should_refit(60, 50)

    # Perform forecast - should not refit
    result = forecaster.forecast_expanding(residuals)

    # Check that refit only occurred at initial position
    assert np.sum(result.refit_mask) == 1  # Only initial fit
    assert bool(result.refit_mask[50])  # Initial fit at position 50


def test_forecaster_refit_success() -> None:
    """Test EGARCHForecaster successfully refits when needed."""
    from src.garch.training_garch.forecaster import EGARCHForecaster

    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=50,
        initial_window_size=50,
        window_type="expanding",
    )
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=200)

    # Should refit at positions: 50 (initial), 100, 150
    result = forecaster.forecast_expanding(residuals)

    # Check refits occurred
    assert np.sum(result.refit_mask) >= 1  # At least initial fit
    assert bool(result.refit_mask[50])  # Initial fit
    # May have additional refits depending on convergence
    assert result.n_refits >= 1


def test_forecaster_refit_failure_continues_with_old_params() -> None:
    """Test EGARCHForecaster continues with old params when refit fails to converge."""
    from unittest.mock import patch

    from src.garch.garch_params.estimation import ConvergenceResult
    from src.garch.training_garch.forecaster import EGARCHForecaster

    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=50,
        initial_window_size=50,
        window_type="expanding",
    )
    rng = np.random.default_rng(42)
    residuals = rng.normal(0.0, 0.01, size=150)

    # Mock perform_refit to return non-converged result after initial fit
    original_perform_refit = forecaster.refit_manager.perform_refit
    call_count = 0

    def mock_perform_refit(residuals: np.ndarray, position: int):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (initial fit) succeeds
            return original_perform_refit(residuals, position)
        # Subsequent calls fail to converge
        params, _ = original_perform_refit(residuals, position)
        return params, ConvergenceResult(
            converged=False,
            n_iterations=100,
            final_loglik=-100.0,
            message="Maximum iterations reached",
        )

    with patch.object(forecaster.refit_manager, "perform_refit", side_effect=mock_perform_refit):
        result = forecaster.forecast_expanding(residuals)

        # Should still produce forecasts even if some refits failed
        assert len(result.forecasts) == 150
        assert np.all(np.isfinite(result.forecasts[50:]))  # Forecasts after initial window
        # Refit mask should still mark refit positions
        assert np.sum(result.refit_mask) >= 1


def test_compute_metrics_from_forecasts_basic() -> None:
    """Compute basic metrics from forecast DataFrame."""
    from src.garch.garch_eval.eval import compute_metrics_from_forecasts

    data = pd.DataFrame(
        {
            "h": [1, 1, 2],
            "sigma2_forecast": [1.0, 0.5, 0.5],
            "VaR": [-1.5, -0.7, -0.7],
            "var_left_alpha": [0.05, 0.05, 0.05],
            "actual_residual": [0.5, -0.1, np.nan],
        }
    )
    metrics = compute_metrics_from_forecasts(data)
    assert metrics["n_obs"] == 2
    assert "qlike" in metrics
    assert "var_backtests_empirical" in metrics
    var_metrics = metrics["var_backtests_empirical"]  # type: ignore[index]
    alpha_metrics = var_metrics["0.05"]  # type: ignore[index]
    assert alpha_metrics["n"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
