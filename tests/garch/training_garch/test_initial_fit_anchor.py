"""Tests for initial fit anchoring in EGARCHForecaster.

These tests ensure that, even when the forecaster is configured with a
rolling window for refits, the expanding forecast on TRAIN anchors the
initial fit and scheduling start at ``initial_window_size`` (not the
rolling ``window_size``). This matches the training setup and avoids
instability when early windows are too short or unrepresentative.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.garch.garch_params.estimation import ConvergenceResult
from src.garch.training_garch.forecaster import EGARCHForecaster


def test_expanding_on_train_anchors_at_initial_window_even_if_rolling() -> None:
    """Expanding forecast uses initial_window_size as anchor with rolling refits.

    The initial fit position and the refit scheduling anchor must be
    ``initial_window_size`` regardless of the configured refit window type.
    This guarantees that the first forecasts appear from ``initial_window_size``
    onward, not from ``window_size``.
    """
    rng = np.random.default_rng(123)
    n = 80
    residuals = rng.normal(0.0, 0.01, size=n)

    initial_window = 50
    window_size = 10  # rolling window size used by RefitManager

    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=10_000_000,  # avoid extra refits beyond initial
        window_type="rolling",
        window_size=window_size,
        initial_window_size=initial_window,
    )

    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    # Mock perform_refit to avoid heavy MLE and ensure quick, deterministic run
    dummy_params = {"omega": -5.0, "alpha": 0.1, "gamma": 0.0, "beta": 0.95, "loglik": -100.0}
    dummy_conv = ConvergenceResult(
        converged=True,
        n_iterations=10,
        final_loglik=-100.0,
        message="OK",
    )

    with patch.object(
        forecaster.refit_manager, "perform_refit", return_value=(dummy_params, dummy_conv)
    ):
        result = forecaster.forecast_expanding(residuals, dates=dates)

    # Forecasts should start at initial_window, regardless of rolling window size
    assert np.isnan(result.forecasts[:initial_window]).all()
    assert np.isfinite(result.forecasts[initial_window:]).all()

    # params_history length equals number of forecasts produced
    expected_len = n - initial_window
    assert len(result.params_history) == expected_len


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
