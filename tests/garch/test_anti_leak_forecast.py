from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent.parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from src.garch.training_garch.forecaster import EGARCHForecaster


def test_forecast_at_t_does_not_use_e_t() -> None:
    """Ensure one-step forecast at t is independent of residual e_t.

    Why: Anti-leakage guarantee requires that σ²_{t+1}|t depends only on
    residuals up to (not including) position t. Changing e_t must not affect
    the forecast at index t.
    """
    # Deterministic residuals
    n = 40
    residuals = np.linspace(-0.5, 0.5, num=n).astype(float)

    # Minimal window settings to start forecasting early in the series
    min_window = 10
    initial_window = 10

    # Fixed, stable EGARCH(1,1) parameters (normal dist)
    fixed_params = {
        "omega": 0.01,
        "alpha": 0.05,
        "gamma": 0.0,
        "beta": 0.9,
    }

    # Build forecaster with fixed parameters (no refits)
    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=5,
        window_type="expanding",
        window_size=None,
        initial_window_size=initial_window,
        min_window_size=min_window,
        use_fixed_params=True,
    )
    forecaster.refit_manager.current_params = dict(fixed_params)

    # Baseline forecast
    result = forecaster.forecast_expanding(residuals)
    t = min_window  # first forecasted position
    s2_t_baseline = float(result.forecasts[t])

    # Shock only e_t and recompute (should not change forecast at index t)
    residuals_shocked = residuals.copy()
    residuals_shocked[t] = residuals_shocked[t] + 10.0

    result_shocked = forecaster.forecast_expanding(residuals_shocked)
    s2_t_shocked = float(result_shocked.forecasts[t])

    assert np.isfinite(s2_t_baseline)
    assert np.isfinite(s2_t_shocked)
    assert s2_t_baseline == s2_t_shocked, (
        "Forecast at t must not depend on e_t (anti-leakage violated)"
    )

if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
