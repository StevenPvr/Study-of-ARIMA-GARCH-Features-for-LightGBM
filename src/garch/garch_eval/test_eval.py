"""Unit tests for GARCH evaluation (forecasts, VaR, intervals)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
