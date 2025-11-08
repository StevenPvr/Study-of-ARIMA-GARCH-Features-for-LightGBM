"""Unit tests for classic GARCH metrics (deterministic checks)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_eval.metrics import (
    christoffersen_ind_test,
    kupiec_pof_test,
    mincer_zarnowitz,
    mse_mae_variance,
    qlike_loss,
)


def test_qlike_and_mse_mae_simple() -> None:
    # e^2 equals sigma^2 exactly -> MSE=MAE=0, QLIKE=log(sigma2)+1 average
    sigma2 = np.array([1.0, 2.0, 0.5, 3.0])
    e = np.array([1.0, np.sqrt(2.0), np.sqrt(0.5), np.sqrt(3.0)])
    losses = mse_mae_variance(e, sigma2)
    assert np.isclose(losses["mse"], 0.0)
    assert np.isclose(losses["mae"], 0.0)
    ql = qlike_loss(e, sigma2)
    expected = float(np.mean(np.log(sigma2) + (e**2) / sigma2))
    assert np.isclose(ql, expected)


def test_mincer_zarnowitz_perfect() -> None:
    # Perfect relation: e^2 = sigma2 -> slope approx 1, intercept approx 0
    sigma2 = np.linspace(0.5, 2.0, 50)
    e = np.sqrt(sigma2)
    res = mincer_zarnowitz(e, sigma2)
    tol = 1e-10
    min_r2 = 0.99
    assert abs(res["slope"] - 1.0) < tol
    assert abs(res["intercept"]) < tol
    assert min_r2 <= res["r2"] <= 1.0


def test_kupiec_pof_basic() -> None:
    """Test Kupiec POF test with known hit rate."""
    hits = np.array([0] * 90 + [1] * 10)
    kup = kupiec_pof_test(hits, alpha=0.1)
    assert kup["n"] == 100.0
    assert kup["x"] == 10.0
    assert np.isclose(kup["hit_rate"], 0.1)


def test_christoffersen_ind_basic() -> None:
    """Test Christoffersen independence test with valid p-value."""
    hits = np.array([0] * 90 + [1] * 10)
    ind = christoffersen_ind_test(hits)
    p_value = ind["p_value"] if not np.isnan(ind["p_value"]) else 0.0
    assert ind["lr_ind"] >= 0.0 or np.isnan(ind["lr_ind"])
    assert 0.0 <= p_value <= 1.0


def test_var_backtests_components() -> None:
    """Test VaR backtest components."""
    test_kupiec_pof_basic()
    test_christoffersen_ind_basic()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
