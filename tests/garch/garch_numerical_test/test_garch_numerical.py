"""Unit tests for GARCH numerical tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None

from src.garch.garch_numerical_test.garch_numerical import (
    breusch_pagan_test,
    engle_arch_lm_test,
    ljung_box_squared_test,
    ljung_box_test,
    mcleod_li_test,
    run_all_tests,
    white_test,
)
from src.garch.garch_numerical_test.utils import (
    validate_all_tests_result,
    validate_arch_lm_result,
    validate_ljung_box_result,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixture for random number generator with fixed seed."""
    return np.random.default_rng(42)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_ljung_box_test(rng: np.random.Generator) -> None:
    """Test Ljung-Box test on residuals."""
    white_noise = rng.standard_normal(100)
    result = ljung_box_test(white_noise, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_ljung_box_squared_test(rng: np.random.Generator) -> None:
    """Test Ljung-Box test on squared residuals."""
    residuals = rng.standard_normal(100)
    result = ljung_box_squared_test(residuals, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


def test_engle_arch_lm_test(rng: np.random.Generator) -> None:
    """Test Engle ARCH-LM test."""
    residuals = rng.standard_normal(100)
    result = engle_arch_lm_test(residuals, lags=5, alpha=0.05)
    validate_arch_lm_result(result, 5)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_mcleod_li_test(rng: np.random.Generator) -> None:
    """Test McLeod-Li test."""
    residuals = rng.standard_normal(100)
    result = mcleod_li_test(residuals, lags=5, alpha=0.05)
    validate_ljung_box_result(result, 5)


def test_breusch_pagan_test(rng: np.random.Generator) -> None:
    """Test Breusch-Pagan heteroskedasticity test."""
    residuals = rng.standard_normal(100)
    result = breusch_pagan_test(residuals, alpha=0.05)
    assert "lm_stat" in result
    assert "p_value" in result
    assert "df" in result
    assert "reject" in result
    assert "n" in result
    assert result["n"] == 100
    assert isinstance(result["reject"], bool)


def test_white_test(rng: np.random.Generator) -> None:
    """Test White heteroskedasticity test."""
    residuals = rng.standard_normal(100)
    result = white_test(residuals, alpha=0.05)
    assert "lm_stat" in result
    assert "p_value" in result
    assert "df" in result
    assert "reject" in result
    assert "n" in result
    assert result["n"] == 100
    assert isinstance(result["reject"], bool)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_run_all_tests(rng: np.random.Generator) -> None:
    """Test running all tests together."""
    residuals = rng.standard_normal(100)
    results = run_all_tests(residuals, ljung_box_lags=5, arch_lm_lags=5, alpha=0.05)
    validate_all_tests_result(results)
    assert "breusch_pagan" in results
    assert "white" in results


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_run_all_tests_without_heteroskedasticity(rng: np.random.Generator) -> None:
    """Test running all tests without heteroskedasticity diagnostics."""
    residuals = rng.standard_normal(120)
    results = run_all_tests(
        residuals,
        ljung_box_lags=4,
        arch_lm_lags=3,
        alpha=0.1,
        include_heteroskedasticity=False,
    )
    validate_all_tests_result(results)
    assert "breusch_pagan" not in results
    assert "white" not in results


def test_empty_residuals_raise_value_error() -> None:
    """Test handling of empty residuals."""
    empty = np.array([])
    with pytest.raises(ValueError, match="Residuals array is empty"):
        ljung_box_test(empty, lags=5)


def test_invalid_lags_raise_value_error(rng: np.random.Generator) -> None:
    """Test handling of invalid lag specification."""
    residuals = rng.standard_normal(50)
    with pytest.raises(ValueError, match="Invalid lags"):
        ljung_box_test(residuals, lags=0)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels is required for Ljung-Box diagnostics")
def test_nan_handling(rng: np.random.Generator) -> None:
    """Test handling of NaN values."""
    residuals = rng.standard_normal(100)
    residuals[10:20] = np.nan
    result = ljung_box_test(residuals, lags=5)
    assert result["n"] == 90  # Should exclude NaNs


def test_breusch_pagan_insufficient_observations() -> None:
    """Test Breusch-Pagan with insufficient observations."""
    residuals = np.array([1.0, 2.0])  # n=2 < 3
    result = breusch_pagan_test(residuals, alpha=0.05)
    assert np.isnan(result["lm_stat"])
    assert np.isnan(result["p_value"])
    assert result["reject"] is False
    assert result["n"] == 2


def test_white_test_insufficient_observations() -> None:
    """Test White test with insufficient observations."""
    residuals = np.array([1.0, 2.0, 3.0, 4.0])  # n=4 < 5
    result = white_test(residuals, alpha=0.05)
    assert np.isnan(result["lm_stat"])
    assert np.isnan(result["p_value"])
    assert result["reject"] is False
    assert result["n"] == 4


def test_arch_lm_insufficient_observations() -> None:
    """Test ARCH-LM with n <= lags."""
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # n=5
    with pytest.raises(ValueError, match="Insufficient data"):
        engle_arch_lm_test(residuals, lags=5, alpha=0.05)  # n <= lags


def test_breusch_pagan_with_exog(rng: np.random.Generator) -> None:
    """Test Breusch-Pagan with custom exogenous variables."""
    residuals = rng.standard_normal(100)
    exog = rng.standard_normal((100, 2))  # 2 custom regressors
    result = breusch_pagan_test(residuals, exog=exog, alpha=0.05)
    assert "lm_stat" in result
    assert "p_value" in result
    assert result["n"] == 100
    # df should be 2 (number of exog vars, intercept added automatically)
    assert result["df"] == 2.0


def test_constant_residuals() -> None:
    """Test with constant residuals (zero variance)."""
    residuals = np.ones(100)  # All same value
    result = breusch_pagan_test(residuals, alpha=0.05)
    # Should handle gracefully (R²=0 → LM=0)
    assert result["lm_stat"] >= 0.0
    assert result["n"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
