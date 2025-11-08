"""Unit tests for GARCH numerical tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_numerical_test.garch_numerical import (
    engle_arch_lm_test,
    ljung_box_squared_test,
    ljung_box_test,
    mcleod_li_test,
    run_all_tests,
)


def _assert_keys_present(result: dict[str, Any], required_keys: list[str]) -> None:
    """Assert that all required keys are present in result.

    Args:
        result: Test result dictionary.
        required_keys: List of required key names.
    """
    for key in required_keys:
        assert key in result


def _validate_ljung_box_result(result: dict[str, Any], expected_lags: int) -> None:
    """Validate Ljung-Box test result structure.

    Args:
        result: Test result dictionary.
        expected_lags: Expected number of lags.
    """
    _assert_keys_present(result, ["lb_stat", "lb_pvalue", "reject"])
    assert len(result["lb_stat"]) == expected_lags
    assert len(result["lb_pvalue"]) == expected_lags


def _validate_arch_lm_result(result: dict[str, Any], expected_df: int) -> None:
    """Validate ARCH-LM test result structure.

    Args:
        result: Test result dictionary.
        expected_df: Expected degrees of freedom.
    """
    _assert_keys_present(result, ["lm_stat", "p_value", "df", "reject"])
    assert result["df"] == expected_df


def _validate_all_tests_result(results: dict[str, Any]) -> None:
    """Validate that all test results are present and valid.

    Args:
        results: Dictionary containing all test results.
    """
    required_keys = ["ljung_box_residuals", "ljung_box_squared", "engle_arch_lm", "mcleod_li"]
    for key in required_keys:
        assert key in results
        assert isinstance(results[key], dict)


def test_ljung_box_test() -> None:
    """Test Ljung-Box test on residuals."""
    rng = np.random.default_rng(42)
    white_noise = rng.standard_normal(100)
    result = ljung_box_test(white_noise, lags=5, alpha=0.05)
    _validate_ljung_box_result(result, 5)


def test_ljung_box_squared_test() -> None:
    """Test Ljung-Box test on squared residuals."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = ljung_box_squared_test(residuals, lags=5, alpha=0.05)
    _validate_ljung_box_result(result, 5)


def test_engle_arch_lm_test() -> None:
    """Test Engle ARCH-LM test."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = engle_arch_lm_test(residuals, lags=5, alpha=0.05)
    _validate_arch_lm_result(result, 5)


def test_mcleod_li_test() -> None:
    """Test McLeod-Li test."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    result = mcleod_li_test(residuals, lags=5, alpha=0.05)
    _validate_ljung_box_result(result, 5)


def test_run_all_tests() -> None:
    """Test running all tests together."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    results = run_all_tests(residuals, ljung_box_lags=5, arch_lm_lags=5, alpha=0.05)
    _validate_all_tests_result(results)


def test_empty_residuals() -> None:
    """Test handling of empty residuals."""
    empty = np.array([])
    result = ljung_box_test(empty, lags=5)
    assert result["n"] == 0
    assert not result["reject"]


def test_nan_handling() -> None:
    """Test handling of NaN values."""
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(100)
    residuals[10:20] = np.nan
    result = ljung_box_test(residuals, lags=5)
    assert result["n"] == 90  # Should exclude NaNs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
