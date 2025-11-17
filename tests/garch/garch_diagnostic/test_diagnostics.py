"""Unit tests for post-estimation diagnostics (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_diagnostic.diagnostics import (
    compute_distribution_diagnostics,
    compute_ljung_box_on_std,
    compute_ljung_box_on_std_squared,
    save_acf_pacf_std_plots,
    save_acf_pacf_std_squared_plots,
    save_qq_plot_std_residuals,
    save_residual_plots,
)


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(300)


@pytest.fixture
def sample_garch_params() -> dict[str, float]:
    """Generate sample GARCH parameters."""
    return {"omega": 0.05, "alpha": 0.05, "beta": 0.9, "gamma": 0.0}


def _assert_ljung_box_keys(lb: dict[str, list[int] | list[float]]) -> None:
    """Assert Ljung-Box result has correct keys."""
    assert set(lb.keys()) == {"lags", "lb_stat", "lb_pvalue"}


def _assert_ljung_box_lengths(lb: dict[str, list[int] | list[float]], expected_lags: int) -> None:
    """Assert Ljung-Box result has correct lengths."""
    assert len(lb["lags"]) == expected_lags
    assert len(lb["lb_stat"]) == expected_lags
    assert len(lb["lb_pvalue"]) == expected_lags


def _assert_ljung_box_types(lb: dict[str, list[int] | list[float]]) -> None:
    """Assert Ljung-Box result has correct types."""
    assert all(isinstance(x, (int, float)) for x in lb["lb_stat"])
    assert all(isinstance(x, (int, float)) for x in lb["lb_pvalue"])


def _assert_ljung_box_structure(lb: dict[str, list[int] | list[float]], expected_lags: int) -> None:
    """Assert Ljung-Box result has correct structure."""
    _assert_ljung_box_keys(lb)
    _assert_ljung_box_lengths(lb, expected_lags)
    _assert_ljung_box_types(lb)


def test_std_squared_plots_and_ljungbox(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test ACF/PACF plots and Ljung-Box test on squared standardized residuals."""
    out = save_acf_pacf_std_squared_plots(
        sample_residuals, sample_garch_params, lags=10, outdir=tmp_path, filename="stdacf.png"
    )
    assert out.exists()
    lb = compute_ljung_box_on_std_squared(sample_residuals, sample_garch_params, lags=10)
    _assert_ljung_box_structure(lb, 10)


def test_std_plots_and_ljungbox(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test ACF/PACF plots and Ljung-Box test on standardized residuals."""
    out = save_acf_pacf_std_plots(
        sample_residuals, sample_garch_params, lags=10, outdir=tmp_path, filename="stdacf_z.png"
    )
    assert out.exists()
    lb = compute_ljung_box_on_std(sample_residuals, sample_garch_params, lags=10)
    assert set(lb.keys()) == {"lags", "lb_stat", "lb_pvalue"}
    assert len(lb["lags"]) == 10


def _assert_distribution_diagnostics_structure(diag: dict[str, float | str | None]) -> None:
    """Assert distribution diagnostics result has correct structure."""
    required_keys = {
        "dist",
        "skewness",
        "kurtosis",
        "jarque_bera_stat",
        "jarque_bera_pvalue",
        "ks_stat",
        "ks_pvalue",
    }
    assert required_keys.issubset(set(diag.keys()))
    assert isinstance(diag["skewness"], float)
    assert isinstance(diag["kurtosis"], float)


def test_distribution_diagnostics_normal(
    sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test distribution diagnostics for normal distribution."""
    diag = compute_distribution_diagnostics(
        sample_residuals, sample_garch_params, dist="normal", nu=None
    )
    _assert_distribution_diagnostics_structure(diag)
    assert diag["dist"] == "normal"


def test_distribution_diagnostics_student(
    sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test distribution diagnostics for Student-t distribution."""
    diag = compute_distribution_diagnostics(
        sample_residuals, sample_garch_params, dist="student", nu=5.0
    )
    assert diag["dist"] == "student"
    assert diag["nu"] == 5.0


def test_qq_plot_normal(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test QQ plot generation for normal distribution."""
    out = save_qq_plot_std_residuals(
        sample_residuals,
        sample_garch_params,
        dist="normal",
        nu=None,
        outdir=tmp_path,
        filename="qq_normal.png",
    )
    assert out.exists()


def test_qq_plot_student(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test QQ plot generation for Student-t distribution."""
    out = save_qq_plot_std_residuals(
        sample_residuals,
        sample_garch_params,
        dist="student",
        nu=5.0,
        outdir=tmp_path,
        filename="qq_student.png",
    )
    assert out.exists()


def test_residual_plots_with_params(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test residual plots with GARCH parameters."""
    resid_train = sample_residuals[:200]
    resid_test = sample_residuals[200:]
    out = save_residual_plots(
        resid_train,
        resid_test,
        garch_params=sample_garch_params,
        outdir=tmp_path,
        filename="residuals.png",
    )
    assert out.exists()


def test_residual_plots_without_params(tmp_path: Path, sample_residuals: np.ndarray) -> None:
    """Test residual plots without GARCH parameters."""
    resid_train = sample_residuals[:200]
    resid_test = sample_residuals[200:]
    out = save_residual_plots(
        resid_train,
        resid_test,
        garch_params=None,
        outdir=tmp_path,
        filename="residuals_no_params.png",
    )
    assert out.exists()


def test_residual_plots_empty_test(
    tmp_path: Path, sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test residual plots with empty test set."""
    resid_train = sample_residuals
    resid_test = np.array([])
    out = save_residual_plots(
        resid_train,
        resid_test,
        garch_params=sample_garch_params,
        outdir=tmp_path,
        filename="residuals_empty_test.png",
    )
    assert out.exists()


def test_ljung_box_edge_cases(
    sample_residuals: np.ndarray, sample_garch_params: dict[str, float]
) -> None:
    """Test Ljung-Box with edge cases."""
    # Test with small number of lags
    lb = compute_ljung_box_on_std_squared(sample_residuals, sample_garch_params, lags=1)
    assert len(lb["lags"]) == 1

    # Test with large number of lags (but less than sample size)
    lb = compute_ljung_box_on_std_squared(sample_residuals, sample_garch_params, lags=50)
    assert len(lb["lags"]) == 50


def test_missing_params_raises_error(sample_residuals: np.ndarray) -> None:
    """Test that missing GARCH parameters raise appropriate errors."""
    invalid_params = {"alpha": 0.05, "beta": 0.9}  # Missing omega
    with pytest.raises(KeyError):
        compute_ljung_box_on_std_squared(sample_residuals, invalid_params, lags=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
