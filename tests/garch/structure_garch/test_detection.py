"""Unit tests for identification step (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.structure_garch.detection import detect_heteroskedasticity, plot_arch_diagnostics
from src.garch.structure_garch.utils import (
    compute_arch_lm_test,
    compute_squared_acf,
    prepare_residuals,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(200)


@pytest.fixture
def garch_like_residuals() -> np.ndarray:
    """Generate GARCH-like residuals with autocorrelation in squares."""
    rng = np.random.default_rng(123)
    n = 300
    e = np.zeros(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal() * (1 + 0.5 * abs(e[t - 1]))
    return e


# ============================================================================
# ACF Tests
# ============================================================================


def test_compute_squared_acf_pattern() -> None:
    """Test compute_squared_acf detects autocorrelation pattern."""
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)
    acf = compute_squared_acf(resid, nlags=5)
    assert len(acf) == 5
    assert abs(acf[1]) > 0.5


def test_compute_squared_acf_basic(sample_residuals: np.ndarray) -> None:
    """Test compute_squared_acf returns correct shape and finite values."""
    acf = compute_squared_acf(sample_residuals, nlags=10)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 10
    assert np.all(np.isfinite(acf))
    assert np.all(acf >= -1) and np.all(acf <= 1)


def test_compute_squared_acf_garch_like(garch_like_residuals: np.ndarray) -> None:
    """Test compute_squared_acf detects autocorrelation in GARCH-like residuals."""
    acf = compute_squared_acf(garch_like_residuals, nlags=5)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 5
    # Should have significant autocorrelation at lag 1
    assert abs(acf[0]) > 0.1


# ============================================================================
# ARCH-LM Test Tests
# ============================================================================


def test_arch_lm_pvalue_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test compute_arch_lm_test uses chi2_sf for p-value computation."""
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)

    def fake_sf(x: float, df: int) -> float:
        return 0.001

    monkeypatch.setattr("src.garch.structure_garch.utils.chi2_sf", fake_sf, raising=True)
    out = compute_arch_lm_test(resid, lags=4)
    assert out["df"] == 4
    assert out["p_value"] == pytest.approx(0.001)


def test_compute_arch_lm_test_basic(sample_residuals: np.ndarray) -> None:
    """Test compute_arch_lm_test returns correct structure."""
    result = compute_arch_lm_test(sample_residuals, lags=5)
    assert isinstance(result, dict)
    assert "lm_stat" in result
    assert "p_value" in result
    assert "df" in result
    assert result["df"] == 5
    assert np.isfinite(result["lm_stat"]) or np.isnan(result["lm_stat"])
    assert np.isfinite(result["p_value"]) or np.isnan(result["p_value"])


def test_compute_arch_lm_test_garch_like(garch_like_residuals: np.ndarray) -> None:
    """Test compute_arch_lm_test detects ARCH effect in GARCH-like residuals."""
    result = compute_arch_lm_test(garch_like_residuals, lags=5)
    assert isinstance(result, dict)
    assert "lm_stat" in result
    assert "p_value" in result
    # Should have finite values for reasonable sample size
    assert np.isfinite(result["lm_stat"])
    assert np.isfinite(result["p_value"])


# ============================================================================
# Heteroskedasticity Detection Tests
# ============================================================================


def test_detect_heteroskedasticity_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test detect_heteroskedasticity sets flags correctly when ARCH effect is present."""
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 100)

    def fake_sf(x: float, df: int) -> float:
        return 0.001

    monkeypatch.setattr("src.garch.structure_garch.utils.chi2_sf", fake_sf, raising=True)
    out = detect_heteroskedasticity(resid, lags=4, acf_lags=10, alpha=0.05)
    assert out["arch_effect_present"] is True
    assert out["acf_significant"] is True
    assert isinstance(out["acf_squared"], list)
    assert len(cast(list, out["acf_squared"])) == 10
    assert "arch_lm" in out
    assert "acf_significance_level" in out


def test_detect_heteroskedasticity_no_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test detect_heteroskedasticity sets flags correctly when no ARCH effect."""
    # White noise residuals
    rng = np.random.default_rng(42)
    resid = rng.standard_normal(200)

    def fake_sf(x: float, df: int) -> float:
        return 0.5  # High p-value (no ARCH effect)

    monkeypatch.setattr("src.garch.structure_garch.utils.chi2_sf", fake_sf, raising=True)
    out = detect_heteroskedasticity(resid, lags=4, acf_lags=10, alpha=0.05)
    assert out["arch_effect_present"] is False
    assert "acf_squared" in out
    assert "arch_lm" in out


def test_detect_heteroskedasticity_structure(sample_residuals: np.ndarray) -> None:
    """Test detect_heteroskedasticity returns correct structure."""
    out = detect_heteroskedasticity(sample_residuals, lags=5, acf_lags=10, alpha=0.05)
    assert isinstance(out, dict)
    assert "arch_lm" in out
    assert "acf_squared" in out
    assert "acf_significance_level" in out
    assert "arch_effect_present" in out
    assert "acf_significant" in out
    assert isinstance(out["arch_lm"], dict)
    assert isinstance(out["acf_squared"], list)
    assert isinstance(out["arch_effect_present"], bool)
    assert isinstance(out["acf_significant"], bool)


# ============================================================================
# Residual Preparation Tests
# ============================================================================


def test_prepare_residuals_requires_sarima_column() -> None:
    """Test prepare_residuals raises ValueError when sarima_resid column is missing."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "split": ["test"] * 5,
            "weighted_log_return": np.random.randn(5) * 0.01,
        }
    )
    with pytest.raises(ValueError, match="arima_resid|sarima_resid"):
        prepare_residuals(df, use_test_only=True)


def test_prepare_residuals_with_sarima_column() -> None:
    """Test prepare_residuals succeeds when sarima_resid column is present."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "split": ["test"] * 10,
            "weighted_log_return": np.random.randn(10) * 0.01,
            "sarima_resid": np.random.randn(10) * 0.01,
        }
    )
    residuals = prepare_residuals(df, use_test_only=True)
    assert isinstance(residuals, np.ndarray)
    assert len(residuals) == 10


# ============================================================================
# Plotting Tests
# ============================================================================


def test_plot_arch_diagnostics_writes_file(tmp_path: Path) -> None:
    """Test plot_arch_diagnostics creates output file."""
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)
    out_file = tmp_path / "diag.png"
    path = plot_arch_diagnostics(resid, acf_lags=10, out_path=out_file)
    assert path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_arch_diagnostics_default_path(sample_residuals: np.ndarray, tmp_path: Path) -> None:
    """Test plot_arch_diagnostics uses default path when not provided."""
    from unittest.mock import patch

    test_file = tmp_path / "structure.png"
    with patch("src.garch.structure_garch.utils.GARCH_STRUCTURE_PLOT", test_file):
        path = plot_arch_diagnostics(sample_residuals, acf_lags=10, out_path=None)
        assert path == test_file
        assert test_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
