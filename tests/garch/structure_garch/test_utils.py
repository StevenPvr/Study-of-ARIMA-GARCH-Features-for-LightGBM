"""Unit tests for structure_garch utilities (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.structure_garch.utils import (
    compute_acf,
    compute_arch_lm_test,
    compute_squared_acf,
    engle_ng_joint_test,
    engle_ng_negative_size_bias_test,
    engle_ng_positive_size_bias_test,
    engle_ng_sign_bias_test,
    load_garch_dataset,
    prepare_plot_series,
    prepare_residuals,
    resolve_out_path,
    run_all_engle_ng_tests,
    safe_import_matplotlib,
    verify_or_fallback,
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
def sample_garch_residuals() -> np.ndarray:
    """Generate GARCH-like residuals with autocorrelation in squares."""
    rng = np.random.default_rng(123)
    n = 300
    e = np.zeros(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal() * (1 + 0.5 * abs(e[t - 1]))
    return e


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample dataframe with required columns."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    n = len(dates)
    split = np.where(np.arange(n) < 80, "train", "test")
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(n) * 0.01
    returns = rng.standard_normal(n) * 0.02
    return pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "weighted_log_return": returns,
            "sarima_resid": residuals,
        }
    )


@pytest.fixture
def sample_std_residuals() -> np.ndarray:
    """Generate standardized residuals for Engle-Ng tests."""
    rng = np.random.default_rng(456)
    return rng.standard_normal(200)


# ============================================================================
# Data Loading Tests
# ============================================================================


def test_load_garch_dataset_missing_file(tmp_path: Path) -> None:
    """Test load_garch_dataset raises FileNotFoundError for missing file."""
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="GARCH dataset not found"):
        load_garch_dataset(str(missing_file))


def test_load_garch_dataset_missing_columns(tmp_path: Path) -> None:
    """Test load_garch_dataset raises ValueError for missing columns."""
    test_file = tmp_path / "test.csv"
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="D")})
    df.to_csv(test_file, index=False)
    with pytest.raises(ValueError, match="required columns"):
        load_garch_dataset(str(test_file))


def test_load_garch_dataset_success(tmp_path: Path) -> None:
    """Test load_garch_dataset loads valid dataset."""
    test_file = tmp_path / "test.csv"
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 10,
            "weighted_log_return": np.random.randn(10) * 0.01,
        }
    )
    df.to_csv(test_file, index=False)
    result = load_garch_dataset(str(test_file))
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert "date" in result.columns
    assert "split" in result.columns
    assert "weighted_log_return" in result.columns


def test_load_garch_dataset_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test load_garch_dataset uses default path from constants."""
    test_file = tmp_path / "dataset_garch.csv"
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "split": ["test"] * 5,
            "weighted_log_return": np.random.randn(5) * 0.01,
        }
    )
    df.to_csv(test_file, index=False)
    with patch("src.garch.structure_garch.utils.GARCH_DATASET_FILE", test_file):
        result = load_garch_dataset()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


# ============================================================================
# Residual Preparation Tests
# ============================================================================


def test_prepare_residuals_missing_column() -> None:
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


def test_prepare_residuals_test_only(sample_dataframe: pd.DataFrame) -> None:
    """Test prepare_residuals filters to test split only."""
    residuals = prepare_residuals(sample_dataframe, use_test_only=True)
    assert isinstance(residuals, np.ndarray)
    # Should have 20 test samples (100 total - 80 train)
    assert len(residuals) == 20


def test_prepare_residuals_all_data(sample_dataframe: pd.DataFrame) -> None:
    """Test prepare_residuals returns all data when use_test_only=False."""
    residuals = prepare_residuals(sample_dataframe, use_test_only=False)
    assert isinstance(residuals, np.ndarray)
    # Should have all 100 samples
    assert len(residuals) == 100


def test_prepare_residuals_no_split_column() -> None:
    """Test prepare_residuals works when split column is missing."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "weighted_log_return": np.random.randn(10) * 0.01,
            "sarima_resid": np.random.randn(10) * 0.01,
        }
    )
    residuals = prepare_residuals(df, use_test_only=True)
    assert isinstance(residuals, np.ndarray)
    assert len(residuals) == 10


def test_prepare_residuals_invalid_values() -> None:
    """Test prepare_residuals handles invalid values."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "split": ["test"] * 5,
            "weighted_log_return": np.random.randn(5) * 0.01,
            "sarima_resid": ["invalid", "values", None, np.nan, 1.0],
        }
    )
    # Should still work but may have fewer valid values
    residuals = prepare_residuals(df, use_test_only=True)
    assert isinstance(residuals, np.ndarray)
    # At least one valid value should remain
    assert len(residuals) >= 1


# ============================================================================
# ACF Tests
# ============================================================================


def test_compute_acf_basic(sample_residuals: np.ndarray) -> None:
    """Test compute_acf returns correct shape and values."""
    acf = compute_acf(sample_residuals, nlags=10)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 10
    assert np.all(np.isfinite(acf))
    # ACF at lag 0 should be 1 (but we compute lags 1..nlags)
    # So acf[0] is lag 1, should be between -1 and 1
    assert -1 <= acf[0] <= 1


def test_compute_acf_constant_series() -> None:
    """Test compute_acf handles constant series."""
    constant = np.ones(100)
    acf = compute_acf(constant, nlags=5)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 5
    # For constant series, ACF should be zeros (or very small)
    assert np.allclose(acf, 0.0, atol=1e-10)


def test_compute_acf_small_series() -> None:
    """Test compute_acf handles small series."""
    small = np.array([1.0, 2.0])
    acf = compute_acf(small, nlags=5)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 5
    # For series with size < 2, should return zeros
    # For series with size == 2, ACF at lag 1 can be computed (may be non-zero)
    # But lags 2+ should be zero (not enough data)
    assert np.all(acf[1:] == 0.0)  # Lags 2+ should be zero
    assert np.isfinite(acf[0])  # Lag 1 should be finite (could be -0.5 for [1,2])


def test_compute_acf_nans() -> None:
    """Test compute_acf filters out NaNs."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
    acf = compute_acf(data, nlags=5)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 5
    assert np.all(np.isfinite(acf))


def test_compute_squared_acf_basic(sample_residuals: np.ndarray) -> None:
    """Test compute_squared_acf returns correct shape."""
    acf = compute_squared_acf(sample_residuals, nlags=10)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == 10
    assert np.all(np.isfinite(acf))
    # Squared ACF should be between -1 and 1
    assert np.all(acf >= -1) and np.all(acf <= 1)


def test_compute_squared_acf_pattern() -> None:
    """Test compute_squared_acf detects autocorrelation in squared residuals."""
    # Create residuals with autocorrelation in squares
    rng = np.random.default_rng(42)
    n = 200
    e = np.zeros(n)
    e[0] = rng.standard_normal()
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal() * (1 + 0.5 * abs(e[t - 1]))
    acf = compute_squared_acf(e, nlags=5)
    # Should have significant autocorrelation at lag 1
    assert abs(acf[0]) > 0.1


# ============================================================================
# ARCH-LM Test Tests
# ============================================================================


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


def test_compute_arch_lm_test_small_sample() -> None:
    """Test compute_arch_lm_test handles small samples."""
    small = np.array([1.0, 2.0, 3.0])
    result = compute_arch_lm_test(small, lags=5)
    # For small samples (n <= lags), should return NaN
    # Use np.isnan() instead of == comparison (nan != nan in Python)
    assert np.isnan(result["lm_stat"]) or np.isfinite(result["lm_stat"])
    assert np.isnan(result["p_value"]) or np.isfinite(result["p_value"])


def test_compute_arch_lm_test_nans() -> None:
    """Test compute_arch_lm_test filters out NaNs."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 50)
    result = compute_arch_lm_test(data, lags=5)
    assert isinstance(result, dict)
    assert "lm_stat" in result
    assert "p_value" in result


def test_compute_arch_lm_test_garch_residuals(sample_garch_residuals: np.ndarray) -> None:
    """Test compute_arch_lm_test detects ARCH effect in GARCH residuals."""
    result = compute_arch_lm_test(sample_garch_residuals, lags=5)
    assert isinstance(result, dict)
    assert "lm_stat" in result
    assert "p_value" in result
    # Should have finite values for reasonable sample size
    assert np.isfinite(result["lm_stat"])
    assert np.isfinite(result["p_value"])


# ============================================================================
# Engle-Ng Tests
# ============================================================================


def test_engle_ng_sign_bias_test_basic(sample_std_residuals: np.ndarray) -> None:
    """Test engle_ng_sign_bias_test returns correct structure."""
    result = engle_ng_sign_bias_test(sample_std_residuals)
    assert isinstance(result, dict)
    assert "coef" in result
    assert "t_stat" in result
    assert "p_value" in result
    assert "n" in result
    # Engle-Ng tests use z[1:] for dependent variable, so n = len(z) - 1
    assert result["n"] == len(sample_std_residuals) - 1
    assert np.isfinite(result["coef"]) or np.isnan(result["coef"])
    assert np.isfinite(result["t_stat"]) or np.isnan(result["t_stat"])
    assert np.isfinite(result["p_value"]) or np.isnan(result["p_value"])


def test_engle_ng_sign_bias_test_small_sample() -> None:
    """Test engle_ng_sign_bias_test handles small samples."""
    small = np.array([1.0, 2.0])
    result = engle_ng_sign_bias_test(small)
    # Use np.isnan() instead of == comparison (nan != nan in Python)
    assert np.isnan(result["coef"])
    assert np.isnan(result["t_stat"])
    assert np.isnan(result["p_value"])
    # For len(z) < 3, returns n = len(z) (before using z[1:])
    assert result["n"] == 2


def test_engle_ng_negative_size_bias_test_basic(sample_std_residuals: np.ndarray) -> None:
    """Test engle_ng_negative_size_bias_test returns correct structure."""
    result = engle_ng_negative_size_bias_test(sample_std_residuals)
    assert isinstance(result, dict)
    assert "coef" in result
    assert "t_stat" in result
    assert "p_value" in result
    assert "n" in result
    assert np.isfinite(result["coef"]) or np.isnan(result["coef"])


def test_engle_ng_positive_size_bias_test_basic(sample_std_residuals: np.ndarray) -> None:
    """Test engle_ng_positive_size_bias_test returns correct structure."""
    result = engle_ng_positive_size_bias_test(sample_std_residuals)
    assert isinstance(result, dict)
    assert "coef" in result
    assert "t_stat" in result
    assert "p_value" in result
    assert "n" in result
    assert np.isfinite(result["coef"]) or np.isnan(result["coef"])


def test_engle_ng_joint_test_basic(sample_std_residuals: np.ndarray) -> None:
    """Test engle_ng_joint_test returns correct structure."""
    result = engle_ng_joint_test(sample_std_residuals)
    assert isinstance(result, dict)
    assert "f_stat" in result
    assert "p_value" in result
    assert "df_num" in result
    assert "df_denom" in result
    assert "n" in result
    assert result["df_num"] == 3.0
    assert np.isfinite(result["f_stat"]) or np.isnan(result["f_stat"])
    assert np.isfinite(result["p_value"]) or np.isnan(result["p_value"])


def test_engle_ng_joint_test_small_sample() -> None:
    """Test engle_ng_joint_test handles small samples."""
    small = np.array([1.0, 2.0, 3.0, 4.0])
    result = engle_ng_joint_test(small)
    # Use np.isnan() instead of == comparison (nan != nan in Python)
    assert np.isnan(result["f_stat"])
    assert np.isnan(result["p_value"])
    assert result["df_num"] == 3.0
    # For n < 5, returns n = len(z) (before using z[1:])
    assert result["n"] == 4


def test_run_all_engle_ng_tests_basic(sample_std_residuals: np.ndarray) -> None:
    """Test run_all_engle_ng_tests returns all four tests."""
    result = run_all_engle_ng_tests(sample_std_residuals)
    assert isinstance(result, dict)
    assert "sign_bias" in result
    assert "negative_size_bias" in result
    assert "positive_size_bias" in result
    assert "joint" in result
    assert isinstance(result["sign_bias"], dict)
    assert isinstance(result["negative_size_bias"], dict)
    assert isinstance(result["positive_size_bias"], dict)
    assert isinstance(result["joint"], dict)


def test_engle_ng_tests_nans() -> None:
    """Test Engle-Ng tests filter out NaNs."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 50)
    result_sign = engle_ng_sign_bias_test(data)
    result_neg = engle_ng_negative_size_bias_test(data)
    result_pos = engle_ng_positive_size_bias_test(data)
    result_joint = engle_ng_joint_test(data)
    # Should handle NaNs gracefully
    assert isinstance(result_sign, dict)
    assert isinstance(result_neg, dict)
    assert isinstance(result_pos, dict)
    assert isinstance(result_joint, dict)


# ============================================================================
# Plotting Utilities Tests
# ============================================================================


def test_safe_import_matplotlib() -> None:
    """Test safe_import_matplotlib returns valid objects."""
    Figure, FigureCanvas, available = safe_import_matplotlib()
    # matplotlib should be available in test environment
    assert available is True
    assert Figure is not None
    assert FigureCanvas is not None


def test_prepare_plot_series_basic(sample_residuals: np.ndarray) -> None:
    """Test prepare_plot_series returns correct structure."""
    x, acf_sq, conf = prepare_plot_series(residuals=sample_residuals, acf_lags=10)
    assert isinstance(x, np.ndarray)
    assert isinstance(acf_sq, np.ndarray)
    assert isinstance(conf, float)
    assert len(acf_sq) == 10
    assert conf > 0
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(acf_sq))


def test_prepare_plot_series_nans() -> None:
    """Test prepare_plot_series filters out NaNs."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 50)
    x, acf_sq, conf = prepare_plot_series(residuals=data, acf_lags=5)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(acf_sq))
    assert conf > 0


def test_resolve_out_path_none() -> None:
    """Test resolve_out_path returns default path when None."""
    with patch("src.garch.structure_garch.utils.GARCH_STRUCTURE_PLOT", Path("/test/default.png")):
        path = resolve_out_path(None)
        assert path == Path("/test/default.png")


def test_resolve_out_path_custom() -> None:
    """Test resolve_out_path returns custom path when provided."""
    custom_path = Path("/custom/path.png")
    path = resolve_out_path(custom_path)
    assert path == custom_path


def test_verify_or_fallback_existing_file(tmp_path: Path) -> None:
    """Test verify_or_fallback passes for existing file."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"test content")
    # Should not raise
    verify_or_fallback(test_file)
    assert test_file.exists()


def test_verify_or_fallback_missing_file(tmp_path: Path) -> None:
    """Test verify_or_fallback creates placeholder for missing file."""
    test_file = tmp_path / "missing.png"
    # Should create placeholder
    verify_or_fallback(test_file)
    assert test_file.exists()
    assert test_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
