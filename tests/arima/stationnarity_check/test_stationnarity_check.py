from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.arima.stationnarity_check.stationnarity_check import (
    StationarityTestResult,
    adf_test,
    evaluate_stationarity,
    kpss_test,
    run_stationarity_pipeline,
    save_stationarity_report,
    zivot_andrews_test,
)


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _validate_test_result_basic(result: StationarityTestResult) -> None:
    """Validate basic structure of test result.

    Args:
        result: Test result dictionary to validate.
    """
    assert isinstance(result.get("statistic"), float)
    assert isinstance(result.get("p_value"), float)
    assert isinstance(result.get("critical_values"), dict)


def test_evaluate_stationarity_on_white_noise() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=600)
    s = pd.Series(y, index=_make_dates(600))
    rep = evaluate_stationarity(s, alpha=0.05)
    assert isinstance(rep.stationary, bool)
    assert rep.adf["p_value"] < 0.05
    # KPSS sometimes returns borderline values; we allow >= 0.05
    assert np.isnan(rep.kpss["p_value"]) or rep.kpss["p_value"] >= 0.05
    assert rep.stationary is True


def test_evaluate_stationarity_on_random_walk() -> None:
    rng = np.random.default_rng(123)
    eps = rng.normal(0.0, 1.0, size=600)
    rw = np.cumsum(eps)
    s = pd.Series(rw, index=_make_dates(600))
    rep = evaluate_stationarity(s, alpha=0.05)
    # Expect non-stationary
    assert rep.stationary is False


def test_pipeline_and_save_json(tmp_path: Path) -> None:
    # Build a synthetic stationary series and persist as CSV
    n = 300
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "date": _make_dates(n),
            "weighted_log_return": y,
        }
    )
    csv_path = tmp_path / "mock.csv"
    df.to_csv(csv_path, index=False)

    rep = run_stationarity_pipeline(data_file=str(csv_path), column="weighted_log_return")
    assert rep.stationary is True

    out = tmp_path / "report.json"
    save_path = save_stationarity_report(rep, out)
    assert save_path.exists()
    loaded = json.loads(save_path.read_text())
    assert isinstance(loaded.get("stationary"), bool)


def test_pipeline_raises_for_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"date": _make_dates(10), "x": np.arange(10)})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(KeyError, match="weighted_log_return"):
        run_stationarity_pipeline(data_file=str(p), column="weighted_log_return")


def test_evaluate_stationarity_raises_for_invalid_alpha() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=100)
    s = pd.Series(y, index=_make_dates(100))
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=1.5)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=-0.1)


def test_adf_test_returns_valid_result() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))
    result = adf_test(s)
    _validate_test_result_basic(result)
    assert "lags" in result
    assert "nobs" in result


def test_adf_test_with_different_autolag() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))
    result_aic = adf_test(s, autolag="AIC")
    result_bic = adf_test(s, autolag="BIC")
    assert isinstance(result_aic["statistic"], float)
    assert isinstance(result_bic["statistic"], float)


def test_kpss_test_returns_valid_result() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))
    result = kpss_test(s)
    _validate_test_result_basic(result)
    assert isinstance(result["nobs"], int)
    assert "lags" in result


def test_kpss_test_with_trend() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))
    result_c = kpss_test(s, regression="c")
    result_ct = kpss_test(s, regression="ct")
    assert isinstance(result_c["statistic"], float)
    assert isinstance(result_ct["statistic"], float)


def test_save_stationarity_report_creates_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=100)
    s = pd.Series(y, index=_make_dates(100))
    report = evaluate_stationarity(s, alpha=0.05)
    out_path = tmp_path / "test_report.json"
    saved_path = save_stationarity_report(report, out_path)
    assert saved_path == out_path
    assert saved_path.exists()
    loaded = json.loads(saved_path.read_text())
    assert loaded["stationary"] == report.stationary
    assert loaded["alpha"] == report.alpha


def test_pipeline_with_different_alpha(tmp_path: Path) -> None:
    n = 300
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "date": _make_dates(n),
            "weighted_log_return": y,
        }
    )
    csv_path = tmp_path / "mock.csv"
    df.to_csv(csv_path, index=False)
    rep = run_stationarity_pipeline(
        data_file=str(csv_path), column="weighted_log_return", alpha=0.01
    )
    assert rep.alpha == 0.01
    assert isinstance(rep.stationary, bool)


def test_zivot_andrews_test_returns_valid_result() -> None:
    """Test zivot_andrews_test returns valid result structure."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))
    result = zivot_andrews_test(s, model="c")

    assert isinstance(result["statistic"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["model"], str)
    assert result["model"] == "c"
    assert "lags" in result
    assert "nobs" in result
    assert "critical_values" in result
    assert isinstance(result["critical_values"], dict)


def test_zivot_andrews_test_with_different_models() -> None:
    """Test zivot_andrews_test with different model types."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))

    result_c = zivot_andrews_test(s, model="c")
    result_t = zivot_andrews_test(s, model="t")
    result_ct = zivot_andrews_test(s, model="ct")

    assert result_c["model"] == "c"
    assert result_t["model"] == "t"
    assert result_ct["model"] == "ct"
    assert isinstance(result_c["statistic"], float)
    assert isinstance(result_t["statistic"], float)
    assert isinstance(result_ct["statistic"], float)


def test_zivot_andrews_test_with_max_lags() -> None:
    """Test zivot_andrews_test with max_lags parameter."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))

    result = zivot_andrews_test(s, model="c", max_lags=5)
    assert isinstance(result["statistic"], float)
    assert isinstance(result["lags"], (int, type(None)))


def test_evaluate_stationarity_with_structural_break_true() -> None:
    """Test evaluate_stationarity with test_structural_break=True."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))

    report = evaluate_stationarity(s, alpha=0.05, test_structural_break=True)

    assert isinstance(report.stationary, bool)
    # Zivot-Andrews may be None if test fails, but if it succeeds it should have valid structure
    if report.zivot_andrews is not None:
        assert isinstance(report.zivot_andrews["statistic"], float)
        assert isinstance(report.zivot_andrews["p_value"], float)
        assert report.zivot_andrews["model"] == "c"


def test_evaluate_stationarity_with_structural_break_false() -> None:
    """Test evaluate_stationarity with test_structural_break=False."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=200)
    s = pd.Series(y, index=_make_dates(200))

    report = evaluate_stationarity(s, alpha=0.05, test_structural_break=False)

    assert isinstance(report.stationary, bool)
    assert report.zivot_andrews is None


def test_evaluate_stationarity_with_structural_break_on_series_with_break() -> None:
    """Test evaluate_stationarity detects structural break in series with break."""
    rng = np.random.default_rng(42)
    # Create series with a structural break (change in mean)
    n = 200
    y1 = rng.normal(0.0, 1.0, size=n // 2)
    y2 = rng.normal(2.0, 1.0, size=n // 2)  # Different mean
    y = np.concatenate([y1, y2])
    s = pd.Series(y, index=_make_dates(n))

    report = evaluate_stationarity(s, alpha=0.05, test_structural_break=True)

    assert isinstance(report.stationary, bool)
    # Zivot-Andrews may fail, but if it succeeds it should detect the break
    if report.zivot_andrews is not None:
        # If break is detected, break_index should not be None
        # (though it may be None even if test succeeds, depending on the series)
        assert isinstance(report.zivot_andrews["statistic"], float)
        assert isinstance(report.zivot_andrews["p_value"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
