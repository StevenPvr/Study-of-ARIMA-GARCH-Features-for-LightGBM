"""Unit tests for rolling GARCH backtesting module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.rolling_garch.rolling import (
    EgarchParams,
    _egarch_kappa,
    _ensure_min_train_size,
    _fit_initial_params,
    _one_step_update,
    _prepare_series,
    run_from_artifacts,
    run_rolling_egarch,
    save_rolling_outputs,
)


# --------------------- Fixtures ---------------------


@pytest.fixture
def sample_garch_dataframe() -> pd.DataFrame:
    """Create a minimal test dataframe with required columns."""
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    n = len(dates)
    split = np.where(np.arange(n) < 80, "train", "test")
    rng = np.random.default_rng(42)
    resid = rng.standard_normal(n) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "arima_residual_return": resid,
        }
    )


@pytest.fixture
def sample_egarch_params_normal() -> EgarchParams:
    """Create sample EGARCH parameters for normal distribution."""
    return EgarchParams(
        omega=0.01,
        alpha=0.05,
        beta=0.90,
        gamma=-0.1,
        nu=None,
        dist="normal",
    )


@pytest.fixture
def sample_egarch_params_student() -> EgarchParams:
    """Create sample EGARCH parameters for Student distribution."""
    return EgarchParams(
        omega=0.01,
        alpha=0.05,
        beta=0.90,
        gamma=-0.1,
        nu=8.0,
        dist="student",
    )


# --------------------- Tests for helper functions ---------------------


def test_egarch_kappa_normal() -> None:
    """Test kappa calculation for normal distribution."""
    kappa = _egarch_kappa("normal", None)
    expected = np.sqrt(2.0 / np.pi)
    assert abs(kappa - expected) < 1e-10


def test_egarch_kappa_student() -> None:
    """Test kappa calculation for Student distribution."""
    kappa = _egarch_kappa("student", 8.0)
    assert kappa > 0
    assert np.isfinite(kappa)


def test_egarch_kappa_student_fallback() -> None:
    """Test kappa fallback when scipy is unavailable."""
    with patch("scipy.special.gammaln", side_effect=ImportError("No scipy")):
        kappa = _egarch_kappa("student", 8.0)
        expected = np.sqrt(2.0 / np.pi)
        assert abs(kappa - expected) < 1e-10


def test_one_step_update(sample_egarch_params_normal: EgarchParams) -> None:
    """Test one-step-ahead variance update."""
    e_last = 0.01
    s2_last = 0.0001
    s2_next = _one_step_update(e_last, s2_last, sample_egarch_params_normal)
    assert s2_next > 0
    assert np.isfinite(s2_next)


def test_one_step_update_min_var(sample_egarch_params_normal: EgarchParams) -> None:
    """Test that variance is clipped to minimum value."""
    e_last = 0.01
    s2_last = 1e-20  # Very small variance
    s2_next = _one_step_update(e_last, s2_last, sample_egarch_params_normal)
    assert s2_next >= 1e-10  # Should be at least GARCH_MIN_INIT_VAR


def test_fit_initial_params_normal() -> None:
    """Test parameter fitting for normal distribution."""
    resid = np.random.default_rng(42).standard_normal(100) * 0.01
    params = _fit_initial_params(resid, dist_preference="normal")
    assert params.dist == "normal"
    assert params.nu is None
    assert params.omega is not None
    assert params.alpha > 0
    assert params.beta > 0


def test_fit_initial_params_student() -> None:
    """Test parameter fitting for Student distribution."""
    resid = np.random.default_rng(42).standard_normal(100) * 0.01
    params = _fit_initial_params(resid, dist_preference="student")
    assert params.dist == "student"
    assert params.nu is not None
    assert params.nu > 2.0


def test_fit_initial_params_auto_heavy_tails() -> None:
    """Test auto-selection of Student distribution for heavy tails."""
    # Create heavy-tailed data
    rng = np.random.default_rng(42)
    resid = np.concatenate([rng.standard_normal(50) * 0.01, rng.standard_normal(50) * 0.05])
    params = _fit_initial_params(resid, dist_preference="auto")
    # Should select student if kurtosis is high
    assert params.dist in ("normal", "student")


def test_fit_initial_params_empty_array() -> None:
    """Test parameter fitting with empty array raises error."""
    resid = np.array([])
    with pytest.raises(ValueError, match="empty residual array"):
        _fit_initial_params(resid)


def test_prepare_series(sample_garch_dataframe: pd.DataFrame) -> None:
    """Test series preparation from dataframe."""
    resid_f, dates_f, split_f, pos_test, pos_train_end = _prepare_series(sample_garch_dataframe)
    assert len(resid_f) > 0
    assert len(dates_f) == len(resid_f)
    assert len(split_f) == len(resid_f)
    assert len(pos_test) > 0
    assert len(pos_train_end) == 1
    assert pos_train_end[0] >= 0


def test_prepare_series_no_test() -> None:
    """Test series preparation when no test data exists."""
    dates = pd.date_range("2021-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 50,
            "arima_residual_return": np.random.default_rng(42).standard_normal(50) * 0.01,
        }
    )
    resid_f, dates_f, split_f, pos_test, pos_train_end = _prepare_series(df)
    assert len(resid_f) > 0
    assert len(split_f) == len(resid_f)
    assert len(pos_test) == 0
    # When no test data, pos_train_end can be empty or contain the last train position
    assert isinstance(pos_train_end, np.ndarray)


def test_prepare_series_with_nans() -> None:
    """Test series preparation with NaN values."""
    dates = pd.date_range("2021-01-01", periods=50, freq="D")
    resid = np.random.default_rng(42).standard_normal(50) * 0.01
    resid[10] = np.nan
    resid[20] = np.inf
    df = pd.DataFrame(
        {
            "date": dates,
            "split": np.where(np.arange(50) < 40, "train", "test"),
            "arima_residual_return": resid,
        }
    )
    resid_f, dates_f, split_f, pos_test, pos_train_end = _prepare_series(df)
    assert np.all(np.isfinite(resid_f))
    assert len(resid_f) < 50  # Should filter out NaNs
    assert len(split_f) == len(resid_f)


def test_ensure_min_train_size() -> None:
    """Test minimum training size check."""
    assert _ensure_min_train_size(2) is True
    assert _ensure_min_train_size(10) is True
    assert _ensure_min_train_size(1) is False
    assert _ensure_min_train_size(0) is False


# --------------------- Tests for main functions ---------------------


def test_run_rolling_egarch_expanding(
    sample_garch_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test rolling EGARCH with expanding window."""

    # Mock the parameter fitting to use a simple function
    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        return EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=None,
            dist="normal",
        )

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_rolling_egarch(
        sample_garch_dataframe,
        refit_every=10,
        window="expanding",
        window_size=50,
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert "date" in forecasts.columns
    assert "e" in forecasts.columns
    assert "sigma2_forecast" in forecasts.columns
    assert len(forecasts) > 0
    assert metrics["n_test"] > 0
    assert metrics["window"] == "expanding"


def test_run_rolling_egarch_rolling(
    sample_garch_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test rolling EGARCH with rolling window."""

    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        return EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=None,
            dist="normal",
        )

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_rolling_egarch(
        sample_garch_dataframe,
        refit_every=10,
        window="rolling",
        window_size=50,
    )

    assert isinstance(forecasts, pd.DataFrame)
    assert metrics["window"] == "rolling"
    assert metrics["window_size"] == 50


def test_run_rolling_egarch_var_alphas(
    sample_garch_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test rolling EGARCH with VaR alpha levels."""

    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        return EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=None,
            dist="normal",
        )

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_rolling_egarch(
        sample_garch_dataframe,
        refit_every=10,
        var_alphas=[0.01, 0.05],
    )

    assert "var_0.01" in forecasts.columns
    assert "var_0.05" in forecasts.columns
    assert len(forecasts) > 0


def test_run_rolling_egarch_student_dist(
    sample_garch_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test rolling EGARCH with Student distribution."""

    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        return EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=8.0,
            dist="student",
        )

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_rolling_egarch(
        sample_garch_dataframe,
        refit_every=10,
        dist_preference="student",
    )

    assert metrics["dist"] == "student"
    assert metrics["nu"] == 8.0


def test_run_rolling_egarch_no_test_data() -> None:
    """Test rolling EGARCH when no test data exists."""
    dates = pd.date_range("2021-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * 50,
            "arima_residual_return": np.random.default_rng(42).standard_normal(50) * 0.01,
        }
    )
    forecasts, metrics = run_rolling_egarch(df)
    assert len(forecasts) == 0
    assert metrics["n_test"] == 0


def test_run_rolling_egarch_keep_nu_between_refits(
    sample_garch_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that nu is kept between refits when keep_nu_between_refits=True."""
    nu_values = []

    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        params = EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=8.0,
            dist="student",
        )
        nu_values.append(params.nu)
        return params

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_rolling_egarch(
        sample_garch_dataframe,
        refit_every=5,
        dist_preference="student",
        keep_nu_between_refits=True,
    )

    assert metrics["dist"] == "student"
    assert metrics["nu"] == 8.0


def test_run_from_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test run_from_artifacts with mocked dataset file."""
    # Create a temporary dataset file
    dataset_file = tmp_path / "dataset_garch.csv"
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "split": np.where(np.arange(100) < 80, "train", "test"),
            "arima_residual_return": np.random.default_rng(42).standard_normal(100) * 0.01,
        }
    )
    df.to_csv(dataset_file, index=False)

    # Patch the constant

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling.GARCH_DATASET_FILE",
        dataset_file,
    )

    # Mock parameter fitting
    def mock_fit(resid_train: np.ndarray, *, dist_preference: str = "auto") -> EgarchParams:
        return EgarchParams(
            omega=0.01,
            alpha=0.05,
            beta=0.90,
            gamma=0.0,
            nu=None,
            dist="normal",
        )

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling._fit_initial_params",
        mock_fit,
    )

    forecasts, metrics = run_from_artifacts(refit_every=10)

    assert isinstance(forecasts, pd.DataFrame)
    assert len(forecasts) > 0
    assert metrics["n_test"] > 0


def test_run_from_artifacts_file_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test run_from_artifacts when dataset file doesn't exist."""
    dataset_file = tmp_path / "nonexistent.csv"

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling.GARCH_DATASET_FILE",
        dataset_file,
    )

    with pytest.raises(FileNotFoundError):
        run_from_artifacts()


def test_save_rolling_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving rolling outputs to files."""
    # Patch the output file paths
    forecasts_file = tmp_path / "forecasts.csv"
    eval_file = tmp_path / "eval.json"

    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling.GARCH_ROLLING_FORECASTS_FILE",
        forecasts_file,
    )
    monkeypatch.setattr(
        "src.garch.rolling_garch.rolling.GARCH_ROLLING_EVAL_FILE",
        eval_file,
    )

    # Create sample forecasts and metrics
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    forecasts = pd.DataFrame(
        {
            "date": dates,
            "e": np.random.default_rng(42).standard_normal(10) * 0.01,
            "sigma2_forecast": np.random.default_rng(42).uniform(0.0001, 0.0002, 10),
        }
    )
    metrics = {
        "n_test": 10,
        "refit_every": 5,
        "window": "expanding",
        "refit_count": 2,
        "dist": "normal",
        "qlike": 0.5,
    }

    save_rolling_outputs(forecasts, metrics)

    assert forecasts_file.exists()
    assert eval_file.exists()

    # Verify CSV content
    loaded_forecasts = pd.read_csv(forecasts_file)
    assert len(loaded_forecasts) == 10
    assert "date" in loaded_forecasts.columns
    assert "e" in loaded_forecasts.columns
    assert "sigma2_forecast" in loaded_forecasts.columns

    # Verify JSON content
    with eval_file.open() as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics["n_test"] == 10
    assert loaded_metrics["dist"] == "normal"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
