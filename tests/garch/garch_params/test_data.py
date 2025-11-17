"""Unit tests for GARCH data loading and preparation functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_params.data import (
    _validate_residuals,
    estimate_egarch_models,
    estimate_single_model,
    load_and_prepare_data,
)
from src.garch.garch_params.estimation import ConvergenceResult


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample dataframe with required columns."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    n = len(dates)
    split = np.where(np.arange(n) < 150, "train", "test")
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal(n) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "sarima_resid": residuals,
        }
    )


@pytest.fixture
def sample_residuals() -> np.ndarray:
    """Generate sample residuals for testing."""
    rng = np.random.default_rng(123)
    return rng.normal(0.0, 0.01, size=600)


# ==================== Validation Tests ====================


def test_validate_residuals_valid(sample_residuals: np.ndarray) -> None:
    """Test _validate_residuals with valid input."""
    # Should not raise
    _validate_residuals(sample_residuals)


def test_validate_residuals_insufficient() -> None:
    """Test _validate_residuals raises ValueError for insufficient data."""
    resid = np.array([1.0, 2.0])  # Too few observations
    with pytest.raises(ValueError, match="Insufficient training residuals"):
        _validate_residuals(resid)


# ==================== Data Loading Tests ====================


@patch("src.garch.garch_params.data.load_garch_dataset")
@patch("src.garch.garch_params.data.prepare_residuals")
def test_load_and_prepare_data_success(
    mock_prepare_residuals: MagicMock,
    mock_load_garch_dataset: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test load_and_prepare_data with successful data loading."""
    mock_load_garch_dataset.return_value = sample_dataframe

    train_resid = np.random.default_rng(42).normal(0.0, 0.01, size=150)
    test_resid = np.random.default_rng(7).normal(0.0, 0.01, size=50)

    def prepare_side_effect(df: pd.DataFrame, use_test_only: bool) -> np.ndarray:
        if use_test_only:
            return test_resid
        return train_resid

    mock_prepare_residuals.side_effect = prepare_side_effect

    resid_train, resid_test = load_and_prepare_data()

    assert resid_train.size == 150
    assert resid_test.size == 50
    assert np.all(np.isfinite(resid_train))
    assert np.all(np.isfinite(resid_test))


@patch("src.garch.garch_params.data.load_garch_dataset")
def test_load_and_prepare_data_no_train_data(mock_load_garch_dataset: MagicMock) -> None:
    """Test load_and_prepare_data raises ValueError when no train data."""
    df = pd.DataFrame({"split": ["test"] * 100})
    mock_load_garch_dataset.return_value = df

    with pytest.raises(ValueError, match="No training data found"):
        load_and_prepare_data()


@patch("src.garch.garch_params.data.load_garch_dataset")
def test_load_and_prepare_data_load_failure(mock_load_garch_dataset: MagicMock) -> None:
    """Test load_and_prepare_data raises error when loading fails."""
    mock_load_garch_dataset.side_effect = Exception("Load failed")

    with pytest.raises(Exception, match="Load failed"):
        load_and_prepare_data()


# ==================== Model Estimation Tests ====================


@patch("src.garch.garch_params.estimation.estimate_egarch_mle")
def test_estimate_single_model_normal(mock_estimate_mle: MagicMock) -> None:
    """Test estimate_single_model for normal distribution."""
    convergence = ConvergenceResult(
        converged=True,
        n_iterations=10,
        final_loglik=-100.0,
        message="Optimization converged",
    )
    mock_estimate_mle.return_value = (
        {"omega": -5.0, "alpha": 0.1, "gamma": 0.0, "beta": 0.95, "loglik": -100.0},
        convergence,
    )

    rng = np.random.default_rng(42)
    resid_train = rng.normal(0.0, 0.01, size=600)

    dist, params = estimate_single_model(resid_train, "normal")

    assert dist == "normal"
    assert params["omega"] == -5.0
    assert params["beta"] == 0.95
    mock_estimate_mle.assert_called_once()


@patch("src.garch.garch_params.estimation.estimate_egarch_mle")
def test_estimate_single_model_failure(mock_estimate_mle: MagicMock) -> None:
    """Test estimate_single_model raises RuntimeError on failure."""
    mock_estimate_mle.side_effect = Exception("MLE failed")

    rng = np.random.default_rng(42)
    resid_train = rng.normal(0.0, 0.01, size=600)

    with pytest.raises(RuntimeError, match="EGARCH MLE failed"):
        estimate_single_model(resid_train, "normal")


@patch("src.garch.garch_params.data.ProcessPoolExecutor")
@patch("src.garch.garch_params.data.as_completed")
@patch("src.garch.garch_params.data.estimate_single_model")
def test_estimate_egarch_models_success(
    mock_estimate_single: MagicMock,
    mock_as_completed: MagicMock,
    mock_executor: MagicMock,
) -> None:
    """Test estimate_egarch_models with successful estimation."""

    def side_effect(resid: np.ndarray, dist: str) -> tuple[str, dict[str, float]]:
        return dist, {
            "omega": -5.0,
            "alpha": 0.1,
            "gamma": 0.0,
            "beta": 0.95,
            "loglik": -100.0,
        }

    mock_estimate_single.side_effect = side_effect

    class DummyFuture:
        def __init__(self, value: tuple[str, dict[str, float]]) -> None:
            self._value = value

        def result(self) -> tuple[str, dict[str, float]]:
            return self._value

    class DummyExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._futures: list[DummyFuture] = []

        def __enter__(self) -> DummyExecutor:
            return self

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _traceback: object | None,
        ) -> None:
            pass

        def submit(self, func: Any, *args: Any, **kwargs: Any) -> DummyFuture:
            future = DummyFuture(func(*args, **kwargs))
            self._futures.append(future)
            return future

    dummy_executor_instance = DummyExecutor()
    mock_executor.return_value = dummy_executor_instance

    rng = np.random.default_rng(42)
    resid_train = rng.normal(0.0, 0.01, size=600)

    def iter_futures(futures: list[DummyFuture]):
        for future in futures:
            yield future

    mock_as_completed.side_effect = lambda futures: iter_futures(futures)

    params_normal, params_student, params_skewt = estimate_egarch_models(resid_train)

    assert params_normal["omega"] == -5.0
    assert params_student["omega"] == -5.0
    assert params_skewt["omega"] == -5.0
    assert mock_estimate_single.call_count == 3


@patch("src.garch.garch_params.data.estimate_single_model")
def test_estimate_egarch_models_failure(mock_estimate_single: MagicMock) -> None:
    """Test estimate_egarch_models raises RuntimeError on failure."""
    mock_estimate_single.side_effect = RuntimeError("Estimation failed")

    rng = np.random.default_rng(42)
    resid_train = rng.normal(0.0, 0.01, size=600)

    with pytest.raises(RuntimeError, match="Failed to estimate EGARCH model"):
        estimate_egarch_models(resid_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
