"""Tests for model evaluation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set matplotlib backend before any imports that use it
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.lightgbm.eval.data_loading import load_dataset
from src.lightgbm.eval.model_evaluation import compute_metrics, evaluate_model


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock dataset CSV file with split column."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "log_volatility": np.random.randn(100),
            "log_volatility_t": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "split": ["train"] * 80 + ["test"] * 20,
        }
    )
    dataset_path = tmp_path / "test_dataset.csv"
    df.to_csv(dataset_path, index=False)
    return dataset_path


@pytest.fixture
def mock_model(tmp_path: Path) -> RandomForestRegressor:
    """Create a mock LightGBM model."""
    np.random.seed(42)
    X_train = pd.DataFrame(
        {
            "log_volatility_t": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        }
    )
    y_train = pd.Series(np.random.randn(100), name="log_volatility")

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def _assert_metrics_present(metrics: dict[str, float]) -> None:
    """Assert all required metrics are present."""
    required_metrics = ["mae", "mse", "rmse", "r2"]
    for metric in required_metrics:
        assert metric in metrics


def _assert_metrics_type(metrics: dict[str, float]) -> None:
    """Assert metrics type is correct."""
    assert isinstance(metrics["mae"], float)


def _assert_non_negative_metrics(metrics: dict[str, float], metric_names: list[str]) -> None:
    """Assert metrics are non-negative."""
    for metric_name in metric_names:
        assert metrics[metric_name] >= 0


def _assert_metrics_ranges(metrics: dict[str, float]) -> None:
    """Assert metrics are in valid ranges."""
    non_negative = ["mae", "mse", "rmse"]
    _assert_non_negative_metrics(metrics, non_negative)
    assert metrics["r2"] <= 1.0


def _assert_metrics_valid(metrics: dict[str, float]) -> None:
    """Assert metrics have valid types and ranges."""
    _assert_metrics_type(metrics)
    _assert_metrics_ranges(metrics)


def test_compute_metrics() -> None:
    """Test computing evaluation metrics."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100) * 0.01)
    y_pred = y_true + np.random.randn(100) * 0.001  # Add small noise

    metrics = compute_metrics(y_true, y_pred.to_numpy())

    _assert_metrics_present(metrics)
    _assert_metrics_valid(metrics)


def _assert_zero_metrics(metrics: dict[str, float], metric_names: list[str]) -> None:
    """Assert metrics are approximately zero."""
    for metric_name in metric_names:
        assert metrics[metric_name] == pytest.approx(0.0, abs=1e-10)


def _assert_perfect_metrics(metrics: dict[str, float]) -> None:
    """Assert metrics indicate perfect prediction."""
    zero_metrics = ["mae", "mse", "rmse"]
    _assert_zero_metrics(metrics, zero_metrics)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_perfect_prediction() -> None:
    """Test metrics with perfect predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = compute_metrics(y_true, y_pred)
    _assert_perfect_metrics(metrics)


def test_compute_metrics_consistency() -> None:
    """Test that RMSE is sqrt of MSE."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred = np.random.randn(100)

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def _assert_required_keys(results: dict, keys: list[str]) -> None:
    """Assert required keys are present in results."""
    for key in keys:
        assert key in results


def _assert_model_name(results: dict, model_name: str) -> None:
    """Assert model name is correct."""
    assert results["model_name"] == model_name


def _assert_evaluation_results_structure(results: dict, model_name: str) -> None:
    """Assert evaluation results have required structure."""
    required_keys = [
        "model_name",
        "test_metrics",
        "test_size",
        "n_features",
        "feature_importances",
        "shap_plot_path",
    ]
    _assert_required_keys(results, required_keys)
    _assert_model_name(results, model_name)


def _assert_evaluation_metrics(results: dict) -> None:
    """Assert evaluation metrics are present."""
    metrics = results["test_metrics"]
    _assert_metrics_present(metrics)


def test_evaluate_model(
    mock_model: RandomForestRegressor,
    mock_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test evaluating model on test set."""
    X_test, y_test = load_dataset(mock_dataset, split="test")

    from src.lightgbm.eval import model_evaluation as eval_module

    monkeypatch.setattr(eval_module, "LIGHTGBM_SHAP_PLOTS_DIR", tmp_path / "shap")

    # Disable SHAP computation to avoid matplotlib compatibility issues
    results = evaluate_model(mock_model, X_test, y_test, "test_model", compute_shap=False)

    _assert_evaluation_results_structure(results, "test_model")
    _assert_evaluation_metrics(results)
    assert len(results["feature_importances"]) == X_test.shape[1]


def test_evaluate_model_without_shap(
    mock_model: RandomForestRegressor,
    mock_dataset: Path,
) -> None:
    """Test evaluating model without computing SHAP values."""
    X_test, y_test = load_dataset(mock_dataset, split="test")

    results = evaluate_model(
        mock_model,
        X_test,
        y_test,
        "test_model_no_shap",
        compute_shap=False,
    )

    _assert_evaluation_results_structure(results, "test_model_no_shap")
    _assert_evaluation_metrics(results)
    assert results["shap_plot_path"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
