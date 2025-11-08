"""Tests for Random Forest evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.random_forest.eval.eval import (
    _run_single_evaluation,
    compute_metrics,
    compute_shap_values,
    evaluate_model,
    load_dataset,
    load_model,
    run_evaluation,
)


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock dataset CSV file with split column.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Path to mock dataset CSV file.
    """
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "weighted_log_return": np.random.randn(100) * 0.01,
            "weighted_log_return_t": np.random.randn(100) * 0.01,
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
def mock_model(tmp_path: Path) -> tuple[RandomForestRegressor, Path]:
    """Create and save a mock Random Forest model.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Tuple of (model, model_path).
    """
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = np.random.randn(100) * 0.01

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    return model, model_path


def _assert_dataset_types(X: pd.DataFrame, y: pd.Series) -> None:
    """Assert dataset types are correct.

    Args:
        X: Features DataFrame.
        y: Target Series.
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def _assert_dataset_rows(X: pd.DataFrame, y: pd.Series, expected_rows: int) -> None:
    """Assert dataset row counts.

    Args:
        X: Features DataFrame.
        y: Target Series.
        expected_rows: Expected number of rows.
    """
    assert len(X) == expected_rows
    assert len(y) == expected_rows


def _assert_dataset_columns(X: pd.DataFrame, y: pd.Series) -> None:
    """Assert dataset columns are correct.

    Args:
        X: Features DataFrame.
        y: Target Series.
    """
    excluded_cols = ["date", "split", "weighted_log_return"]
    for col in excluded_cols:
        assert col not in X.columns
    assert y.name == "weighted_log_return"


def _assert_dataset_structure(X: pd.DataFrame, y: pd.Series, expected_rows: int) -> None:
    """Assert basic dataset structure.

    Args:
        X: Features DataFrame.
        y: Target Series.
        expected_rows: Expected number of rows.
    """
    _assert_dataset_types(X, y)
    _assert_dataset_rows(X, y, expected_rows)
    _assert_dataset_columns(X, y)


def test_load_dataset(mock_dataset: Path) -> None:
    """Test loading dataset from CSV file with test split."""
    X, y = load_dataset(mock_dataset, split="test")

    _assert_dataset_structure(X, y, expected_rows=20)
    assert X.shape[1] == 4  # 4 features (including weighted_log_return_t)


def test_load_dataset_train_split(mock_dataset: Path) -> None:
    """Test loading dataset with train split."""
    X, y = load_dataset(mock_dataset, split="train")

    assert len(X) == 80  # 80 train rows
    assert len(y) == 80


def test_load_dataset_missing_file() -> None:
    """Test loading dataset with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("/nonexistent/dataset.csv"))


def test_load_dataset_empty(tmp_path: Path) -> None:
    """Test loading empty dataset."""
    empty_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty_path)


def test_load_dataset_missing_target(tmp_path: Path) -> None:
    """Test loading dataset without target column."""
    df = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "split": ["test"] * 3})
    dataset_path = tmp_path / "no_target.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="weighted_log_return"):
        load_dataset(dataset_path)


def test_load_model(mock_model: tuple[RandomForestRegressor, Path]) -> None:
    """Test loading trained model."""
    model, model_path = mock_model

    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, RandomForestRegressor)
    assert loaded_model.n_estimators == model.n_estimators


def test_load_model_missing_file() -> None:
    """Test loading model with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_model(Path("/nonexistent/model.joblib"))


def _assert_metrics_present(metrics: dict[str, float]) -> None:
    """Assert all required metrics are present.

    Args:
        metrics: Dictionary of metrics.
    """
    required_metrics = ["mae", "mse", "rmse", "r2", "mape"]
    for metric in required_metrics:
        assert metric in metrics


def _assert_metrics_type(metrics: dict[str, float]) -> None:
    """Assert metrics type is correct.

    Args:
        metrics: Dictionary of metrics.
    """
    assert isinstance(metrics["mae"], float)


def _assert_non_negative_metrics(metrics: dict[str, float], metric_names: list[str]) -> None:
    """Assert metrics are non-negative.

    Args:
        metrics: Dictionary of metrics.
        metric_names: List of metric names to check.
    """
    for metric_name in metric_names:
        assert metrics[metric_name] >= 0


def _assert_metrics_ranges(metrics: dict[str, float]) -> None:
    """Assert metrics are in valid ranges.

    Args:
        metrics: Dictionary of metrics.
    """
    non_negative = ["mae", "mse", "rmse", "mape"]
    _assert_non_negative_metrics(metrics, non_negative)
    assert metrics["r2"] <= 1.0


def _assert_metrics_valid(metrics: dict[str, float]) -> None:
    """Assert metrics have valid types and ranges.

    Args:
        metrics: Dictionary of metrics.
    """
    _assert_metrics_type(metrics)
    _assert_metrics_ranges(metrics)


def test_compute_metrics() -> None:
    """Test computing evaluation metrics."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100) * 0.01)
    y_pred = y_true + np.random.randn(100) * 0.001  # Add small noise

    metrics = compute_metrics(y_true, y_pred)

    _assert_metrics_present(metrics)
    _assert_metrics_valid(metrics)


def _assert_zero_metrics(metrics: dict[str, float], metric_names: list[str]) -> None:
    """Assert metrics are approximately zero.

    Args:
        metrics: Dictionary of metrics.
        metric_names: List of metric names to check.
    """
    for metric_name in metric_names:
        assert metrics[metric_name] == pytest.approx(0.0, abs=1e-10)


def _assert_perfect_metrics(metrics: dict[str, float]) -> None:
    """Assert metrics indicate perfect prediction.

    Args:
        metrics: Dictionary of metrics.
    """
    zero_metrics = ["mae", "mse", "rmse", "mape"]
    _assert_zero_metrics(metrics, zero_metrics)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_perfect_prediction() -> None:
    """Test metrics with perfect predictions."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = compute_metrics(y_true, y_pred)
    _assert_perfect_metrics(metrics)


def _assert_plot_path_exists(plot_path: Path) -> None:
    """Assert plot path exists and is a file.

    Args:
        plot_path: Path to plot.
    """
    assert plot_path.exists()
    assert plot_path.is_file()


def _assert_plot_path_format(plot_path: Path, model_name: str, tmp_path: Path) -> None:
    """Assert plot path has correct format.

    Args:
        plot_path: Path to plot.
        model_name: Expected model name in path.
        tmp_path: Temporary directory path.
    """
    assert plot_path.suffix == ".png"
    assert model_name in plot_path.name
    assert str(tmp_path) in str(plot_path), "Plot should be in temporary directory"


def _assert_shap_plot_path(plot_path: Path, model_name: str, tmp_path: Path) -> None:
    """Assert SHAP plot path is valid.

    Args:
        plot_path: Path to SHAP plot.
        model_name: Expected model name in path.
        tmp_path: Temporary directory path.
    """
    _assert_plot_path_exists(plot_path)
    _assert_plot_path_format(plot_path, model_name, tmp_path)


def _assert_shap_explanation(explanation: Any, X: pd.DataFrame) -> None:
    """Assert SHAP explanation has correct shape.

    Args:
        explanation: SHAP Explanation object.
        X: Features DataFrame.
    """
    assert isinstance(explanation.values, np.ndarray)
    assert explanation.values.shape[0] == len(X)
    assert explanation.values.shape[1] == X.shape[1]


def test_compute_shap_values(
    mock_model: tuple[RandomForestRegressor, Path],
    tmp_path: Path,
) -> None:
    """Test computing SHAP values and creating plots."""
    model, _ = mock_model

    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(50),
            "feature_2": np.random.randn(50),
            "feature_3": np.random.randn(50),
        }
    )

    output_dir = tmp_path / "shap"
    explanation, plot_path = compute_shap_values(
        model, X, "test_model", output_dir=output_dir, max_display=10
    )

    _assert_shap_plot_path(plot_path, "test_model", tmp_path)
    _assert_shap_explanation(explanation, X)


def _assert_required_keys(results: dict[str, Any], keys: list[str]) -> None:
    """Assert required keys are present in results.

    Args:
        results: Results dictionary.
        keys: List of required keys.
    """
    for key in keys:
        assert key in results


def _assert_model_name(results: dict[str, Any], model_name: str) -> None:
    """Assert model name is correct.

    Args:
        results: Evaluation results dictionary.
        model_name: Expected model name.
    """
    assert results["model_name"] == model_name


def _assert_evaluation_results_structure(results: dict[str, Any], model_name: str) -> None:
    """Assert evaluation results have required structure.

    Args:
        results: Evaluation results dictionary.
        model_name: Expected model name.
    """
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


def _assert_shap_plot_in_results(results: dict[str, Any], tmp_path: Path) -> None:
    """Assert SHAP plot path in results is valid.

    Args:
        results: Evaluation results dictionary.
        tmp_path: Temporary directory path.
    """
    shap_plot_path = Path(results["shap_plot_path"])
    assert shap_plot_path.exists(), "SHAP plot file should exist"
    assert str(tmp_path) in str(shap_plot_path), "Plot should be in temporary directory"


def _assert_evaluation_metrics(results: dict[str, Any]) -> None:
    """Assert evaluation metrics are present.

    Args:
        results: Evaluation results dictionary.
    """
    metrics = results["test_metrics"]
    _assert_metrics_present(metrics)


def test_evaluate_model(
    mock_model: tuple[RandomForestRegressor, Path],
    mock_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test evaluating model on test set."""
    model, _ = mock_model
    X_test, y_test = load_dataset(mock_dataset, split="test")

    from src.random_forest.eval import eval as eval_module

    monkeypatch.setattr(eval_module, "RF_SHAP_PLOTS_DIR", tmp_path / "shap")

    results = evaluate_model(model, X_test, y_test, "test_model")

    _assert_evaluation_results_structure(results, "test_model")
    _assert_shap_plot_in_results(results, tmp_path)
    _assert_evaluation_metrics(results)
    assert len(results["feature_importances"]) == X_test.shape[1]


def test_run_single_evaluation(
    mock_dataset: Path,
    mock_model: tuple[RandomForestRegressor, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test running single model evaluation."""
    _, model_path = mock_model

    from src.random_forest.eval import eval as eval_module

    monkeypatch.setattr(eval_module, "RF_SHAP_PLOTS_DIR", tmp_path / "shap")

    model_name, results = _run_single_evaluation(mock_dataset, model_path, "test_model")

    assert model_name == "test_model"
    _assert_evaluation_results_structure(results, "test_model")
    _assert_shap_plot_in_results(results, tmp_path)


def _setup_evaluation_paths(
    mock_dataset: Path,
    model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup monkeypatched paths for evaluation tests.

    Args:
        mock_dataset: Path to mock dataset.
        model_path: Path to mock model.
        tmp_path: Temporary directory path.
        monkeypatch: Pytest monkeypatch fixture.
    """
    from src.random_forest.eval import eval as eval_module

    monkeypatch.setattr(eval_module, "RF_DATASET_COMPLETE", mock_dataset)
    monkeypatch.setattr(eval_module, "RF_DATASET_WITHOUT_INSIGHTS", mock_dataset)
    monkeypatch.setattr(eval_module, "RF_MODELS_DIR", model_path.parent)
    monkeypatch.setattr(eval_module, "RF_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(eval_module, "RF_EVAL_RESULTS_FILE", tmp_path / "eval_results.json")
    monkeypatch.setattr(eval_module, "RF_SHAP_PLOTS_DIR", tmp_path / "shap")


def _create_test_models(model: RandomForestRegressor, model_path: Path) -> tuple[Path, Path]:
    """Create test model files for both rf_complete and rf_without_insights.

    Args:
        model: Mock Random Forest model.
        model_path: Path to original model file.

    Returns:
        Tuple of (complete_model_path, without_insights_model_path).
    """
    complete_model_path = model_path.parent / "rf_complete.joblib"
    without_model_path = model_path.parent / "rf_without_insights.joblib"
    joblib.dump(model, complete_model_path)
    joblib.dump(model, without_model_path)
    return complete_model_path, without_model_path


def _assert_model_results_structure(results: dict[str, Any], model_name: str) -> None:
    """Assert model results have required structure.

    Args:
        results: Model results dictionary.
        model_name: Model name.
    """
    assert "test_metrics" in results
    assert "test_size" in results
    assert "n_features" in results
    assert "feature_importances" in results


def _assert_model_shap_plot(results: dict[str, Any], model_name: str, tmp_path: Path) -> None:
    """Assert SHAP plot for a model is valid.

    Args:
        results: Model results dictionary.
        model_name: Model name.
        tmp_path: Temporary directory path.
    """
    if "shap_plot_path" in results:
        shap_plot_path = Path(results["shap_plot_path"])
        assert shap_plot_path.exists(), f"SHAP plot file should exist for {model_name}"
        assert str(tmp_path) in str(
            shap_plot_path
        ), f"Plot for {model_name} should be in temporary directory"


def _assert_saved_evaluation_results(tmp_path: Path) -> None:
    """Assert evaluation results file was saved correctly.

    Args:
        tmp_path: Temporary directory path.
    """
    eval_results_path = tmp_path / "eval_results.json"
    assert eval_results_path.exists()

    with open(eval_results_path, "r") as f:
        saved_results = json.load(f)
    assert "rf_complete" in saved_results
    assert "rf_without_insights" in saved_results


def _verify_evaluation_results(results: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    """Verify evaluation results structure and content.

    Args:
        results: Evaluation results dictionary.
        tmp_path: Temporary directory path.
    """
    assert "rf_complete" in results
    assert "rf_without_insights" in results

    for name in ["rf_complete", "rf_without_insights"]:
        _assert_model_results_structure(results[name], name)
        _assert_model_shap_plot(results[name], name, tmp_path)

    _assert_saved_evaluation_results(tmp_path)


def test_run_evaluation(
    tmp_path: Path,
    mock_dataset: Path,
    mock_model: tuple[RandomForestRegressor, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test running parallel evaluation for both models."""
    _, model_path = mock_model

    _setup_evaluation_paths(mock_dataset, model_path, tmp_path, monkeypatch)
    _create_test_models(mock_model[0], model_path)

    results = run_evaluation()
    _verify_evaluation_results(results, tmp_path)


def test_compute_metrics_consistency() -> None:
    """Test that RMSE is sqrt of MSE."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred = np.random.randn(100)

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
