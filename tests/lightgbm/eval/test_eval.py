"""Tests for LightGBM evaluation orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.lightgbm.eval.eval import _run_single_evaluation, run_evaluation


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock dataset CSV file with split column.

    Creates a dataset with 16 features to match sigma_plus_base expected structure.
    After load_dataset removes date, split, log_volatility, ticker, we need 16 features remaining:
    ticker_id (added by _add_ticker_id_column) + feature_0...feature_14 = 16 total.
    So we create 20 columns total: date, split, log_volatility, ticker, feature_0...feature_14.
    Note: ticker_id is added by _add_ticker_id_column if ticker exists, so we include ticker.
    """
    np.random.seed(42)
    n_features = 15  # Features after exclusion (excluding date, split, log_volatility, ticker, ticker_id)
    data = {
        "date": pd.date_range("2020-01-01", periods=100),
        "log_volatility": np.random.randn(100),  # Target - will be excluded
        "split": ["train"] * 80 + ["test"] * 20,  # Split - will be excluded
        "ticker": ["AAPL"] * 100,  # Will be converted to ticker_id by _add_ticker_id_column
    }
    # Add features matching expected structure (these will remain after exclusion)
    # Use explicit feature names that won't conflict with any metadata columns
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(100)

    df = pd.DataFrame(data)
    # Ensure columns are in a consistent order (sorted for predictability)
    df = df[sorted(df.columns)]
    dataset_path = tmp_path / "test_dataset.csv"
    df.to_csv(dataset_path, index=False)
    return dataset_path


@pytest.fixture
def mock_model(tmp_path: Path) -> tuple[RandomForestRegressor, Path]:
    """Create and save a mock LightGBM model.

    Creates models with 16 features to match the mock dataset structure.
    After processing, the dataset will have feature_0...feature_14 + ticker_id = 16 total.
    The features are sorted alphabetically: feature_0, feature_1, feature_10, ..., feature_14, feature_2, ..., feature_9, ticker_id.
    The model must be trained with the exact same feature names in the same order.
    """
    np.random.seed(42)
    # Create features to match the processed mock dataset structure
    # After processing: feature_0...feature_14 + ticker_id = 16 features (sorted alphabetically)
    n_features = 15  # feature_0...feature_14
    data = {}
    for i in range(n_features):
        data[f"feature_{i}"] = np.random.randn(100)
    data["ticker_id"] = np.random.randint(0, 100, 100, dtype=np.int32)  # Matches what _add_ticker_id_column creates
    
    # Create DataFrame and sort columns alphabetically to match processed dataset
    # Alphabetical order: feature_0, feature_1, feature_10, feature_11, ..., feature_14, feature_2, ..., feature_9, ticker_id
    X_train = pd.DataFrame(data)
    X_train = X_train[sorted(X_train.columns)]
    y_train = pd.Series(np.random.randn(100), name="log_volatility")

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    model_path = tmp_path / "test_model.joblib"
    joblib.dump(model, model_path)

    return model, model_path


def _assert_run_structure(run: dict[str, Any]) -> None:
    """Assert a single run has required structure."""
    assert "metrics" in run
    assert "n_samples" in run
    assert run["n_samples"] > 0
    assert "mae" in run["metrics"]
    assert "mse" in run["metrics"]
    assert "rmse" in run["metrics"]
    assert "r2" in run["metrics"]


def _assert_summary_structure(summary: dict[str, Any]) -> None:
    """Assert summary has required structure."""
    if summary:
        for stats in summary.values():
            assert {"mean", "std", "min", "max"} <= set(stats.keys())


def _assert_resampled_metrics_structure(resampled: dict[str, Any]) -> None:
    """Assert resampled evaluation metrics structure."""
    assert "runs" in resampled
    assert "summary" in resampled

    runs = resampled["runs"]
    assert runs, "Expected at least one resampled run"

    for run in runs:
        _assert_run_structure(run)

    _assert_summary_structure(resampled["summary"])


def _assert_evaluation_results_structure(results: dict[str, Any], model_name: str) -> None:
    """Assert evaluation results have required structure."""
    required_keys = [
        "model_name",
        "test_metrics",
        "test_size",
        "n_features",
        "feature_importances",
        "shap_plot_path",
    ]
    for key in required_keys:
        assert key in results
    assert results["model_name"] == model_name


def test_run_single_evaluation(
    mock_dataset: Path,
    mock_model: tuple[RandomForestRegressor, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test running single model evaluation."""
    _, model_path = mock_model

    import src.constants as constants_module
    from src.lightgbm.eval import model_evaluation

    monkeypatch.setattr(constants_module, "LIGHTGBM_SHAP_PLOTS_DIR", tmp_path / "shap")

    # Disable SHAP computation to avoid matplotlib compatibility issues
    # Patch both the module function and the imported function in eval.py
    original_evaluate_model = model_evaluation.evaluate_model

    def fast_evaluate_model(*args, compute_shap: bool = True, **kwargs):
        """Wrapper to disable SHAP computation in tests."""
        return original_evaluate_model(*args, compute_shap=False, **kwargs)

    monkeypatch.setattr(model_evaluation, "evaluate_model", fast_evaluate_model)
    # Also patch the imported function in eval module (it's imported at module level)
    from src.lightgbm.eval import eval as eval_module

    monkeypatch.setattr(eval_module, "evaluate_model", fast_evaluate_model)

    # Also mock shap.plots.beeswarm as a fallback to avoid matplotlib issues
    def mock_beeswarm(*args, **kwargs):
        """Mock beeswarm plot function."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.close(fig)
        return ax

    monkeypatch.setattr("shap.plots.beeswarm", mock_beeswarm)

    model_name, results = _run_single_evaluation(mock_dataset, model_path, "test_model")

    assert model_name == "test_model"
    _assert_evaluation_results_structure(results, "test_model")


def _setup_evaluation_paths(
    mock_dataset: Path,
    model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup monkeypatched paths for evaluation tests."""
    import src.constants as constants_module
    from src.lightgbm.eval import eval as eval_module
    from src.lightgbm import model_utils as model_utils_module

    # Patch constants module
    monkeypatch.setattr(constants_module, "LIGHTGBM_DATASET_COMPLETE_FILE", mock_dataset)
    monkeypatch.setattr(constants_module, "LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE", mock_dataset)
    monkeypatch.setattr(constants_module, "LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE", mock_dataset)
    monkeypatch.setattr(constants_module, "LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE", mock_dataset)
    monkeypatch.setattr(constants_module, "LIGHTGBM_MODELS_DIR", model_path.parent)
    monkeypatch.setattr(constants_module, "LIGHTGBM_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        constants_module, "LIGHTGBM_EVAL_RESULTS_FILE", tmp_path / "eval_results.json"
    )
    monkeypatch.setattr(constants_module, "LIGHTGBM_SHAP_PLOTS_DIR", tmp_path / "shap")
    monkeypatch.setattr(
        constants_module, "LIGHTGBM_PERMUTATION_RESULTS_FILE", tmp_path / "perm.json"
    )

    # Also patch the eval module's imported constants (they're imported at module level)
    monkeypatch.setattr(eval_module, "LIGHTGBM_DATASET_COMPLETE_FILE", mock_dataset)
    monkeypatch.setattr(eval_module, "LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE", mock_dataset)
    monkeypatch.setattr(eval_module, "LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE", mock_dataset)
    monkeypatch.setattr(eval_module, "LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE", mock_dataset)
    monkeypatch.setattr(eval_module, "LIGHTGBM_MODELS_DIR", model_path.parent)
    monkeypatch.setattr(eval_module, "LIGHTGBM_EVAL_RESULTS_FILE", tmp_path / "eval_results.json")

    # Patch model_utils module's imported constants (they're imported at module level)
    monkeypatch.setattr(model_utils_module, "LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE", mock_dataset)
    monkeypatch.setattr(model_utils_module, "LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE", mock_dataset)
    monkeypatch.setattr(model_utils_module, "LIGHTGBM_MODELS_DIR", model_path.parent)

    monkeypatch.setattr(constants_module, "LIGHTGBM_PERMUTATION_PLOTS_DIR", tmp_path / "perm_plots")


def _create_test_models(model: RandomForestRegressor, model_path: Path) -> tuple[Path, Path]:
    """Create test model files for both lightgbm_complete and lightgbm_without_insights."""
    # Ensure the models directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    complete_model_path = model_path.parent / "lightgbm_complete.joblib"
    without_model_path = model_path.parent / "lightgbm_without_insights.joblib"
    sigma_plus_base_model_path = model_path.parent / "lightgbm_sigma_plus_base.joblib"
    log_volatility_model_path = model_path.parent / "lightgbm_log_volatility_only.joblib"

    for target_path in [
        complete_model_path,
        without_model_path,
        sigma_plus_base_model_path,
        log_volatility_model_path,
    ]:
        joblib.dump(model, target_path)
        # Verify the file was created
        assert target_path.exists(), f"Failed to create model file: {target_path}"

    return complete_model_path, without_model_path


def _assert_model_results_structure(results: dict[str, Any], model_name: str) -> None:
    """Assert model results have required structure."""
    assert "test_metrics" in results
    assert "test_size" in results
    assert "n_features" in results
    assert "feature_importances" in results


def _assert_model_shap_plot(results: dict[str, Any], model_name: str, tmp_path: Path) -> None:
    """Assert SHAP plot for a model is valid."""
    if "shap_plot_path" in results and results["shap_plot_path"] is not None:
        shap_plot_path = Path(results["shap_plot_path"])
        assert shap_plot_path.exists(), f"SHAP plot file should exist for {model_name}"
        assert str(tmp_path) in str(
            shap_plot_path
        ), f"Plot for {model_name} should be in temporary directory"


def _assert_saved_evaluation_results(tmp_path: Path) -> None:
    """Assert evaluation results file was saved correctly."""
    eval_results_path = tmp_path / "eval_results.json"
    assert eval_results_path.exists()

    with open(eval_results_path, "r") as f:
        saved_results = json.load(f)
    assert "lightgbm_complete" in saved_results
    assert "lightgbm_without_insights" in saved_results


def _verify_evaluation_results(results: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    """Verify evaluation results structure and content."""
    required_models = {"lightgbm_complete", "lightgbm_without_insights"}
    assert required_models <= set(results.keys())
    optional_models = {"lightgbm_sigma_plus_base", "lightgbm_log_volatility_only"}
    result_keys = set(results.keys())
    present_optional = optional_models & result_keys
    assert present_optional <= optional_models

    for name, _model_results in results.items():
        if name == "statistical_tests":
            continue
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

    # Mock statistical tests and permutation importance to avoid slow operations
    from src.lightgbm.eval import eval as eval_module
    from src.lightgbm.eval import model_evaluation, statistical_comparison

    def mock_perform_statistical_tests(_results_dict: dict[str, Any]) -> dict[str, Any]:
        """Mock statistical tests to return empty results quickly."""
        return {}

    def mock_compute_permutation_importance(_tasks: list[tuple[Path, Path, str]]) -> dict[str, Any]:
        """Mock permutation importance to return empty results quickly."""
        return {}

    # Store original evaluate_model to wrap it
    original_evaluate_model = model_evaluation.evaluate_model

    def fast_evaluate_model(*args, compute_shap: bool = True, **kwargs):
        """Wrapper to disable SHAP computation in tests."""
        return original_evaluate_model(*args, compute_shap=False, **kwargs)

    monkeypatch.setattr(
        statistical_comparison, "perform_statistical_tests", mock_perform_statistical_tests
    )
    # Also patch the imported function in eval module (it's imported at module level)
    monkeypatch.setattr(eval_module, "perform_statistical_tests", mock_perform_statistical_tests)
    # Note: _compute_permutation_importance is commented out in eval.py, so no need to mock it
    monkeypatch.setattr(model_evaluation, "evaluate_model", fast_evaluate_model)
    # Also patch the imported function in eval module (it's imported at module level)
    monkeypatch.setattr(eval_module, "evaluate_model", fast_evaluate_model)

    # Also mock shap.plots.beeswarm as a fallback to avoid matplotlib issues
    def mock_beeswarm(*args, **kwargs):
        """Mock beeswarm plot function."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.close(fig)
        return ax

    monkeypatch.setattr("shap.plots.beeswarm", mock_beeswarm)

    results = run_evaluation()
    _verify_evaluation_results(results, tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
