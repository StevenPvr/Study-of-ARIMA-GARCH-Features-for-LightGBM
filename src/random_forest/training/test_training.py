"""Tests for Random Forest training module."""

from __future__ import annotations

import json
from pathlib import Path
import sys

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

from src.random_forest.training.training import (
    _run_single_training,
    load_dataset,
    load_optimization_results,
    run_training,
    save_model,
    train_random_forest,
)


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock dataset CSV file with split column.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Path to mock dataset CSV file.
    """
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "weighted_log_return": np.random.randn(100) * 0.01,
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
def mock_optimization_results(tmp_path: Path) -> Path:
    """Create mock optimization results JSON file.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Path to mock optimization results JSON file.
    """
    results = {
        "rf_dataset_complete": {
            "best_params": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
            },
            "best_mae": 0.007,
        },
        "rf_dataset_without_insights": {
            "best_params": {
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_split": 4,
                "min_samples_leaf": 1,
                "max_features": "log2",
                "bootstrap": False,
            },
            "best_mae": 0.008,
        },
    }
    results_path = tmp_path / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)
    return results_path


@pytest.fixture
def sample_params() -> dict[str, Any]:
    """Create sample Random Forest parameters.

    Returns:
        Dictionary of sample parameters.
    """
    return {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
    }


def test_load_dataset(mock_dataset: Path) -> None:
    """Test loading dataset from CSV file with train split."""
    X, y = load_dataset(mock_dataset, split="train")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 80  # 80 train rows
    assert len(y) == 80
    assert "date" not in X.columns
    assert "split" not in X.columns
    assert "weighted_log_return" not in X.columns
    assert y.name == "weighted_log_return"
    assert X.shape[1] == 3  # 3 features


def test_load_dataset_test_split(mock_dataset: Path) -> None:
    """Test loading dataset with test split."""
    X, y = load_dataset(mock_dataset, split="test")

    assert len(X) == 20  # 20 test rows
    assert len(y) == 20


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
    df = pd.DataFrame(
        {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "split": ["train"] * 3}
    )
    dataset_path = tmp_path / "no_target.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="weighted_log_return"):
        load_dataset(dataset_path)


def test_load_dataset_no_split_column(tmp_path: Path) -> None:
    """Test loading dataset without split column."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "weighted_log_return": np.random.randn(100) * 0.01,
            "feature_1": np.random.randn(100),
        }
    )
    dataset_path = tmp_path / "no_split.csv"
    df.to_csv(dataset_path, index=False)

    # Should work fine - no filtering when split column absent
    X, y = load_dataset(dataset_path, split="train")
    assert len(X) == 100


def test_load_optimization_results(mock_optimization_results: Path) -> None:
    """Test loading optimization results from JSON."""
    results = load_optimization_results(mock_optimization_results)

    assert "rf_dataset_complete" in results
    assert "rf_dataset_without_insights" in results
    assert "best_params" in results["rf_dataset_complete"]
    assert "best_params" in results["rf_dataset_without_insights"]


def test_load_optimization_results_missing_file() -> None:
    """Test loading optimization results with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_optimization_results(Path("/nonexistent/results.json"))


def test_load_optimization_results_invalid_format(tmp_path: Path) -> None:
    """Test loading optimization results with invalid format."""
    invalid_path = tmp_path / "invalid.json"
    with open(invalid_path, "w") as f:
        json.dump({"invalid": "format"}, f)

    with pytest.raises(ValueError, match="Missing required key"):
        load_optimization_results(invalid_path)


def test_train_random_forest(mock_dataset: Path, sample_params: dict[str, Any]) -> None:
    """Test training Random Forest model."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model, info = train_random_forest(X_train, y_train, sample_params)

    assert isinstance(model, RandomForestRegressor)
    assert "train_log_loss" in info
    assert "train_size" in info
    assert "n_features" in info
    assert info["train_size"] == len(X_train)
    assert info["n_features"] == X_train.shape[1]
    assert isinstance(info["train_log_loss"], float)
    assert info["train_log_loss"] >= 0  # Log loss should be non-negative


def test_train_random_forest_empty_data(sample_params: dict[str, Any]) -> None:
    """Test training with empty data."""
    X_empty = pd.DataFrame()
    y_empty = pd.Series([], dtype=float)

    with pytest.raises(ValueError, match="empty"):
        train_random_forest(X_empty, y_empty, sample_params)


def test_save_model(
    tmp_path: Path, mock_dataset: Path, sample_params: dict[str, Any]
) -> None:
    """Test saving model and metadata."""
    X_train, y_train = load_dataset(mock_dataset, split="train")
    model, info = train_random_forest(X_train, y_train, sample_params)

    output_dir = tmp_path / "models"
    model_path, metadata_path = save_model(
        model, info, sample_params, "test_model", output_dir
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert model_path.suffix == ".joblib"
    assert metadata_path.suffix == ".json"

    # Test loading saved model
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, RandomForestRegressor)

    # Test loading metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "test_model"
    assert "params" in metadata
    assert "train_info" in metadata
    assert metadata["random_state"] == 42


def test_run_single_training(
    mock_dataset: Path, sample_params: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    """Test running single model training."""
    # Monkeypatch the models directory
    from src.random_forest.training import training as training_module

    monkeypatch.setattr(training_module, "RF_MODELS_DIR", tmp_path / "models")

    model_name, results = _run_single_training(mock_dataset, "test_model", sample_params)

    assert model_name == "test_model"
    assert "model_name" in results
    assert "model_path" in results
    assert "metadata_path" in results
    assert "train_info" in results
    assert "params" in results
    assert Path(results["model_path"]).exists()
    assert Path(results["metadata_path"]).exists()
    
    # Check that train info contains log loss
    assert "train_log_loss" in results["train_info"]
    assert "train_size" in results["train_info"]
    assert "n_features" in results["train_info"]


def test_run_training(
    tmp_path: Path, mock_optimization_results: Path, mock_dataset: Path, monkeypatch: Any
) -> None:
    """Test running parallel training for both models."""
    # Monkeypatch the paths
    from src.random_forest.training import training as training_module

    monkeypatch.setattr(training_module, "RF_DATASET_COMPLETE", mock_dataset)
    monkeypatch.setattr(training_module, "RF_DATASET_WITHOUT_INSIGHTS", mock_dataset)
    monkeypatch.setattr(training_module, "RF_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(training_module, "RF_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        training_module, "RF_TRAINING_RESULTS_FILE", tmp_path / "training_results.json"
    )

    results = run_training(optimization_results_path=mock_optimization_results)

    assert "rf_complete" in results
    assert "rf_without_insights" in results

    for name in ["rf_complete", "rf_without_insights"]:
        assert "train_info" in results[name]
        assert "params" in results[name]
        assert "model_path" in results[name]
        assert Path(results[name]["model_path"]).exists()
        
        # Check that train info contains log loss
        assert "train_log_loss" in results[name]["train_info"]
        assert "train_size" in results[name]["train_info"]
        assert "n_features" in results[name]["train_info"]

    # Check training results file was saved
    training_results_path = tmp_path / "training_results.json"
    assert training_results_path.exists()

    with open(training_results_path, "r") as f:
        saved_results = json.load(f)
    assert "rf_complete" in saved_results
    assert "rf_without_insights" in saved_results


def test_train_random_forest_metrics_range(
    mock_dataset: Path, sample_params: dict[str, Any]
) -> None:
    """Test that training log loss is in expected range."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model, info = train_random_forest(X_train, y_train, sample_params)

    # Log loss should be non-negative
    assert info["train_log_loss"] >= 0
    assert isinstance(info["train_log_loss"], float)


def test_train_random_forest_deterministic(
    mock_dataset: Path, sample_params: dict[str, Any]
) -> None:
    """Test that training is deterministic with same random state."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model1, info1 = train_random_forest(X_train, y_train, sample_params)
    model2, info2 = train_random_forest(X_train, y_train, sample_params)

    # Log loss should be identical due to fixed random state
    assert info1["train_log_loss"] == info2["train_log_loss"]


def test_load_dataset_invalid_split(mock_dataset: Path) -> None:
    """Test loading dataset with empty split."""
    # Create dataset with only test data, then try to load train
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20),
            "weighted_log_return": np.random.randn(20) * 0.01,
            "feature_1": np.random.randn(20),
            "split": ["test"] * 20,
        }
    )
    dataset_path = mock_dataset.parent / "test_only.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="No data found for split"):
        load_dataset(dataset_path, split="train")
        

if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover