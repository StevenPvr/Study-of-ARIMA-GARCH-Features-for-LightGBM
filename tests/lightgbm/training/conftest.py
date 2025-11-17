"""Shared fixtures and test helpers for LightGBM training tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

import lightgbm as lgb


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
            "log_volatility": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "ticker": ["AAA"] * 80 + ["BBB"] * 20,
            "ticker_id": [0] * 80 + [1] * 20,
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
        "lightgbm_dataset_complete": {
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
        "lightgbm_dataset_without_insights": {
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
    """Create sample LightGBM parameters.

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


# Test helper functions
def _assert_model_info_keys(info: dict[str, Any]) -> None:
    """Assert model info has required keys.

    Args:
        info: Model info dictionary.
    """
    assert "train_rmse" in info
    assert "train_size" in info
    assert "n_features" in info


def _assert_model_info_values(info: dict[str, Any], X_train: pd.DataFrame) -> None:
    """Assert model info values are correct.

    Args:
        info: Model info dictionary.
        X_train: Training features DataFrame.
    """
    assert info["train_size"] == len(X_train)
    assert info["n_features"] == X_train.shape[1]
    assert isinstance(info["train_rmse"], float)
    assert info["train_rmse"] >= 0


def _assert_model_info(info: dict[str, Any], X_train: pd.DataFrame) -> None:
    """Assert model info structure and values.

    Args:
        info: Model info dictionary.
        X_train: Training features DataFrame.
    """
    _assert_model_info_keys(info)
    _assert_model_info_values(info, X_train)


def _assert_dataset_types(X: pd.DataFrame, y: pd.Series, expected_rows: int) -> None:
    """Assert dataset types and row counts.

    Args:
        X: Features DataFrame.
        y: Target Series.
        expected_rows: Expected number of rows.
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == expected_rows
    assert len(y) == expected_rows


def _assert_dataset_columns(X: pd.DataFrame, y: pd.Series) -> None:
    """Assert dataset columns are correct.

    Args:
        X: Features DataFrame.
        y: Target Series.
    """
    assert "date" not in X.columns
    assert "split" not in X.columns
    assert "log_volatility" not in X.columns
    assert y.name == "log_volatility"


def _assert_dataset_structure(X: pd.DataFrame, y: pd.Series, expected_rows: int) -> None:
    """Assert dataset structure is correct.

    Args:
        X: Features DataFrame.
        y: Target Series.
        expected_rows: Expected number of rows.
    """
    _assert_dataset_types(X, y, expected_rows)
    _assert_dataset_columns(X, y)


def _assert_saved_model(model_path: Path, metadata_path: Path) -> None:
    """Assert saved model files exist and are correct.

    Args:
        model_path: Path to model file.
        metadata_path: Path to metadata file.
    """
    assert model_path.exists()
    assert metadata_path.exists()
    assert model_path.suffix == ".joblib"
    assert metadata_path.suffix == ".json"


def _assert_loaded_model(model_path: Path) -> lgb.LGBMRegressor:
    """Assert model can be loaded and is correct type.

    Args:
        model_path: Path to model file.

    Returns:
        Loaded model.
    """
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, lgb.LGBMRegressor)
    return loaded_model


def _assert_metadata(metadata_path: Path, expected_name: str) -> dict[str, Any]:
    """Assert metadata file structure and content.

    Args:
        metadata_path: Path to metadata file.
        expected_name: Expected model name.

    Returns:
        Loaded metadata dictionary.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == expected_name
    assert "params" in metadata
    assert "train_info" in metadata
    assert metadata["random_state"] == 42
    return metadata


def _assert_training_results_keys(results: dict[str, Any]) -> None:
    """Assert training results have required keys.

    Args:
        results: Training results dictionary.
    """
    assert "model_path" in results
    assert "metadata_path" in results
    assert "train_info" in results
    assert "params" in results


def _assert_training_results_files(results: dict[str, Any]) -> None:
    """Assert training results files exist.

    Args:
        results: Training results dictionary.
    """
    assert Path(results["model_path"]).exists()
    assert Path(results["metadata_path"]).exists()


def _assert_training_results_info(results: dict[str, Any]) -> None:
    """Assert training results info structure.

    Args:
        results: Training results dictionary.
    """
    assert "train_rmse" in results["train_info"]
    assert "train_size" in results["train_info"]
    assert "n_features" in results["train_info"]


def _assert_training_results(results: dict[str, Any], expected_name: str) -> None:
    """Assert training results structure.

    Args:
        results: Training results dictionary.
        expected_name: Expected model name.
    """
    assert results["model_name"] == expected_name
    _assert_training_results_keys(results)
    _assert_training_results_files(results)
    _assert_training_results_info(results)


def _assert_results_file(results_file: Path) -> dict[str, Any]:
    """Assert results file exists and contains expected keys.

    Args:
        results_file: Path to results file.

    Returns:
        Loaded results dictionary.
    """
    assert results_file.exists()
    with open(results_file, "r") as f:
        saved_results = json.load(f)
    assert "lightgbm_complete" in saved_results
    assert "lightgbm_without_insights" in saved_results
    return saved_results
