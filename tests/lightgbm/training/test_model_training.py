"""Tests for model training functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any

import pandas as pd
import pytest

import lightgbm as lgb
from src.lightgbm.training.training import (
    _run_single_training,
    load_dataset,
    save_model,
    train_lightgbm,
)
from tests.lightgbm.training.conftest import (
    _assert_loaded_model,
    _assert_metadata,
    _assert_model_info,
    _assert_saved_model,
    _assert_training_results,
)


def test_train_lightgbm(mock_dataset: Path, sample_params: dict[str, Any]) -> None:
    """Test training LightGBM model."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model, info = train_lightgbm(X_train, y_train, sample_params)

    assert isinstance(model, lgb.LGBMRegressor)
    _assert_model_info(info, X_train)


def test_train_lightgbm_empty_data(sample_params: dict[str, Any]) -> None:
    """Test training with empty data."""
    X_empty = pd.DataFrame()
    y_empty = pd.Series([], dtype=float)

    with pytest.raises(ValueError, match="empty"):
        train_lightgbm(X_empty, y_empty, sample_params)


def test_save_model(tmp_path: Path, mock_dataset: Path, sample_params: dict[str, Any]) -> None:
    """Test saving model and metadata."""
    X_train, y_train = load_dataset(mock_dataset, split="train")
    model, info = train_lightgbm(X_train, y_train, sample_params)

    output_dir = tmp_path / "models"
    model_path, metadata_path = save_model(model, info, sample_params, "test_model", output_dir)

    _assert_saved_model(model_path, metadata_path)
    _assert_loaded_model(model_path)
    _assert_metadata(metadata_path, "test_model")


def test_run_single_training(
    mock_dataset: Path, sample_params: dict[str, Any], tmp_path: Path
) -> None:
    """Test running single model training."""
    models_dir = tmp_path / "models"

    model_name, results = _run_single_training(
        mock_dataset, "test_model", sample_params, models_dir=models_dir
    )

    assert model_name == "test_model"
    _assert_training_results(results, "test_model")


def test_train_lightgbm_metrics_range(mock_dataset: Path, sample_params: dict[str, Any]) -> None:
    """Test that training RMSE is in expected range."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model, info = train_lightgbm(X_train, y_train, sample_params)

    # RMSE should be non-negative
    assert info["train_rmse"] >= 0
    assert isinstance(info["train_rmse"], float)


def test_train_lightgbm_deterministic(mock_dataset: Path, sample_params: dict[str, Any]) -> None:
    """Test that training is deterministic with same random state."""
    X_train, y_train = load_dataset(mock_dataset, split="train")

    model1, info1 = train_lightgbm(X_train, y_train, sample_params)
    model2, info2 = train_lightgbm(X_train, y_train, sample_params)

    # RMSE should be identical due to fixed random state
    assert info1["train_rmse"] == info2["train_rmse"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
