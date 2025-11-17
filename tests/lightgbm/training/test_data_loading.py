"""Tests for data loading functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.lightgbm.training.training import load_dataset, load_optimization_results
from tests.lightgbm.training.conftest import _assert_dataset_structure


def test_load_dataset(mock_dataset: Path) -> None:
    """Test loading dataset from CSV file with train split."""
    X, y = load_dataset(mock_dataset, split="train")

    _assert_dataset_structure(X, y, expected_rows=80)
    assert X.shape[1] == 4  # feature_1, feature_2, feature_3, ticker_id
    assert "ticker_id" in X.columns


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
    df = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "split": ["train"] * 3})
    dataset_path = tmp_path / "no_target.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="log_volatility"):
        load_dataset(dataset_path)


def test_load_dataset_requires_ticker_id(tmp_path: Path) -> None:
    """Ticker column must be accompanied by ticker_id."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=6),
            "log_volatility": np.random.randn(6),
            "feature_1": np.random.randn(6),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "split": ["train"] * 6,
        }
    )
    dataset_path = tmp_path / "ticker_dataset_missing_id.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="ticker_id"):
        load_dataset(dataset_path, split="train")


def test_load_dataset_preserves_existing_ticker_id(tmp_path: Path) -> None:
    """Ticker_id column should be retained as feature."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-02-01", periods=6),
            "log_volatility": np.random.randn(6),
            "feature_1": np.random.randn(6),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ticker_id": [0, 0, 1, 1, 2, 2],
            "split": ["train"] * 6,
        }
    )
    dataset_path = tmp_path / "ticker_dataset_with_id.csv"
    df.to_csv(dataset_path, index=False)

    X, _ = load_dataset(dataset_path, split="train")

    assert "ticker" not in X.columns
    assert "ticker_id" in X.columns
    assert X["ticker_id"].nunique() == 3
    assert X["ticker_id"].dtype.kind in {"i", "u"}


def test_load_dataset_no_split_column(tmp_path: Path) -> None:
    """Test loading dataset without split column."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "log_volatility": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "ticker": ["AAA"] * 100,
            "ticker_id": [0] * 100,
        }
    )
    dataset_path = tmp_path / "no_split.csv"
    df.to_csv(dataset_path, index=False)

    # Should work fine - no filtering when split column absent
    X, y = load_dataset(dataset_path, split="train")
    assert len(X) == 100


def test_load_dataset_invalid_split(mock_dataset: Path) -> None:
    """Test loading dataset with empty split."""
    # Create dataset with only test data, then try to load train
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20),
            "log_volatility": np.random.randn(20),
            "feature_1": np.random.randn(20),
            "split": ["test"] * 20,
        }
    )
    dataset_path = mock_dataset.parent / "test_only.csv"
    df.to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="No data found for split"):
        load_dataset(dataset_path, split="train")


def test_load_optimization_results(mock_optimization_results: Path) -> None:
    """Test loading optimization results from JSON."""
    results = load_optimization_results(mock_optimization_results)

    assert "lightgbm_dataset_complete" in results
    assert "lightgbm_dataset_without_insights" in results
    assert "best_params" in results["lightgbm_dataset_complete"]
    assert "best_params" in results["lightgbm_dataset_without_insights"]


def test_load_optimization_results_missing_file() -> None:
    """Test loading optimization results with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_optimization_results(Path("/nonexistent/results.json"))


def test_load_optimization_results_invalid_format(tmp_path: Path) -> None:
    """Test loading optimization results with invalid format."""
    import json

    invalid_path = tmp_path / "invalid.json"
    with open(invalid_path, "w") as f:
        json.dump({"invalid": "format"}, f)

    with pytest.raises(ValueError, match="Missing required key"):
        load_optimization_results(invalid_path)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
