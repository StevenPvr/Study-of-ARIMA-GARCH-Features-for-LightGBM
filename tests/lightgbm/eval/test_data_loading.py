"""Tests for data loading utilities."""

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
from sklearn.ensemble import RandomForestRegressor

from src.lightgbm.eval.data_loading import load_dataset, load_model


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
def mock_model(tmp_path: Path) -> tuple[RandomForestRegressor, Path]:
    """Create and save a mock LightGBM model.

    Args:
        tmp_path: Temporary directory path.

    Returns:
        Tuple of (model, model_path).
    """
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

    model_path = tmp_path / "test_model.joblib"
    import joblib

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
    excluded_cols = ["date", "split", "log_volatility"]
    for col in excluded_cols:
        assert col not in X.columns
    assert y.name == "log_volatility"


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
    assert X.shape[1] == 4  # 4 features (including log_volatility_t)


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

    with pytest.raises(ValueError, match="log_volatility"):
        load_dataset(dataset_path)


def test_load_dataset_validates_temporal_order(tmp_path: Path) -> None:
    """Test that load_dataset validates temporal order of train/test split."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Create dataset with look-ahead bias (train and test dates mixed)
    mixed_splits = ["train"] * 50 + ["test"] * 30 + ["train"] * 20
    df = pd.DataFrame(
        {
            "date": dates,
            "split": mixed_splits,
            "log_volatility": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        }
    )
    dataset_path = tmp_path / "lookahead_bias.csv"
    df.to_csv(dataset_path, index=False)

    # Should raise ValueError due to look-ahead bias
    with pytest.raises(ValueError, match="Look-ahead bias detected"):
        load_dataset(dataset_path, split="test")


def test_load_dataset_validates_temporal_order_ticker_level(tmp_path: Path) -> None:
    """Test that load_dataset validates temporal order for ticker-level data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    # Create dataset with look-ahead bias for one ticker
    splits_aapl = ["train"] * 50 + ["test"] * 30 + ["train"] * 20
    splits_msft = ["train"] * 80 + ["test"] * 20
    df = pd.DataFrame(
        {
            "date": dates.tolist() + dates.tolist(),
            "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
            "split": splits_aapl + splits_msft,
            "log_volatility": np.random.randn(200),
            "feature_1": np.random.randn(200),
            "feature_2": np.random.randn(200),
        }
    )
    dataset_path = tmp_path / "lookahead_bias_ticker.csv"
    df.to_csv(dataset_path, index=False)

    # Should raise ValueError due to look-ahead bias for AAPL
    with pytest.raises(ValueError, match="Look-ahead bias detected.*AAPL"):
        load_dataset(dataset_path, split="test")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
