"""Unit tests for GARCH data visualization utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_data_visualisation.utils import (
    extract_dates_from_dataframe,
    prepare_test_dataframe,
    prepare_x_axis,
)


def test_prepare_x_axis_with_dates() -> None:
    """Test x-axis preparation with dates."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    result = prepare_x_axis(dates, 10)
    assert len(result) == 10
    assert result.dtype == object


def test_prepare_x_axis_without_dates() -> None:
    """Test x-axis preparation without dates."""
    result = prepare_x_axis(None, 10)
    assert len(result) == 10
    assert result.dtype == float
    assert np.array_equal(result, np.arange(10, dtype=float))


def test_prepare_x_axis_invalid_dates() -> None:
    """Test x-axis preparation with invalid dates falls back to indices."""
    invalid_dates = ["invalid", "dates", "here"]
    result = prepare_x_axis(invalid_dates, 3)
    assert len(result) == 3
    assert result.dtype == float


def test_extract_dates_from_dataframe_with_date() -> None:
    """Test date extraction when date column exists."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "value": [1, 2, 3, 4, 5],
        }
    )
    result = extract_dates_from_dataframe(df)
    assert result is not None
    assert len(result) == 5


def test_extract_dates_from_dataframe_without_date() -> None:
    """Test date extraction when date column is missing."""
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    result = extract_dates_from_dataframe(df)
    assert result is None


def test_prepare_test_dataframe_with_split() -> None:
    """Test test dataframe preparation with valid split."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "split": ["train"] * 5 + ["test"] * 5,
            "weighted_return": np.random.randn(10),
        }
    )
    result = prepare_test_dataframe(df)
    assert result is not None
    assert len(result) == 5
    assert all(result["split"] == "test")


def test_prepare_test_dataframe_without_split() -> None:
    """Test test dataframe preparation without split column."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "weighted_return": np.random.randn(10),
        }
    )
    result = prepare_test_dataframe(df)
    assert result is not None
    assert len(result) == 10


def test_prepare_test_dataframe_empty_test_split() -> None:
    """Test test dataframe preparation with empty test split."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "split": ["train"] * 10,
            "weighted_return": np.random.randn(10),
        }
    )
    result = prepare_test_dataframe(df)
    assert result is not None
    assert len(result) == 10


def test_prepare_test_dataframe_with_log_returns() -> None:
    """Test test dataframe preparation with log returns conversion."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "split": ["test"] * 5,
            "weighted_log_return": np.array([0.01, -0.02, 0.03, -0.01, 0.02]),
        }
    )
    result = prepare_test_dataframe(df)
    assert result is not None
    assert "weighted_return" in result.columns


def test_prepare_test_dataframe_no_returns() -> None:
    """Test test dataframe preparation without returns columns."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "split": ["test"] * 5,
            "other_column": [1, 2, 3, 4, 5],
        }
    )
    result = prepare_test_dataframe(df)
    assert result is None


def test_prepare_test_dataframe_temporal_validation() -> None:
    """Test that temporal split validation is performed."""
    # Create invalid split (test before train)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "split": ["test"] * 5 + ["train"] * 5,
            "weighted_return": np.random.randn(10),
        }
    )
    # Should not raise, but log warning
    result = prepare_test_dataframe(df)
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
