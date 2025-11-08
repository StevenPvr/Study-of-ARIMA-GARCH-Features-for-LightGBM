"""Unit tests for data preparation module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import MagicMock

import numpy as np
import pytest

# Restore real pandas module if it was mocked by conftest
# This is needed because conftest.py mocks pandas globally, but these tests
# need real pandas functionality
if isinstance(sys.modules.get("pandas"), MagicMock):
    # Temporarily remove the mock to import real pandas
    _pandas_mock = sys.modules.pop("pandas", None)
    try:
        import pandas as _real_pandas

        sys.modules["pandas"] = _real_pandas
    except ImportError:
        # If import fails, restore the mock
        if _pandas_mock:
            sys.modules["pandas"] = _pandas_mock
        raise

import importlib

# Now import pandas (will get the real one if we restored it)
import pandas as pd

# Import and reload the module to ensure it uses real pandas
from src.data_preparation import data_preparation

importlib.reload(data_preparation)
from src.data_preparation.data_preparation import load_train_test_data, split_train_test


@pytest.fixture
def sample_weighted_returns() -> pd.DataFrame:
    """Create sample weighted log returns data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 100)
    return pd.DataFrame(
        {
            "date": dates,
            "weighted_log_return": returns,
            "weighted_open": 100 + np.cumsum(returns),
            "weighted_closing": 100 + np.cumsum(returns),
        }
    )


@pytest.fixture
def temp_input_file(sample_weighted_returns: pd.DataFrame, tmp_path: Path) -> Path:
    """Create temporary input file."""
    file_path = tmp_path / "input.csv"
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the file
    sample_weighted_returns.to_csv(file_path, index=False)
    # Verify file was created
    if not file_path.exists():
        msg = f"Failed to create input file at {file_path}"
        raise RuntimeError(msg)
    return file_path


@pytest.fixture
def temp_output_file(tmp_path: Path) -> Path:
    """Create temporary output file path."""
    return tmp_path / "output.csv"


class TestSplitTrainTest:
    """Tests for split_train_test function."""

    def test_split_train_test_default(
        self,
        sample_weighted_returns: pd.DataFrame,
        temp_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_train_test with default parameters."""
        output_file = tmp_path / "split.csv"
        split_train_test(
            train_ratio=0.8,
            input_file=str(temp_input_file),
            output_file=str(output_file),
        )

        assert output_file.exists()
        split_df = pd.read_csv(output_file, parse_dates=["date"])
        assert "split" in split_df.columns
        assert len(split_df[split_df["split"] == "train"]) == 80
        assert len(split_df[split_df["split"] == "test"]) == 20

    def test_split_train_test_custom_ratio(
        self,
        sample_weighted_returns: pd.DataFrame,
        temp_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_train_test with custom train ratio."""
        output_file = tmp_path / "split.csv"
        split_train_test(
            train_ratio=0.7,
            input_file=str(temp_input_file),
            output_file=str(output_file),
        )

        split_df = pd.read_csv(output_file, parse_dates=["date"])
        assert len(split_df[split_df["split"] == "train"]) == 70
        assert len(split_df[split_df["split"] == "test"]) == 30

    def test_split_train_test_invalid_ratio(
        self,
        sample_weighted_returns: pd.DataFrame,
        temp_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_train_test with invalid train ratio."""
        output_file = tmp_path / "split.csv"
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_train_test(
                train_ratio=1.5,
                input_file=str(temp_input_file),
                output_file=str(output_file),
            )

    def test_split_train_test_file_not_found(self, tmp_path: Path) -> None:
        """Test split_train_test with non-existent input file."""
        output_file = tmp_path / "split.csv"
        with pytest.raises(FileNotFoundError):
            split_train_test(
                input_file=str(tmp_path / "nonexistent.csv"),
                output_file=str(output_file),
            )

    def test_split_train_test_temporal_order(
        self,
        sample_weighted_returns: pd.DataFrame,
        temp_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that split maintains temporal order."""
        output_file = tmp_path / "split.csv"
        split_train_test(
            train_ratio=0.8,
            input_file=str(temp_input_file),
            output_file=str(output_file),
        )

        split_df = pd.read_csv(output_file, parse_dates=["date"])
        train_dates = split_df[split_df["split"] == "train"]["date"]
        test_dates = split_df[split_df["split"] == "test"]["date"]

        assert train_dates.max() < test_dates.min()

    def test_split_train_test_empty_dataframe(self, tmp_path: Path) -> None:
        """Test split_train_test with empty DataFrame."""
        empty_file = tmp_path / "empty.csv"
        empty_df = pd.DataFrame({"date": [], "weighted_log_return": []})
        empty_df.to_csv(empty_file, index=False)

        output_file = tmp_path / "split.csv"
        with pytest.raises(ValueError, match="Cleaned data DataFrame is empty"):
            split_train_test(
                input_file=str(empty_file),
                output_file=str(output_file),
            )

    def test_split_train_test_missing_columns(self, tmp_path: Path) -> None:
        """Test split_train_test with missing required columns."""
        invalid_file = tmp_path / "invalid.csv"
        invalid_df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D")})
        invalid_df.to_csv(invalid_file, index=False)

        output_file = tmp_path / "split.csv"
        with pytest.raises(KeyError, match="Missing required columns"):
            split_train_test(
                input_file=str(invalid_file),
                output_file=str(output_file),
            )

    def test_split_train_test_too_small_dataframe(self, tmp_path: Path) -> None:
        """Test split_train_test with DataFrame too small for splitting."""
        small_file = tmp_path / "small.csv"
        small_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=1, freq="D"),
                "weighted_log_return": [0.001],
            }
        )
        small_df.to_csv(small_file, index=False)

        output_file = tmp_path / "split.csv"
        with pytest.raises(ValueError, match="DataFrame must have at least 2 rows"):
            split_train_test(
                input_file=str(small_file),
                output_file=str(output_file),
            )


def _create_split_file(
    sample_weighted_returns: pd.DataFrame, tmp_path: Path, train_size: int = 80
) -> Path:
    """Create a temporary split file for testing.

    Args:
        sample_weighted_returns: Sample weighted returns DataFrame.
        tmp_path: Temporary directory path.
        train_size: Number of training samples.

    Returns:
        Path to the created split file.
    """
    split_file = tmp_path / "split.csv"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_df = pd.DataFrame(
        {
            "date": sample_weighted_returns["date"],
            "weighted_log_return": sample_weighted_returns["weighted_log_return"],
            "split": ["train"] * train_size + ["test"] * (100 - train_size),
        }
    )
    split_df.to_csv(split_file, index=False)
    return split_file


class TestLoadTrainTestData:
    """Tests for load_train_test_data function."""

    def test_load_train_test_data(
        self, sample_weighted_returns: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test load_train_test_data with valid split file."""
        split_file = _create_split_file(sample_weighted_returns, tmp_path)
        train_series, test_series = load_train_test_data(input_file=str(split_file))

        assert len(train_series) == 80
        assert len(test_series) == 20
        assert isinstance(train_series.index, pd.DatetimeIndex)
        assert isinstance(test_series.index, pd.DatetimeIndex)

    def test_load_train_test_data_file_not_found(self, tmp_path: Path) -> None:
        """Test load_train_test_data with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_train_test_data(input_file=str(tmp_path / "nonexistent.csv"))

    def test_load_train_test_data_series_content(
        self, sample_weighted_returns: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test that loaded series contain correct data."""
        split_file = _create_split_file(sample_weighted_returns, tmp_path)
        train_series, test_series = load_train_test_data(input_file=str(split_file))

        expected_train = sample_weighted_returns["weighted_log_return"].iloc[:80]
        expected_test = sample_weighted_returns["weighted_log_return"].iloc[80:]

        pd.testing.assert_series_equal(
            train_series.reset_index(drop=True),
            expected_train.reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            test_series.reset_index(drop=True),
            expected_test.reset_index(drop=True),
        )

    def test_load_train_test_data_empty_dataframe(self, tmp_path: Path) -> None:
        """Test load_train_test_data with empty split file."""
        empty_file = tmp_path / "empty_split.csv"
        empty_df = pd.DataFrame()
        empty_df.to_csv(empty_file, index=False)

        with pytest.raises(
            ValueError, match="Split data file is empty|Split data DataFrame is empty"
        ):
            load_train_test_data(input_file=str(empty_file))

    def test_load_train_test_data_missing_columns(self, tmp_path: Path) -> None:
        """Test load_train_test_data with missing required columns."""
        invalid_file = tmp_path / "invalid_split.csv"
        invalid_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "weighted_log_return": np.random.normal(0.001, 0.01, 10),
                # Missing 'split' column
            }
        )
        invalid_df.to_csv(invalid_file, index=False)

        with pytest.raises(KeyError, match="Missing required columns"):
            load_train_test_data(input_file=str(invalid_file))

    def test_load_train_test_data_empty_train_or_test(self, tmp_path: Path) -> None:
        """Test load_train_test_data when train or test data is empty."""
        split_file = tmp_path / "split_no_test.csv"
        split_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "weighted_log_return": np.random.normal(0.001, 0.01, 10),
                "split": ["train"] * 10,  # Only train, no test
            }
        )
        split_df.to_csv(split_file, index=False)

        with pytest.raises(ValueError, match="Test data is empty after splitting"):
            load_train_test_data(input_file=str(split_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
