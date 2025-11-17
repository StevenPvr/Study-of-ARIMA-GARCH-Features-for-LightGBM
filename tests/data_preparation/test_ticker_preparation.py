"""Unit tests for ticker preparation module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from unittest.mock import MagicMock

import numpy as np
import pytest

# Restore real pandas module if it was mocked by conftest
if isinstance(sys.modules.get("pandas"), MagicMock):
    _pandas_mock = sys.modules.pop("pandas", None)
    try:
        import pandas as _real_pandas

        sys.modules["pandas"] = _real_pandas
    except ImportError:
        if _pandas_mock:
            sys.modules["pandas"] = _pandas_mock
        raise

import importlib

import pandas as pd

from src.data_preparation import ticker_preparation

importlib.reload(ticker_preparation)
from src.data_preparation.ticker_preparation import split_tickers_train_test


@pytest.fixture
def sample_ticker_data() -> pd.DataFrame:
    """Create sample ticker data with multiple tickers."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    np.random.seed(42)

    data_list = []
    for ticker in tickers:
        for date in dates:
            open_price = np.random.uniform(100, 200)
            close_price = np.random.uniform(100, 200)
            high_price = max(open_price, close_price) + np.random.uniform(0, 5)
            low_price = min(open_price, close_price) - np.random.uniform(0, 5)
            data_list.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": np.random.uniform(1000000, 10000000),
                }
            )

    return pd.DataFrame(data_list)


@pytest.fixture
def temp_ticker_input_file(sample_ticker_data: pd.DataFrame, tmp_path: Path) -> Path:
    """Create temporary ticker input file."""
    file_path = tmp_path / "ticker_input.csv"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ticker_data.to_csv(file_path, index=False)
    if not file_path.exists():
        msg = f"Failed to create ticker input file at {file_path}"
        raise RuntimeError(msg)
    return file_path


class TestSplitTickersTrainTest:
    """Tests for split_tickers_train_test function."""

    def test_split_tickers_train_test_default(
        self,
        sample_ticker_data: pd.DataFrame,
        temp_ticker_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_tickers_train_test with default parameters."""
        output_file = tmp_path / "ticker_split.parquet"
        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(temp_ticker_input_file),
            output_file=str(output_file),
        )

        assert output_file.exists()
        split_df = pd.read_parquet(output_file)
        assert "split" in split_df.columns
        assert split_df["tickers"].nunique() == 3

        split_df["date"] = pd.to_datetime(split_df["date"])
        train_ratio_observed = len(split_df[split_df["split"] == "train"]) / len(split_df)
        assert abs(train_ratio_observed - 0.8) < 0.05

        for _, ticker_df in split_df.groupby("tickers"):
            unique_splits = set(ticker_df["split"].unique())
            assert unique_splits == {"train", "test"}
            assert ticker_df["date"].is_monotonic_increasing

    def test_split_tickers_train_test_custom_ratio(
        self,
        sample_ticker_data: pd.DataFrame,
        temp_ticker_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_tickers_train_test with custom train ratio."""
        output_file = tmp_path / "ticker_split.parquet"
        split_tickers_train_test(
            train_ratio=0.7,
            input_file=str(temp_ticker_input_file),
            output_file=str(output_file),
        )

        split_df = pd.read_parquet(output_file)
        split_df["date"] = pd.to_datetime(split_df["date"])
        train_ratio_observed = len(split_df[split_df["split"] == "train"]) / len(split_df)
        assert abs(train_ratio_observed - 0.7) < 0.05

        for _, ticker_df in split_df.groupby("tickers"):
            unique_splits = set(ticker_df["split"].unique())
            assert unique_splits == {"train", "test"}
            assert ticker_df["date"].is_monotonic_increasing

    def test_split_tickers_train_test_invalid_ratio(
        self,
        sample_ticker_data: pd.DataFrame,
        temp_ticker_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test split_tickers_train_test with invalid train ratio."""
        output_file = tmp_path / "ticker_split.parquet"
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_tickers_train_test(
                train_ratio=1.5,
                input_file=str(temp_ticker_input_file),
                output_file=str(output_file),
            )

    def test_split_tickers_train_test_file_not_found(self, tmp_path: Path) -> None:
        """Test split_tickers_train_test with non-existent input file."""
        output_file = tmp_path / "ticker_split.parquet"
        with pytest.raises(FileNotFoundError):
            split_tickers_train_test(
                input_file=str(tmp_path / "nonexistent.csv"),
                output_file=str(output_file),
            )

    def test_split_tickers_train_test_temporal_order(
        self,
        sample_ticker_data: pd.DataFrame,
        temp_ticker_input_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test that split maintains temporal order for each ticker."""
        output_file = tmp_path / "ticker_split.parquet"
        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(temp_ticker_input_file),
            output_file=str(output_file),
        )

        split_df = pd.read_parquet(output_file)
        split_df["date"] = pd.to_datetime(split_df["date"])

        for ticker in split_df["tickers"].unique():
            ticker_df = split_df[split_df["tickers"] == ticker]
            train_dates = ticker_df[ticker_df["split"] == "train"]["date"]
            test_dates = ticker_df[ticker_df["split"] == "test"]["date"]

            assert train_dates.max() < test_dates.min()

    def test_split_tickers_train_test_empty_dataframe(self, tmp_path: Path) -> None:
        """Test split_tickers_train_test with empty DataFrame."""
        empty_file = tmp_path / "empty.csv"
        empty_df = pd.DataFrame(
            {"date": [], "ticker": [], "open": [], "high": [], "low": [], "close": [], "volume": []}
        )
        empty_df.to_csv(empty_file, index=False)

        output_file = tmp_path / "ticker_split.parquet"
        with pytest.raises(ValueError, match="Ticker data DataFrame is empty"):
            split_tickers_train_test(
                input_file=str(empty_file),
                output_file=str(output_file),
            )

    def test_split_tickers_train_test_missing_columns(self, tmp_path: Path) -> None:
        """Test split_tickers_train_test with missing required columns."""
        invalid_file = tmp_path / "invalid.csv"
        invalid_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "ticker": ["AAPL"] * 10,
            }
        )
        invalid_df.to_csv(invalid_file, index=False)

        output_file = tmp_path / "ticker_split.parquet"
        with pytest.raises(KeyError, match="Missing required columns"):
            split_tickers_train_test(
                input_file=str(invalid_file),
                output_file=str(output_file),
            )

    def test_split_tickers_train_test_single_ticker_too_small(self, tmp_path: Path) -> None:
        """Test split_tickers_train_test with single ticker having too few observations."""
        small_file = tmp_path / "small.csv"
        small_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=1, freq="D"),
                "ticker": ["AAPL"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000000.0],
            }
        )
        small_df.to_csv(small_file, index=False)

        output_file = tmp_path / "ticker_split.parquet"
        with pytest.raises(ValueError, match="No ticker data could be split"):
            split_tickers_train_test(
                input_file=str(small_file),
                output_file=str(output_file),
            )

    def test_split_tickers_train_test_mixed_ticker_sizes(self, tmp_path: Path) -> None:
        """Test split_tickers_train_test with tickers having different sizes."""
        mixed_file = tmp_path / "mixed.csv"
        dates_long = pd.date_range("2020-01-01", periods=50, freq="D")
        dates_short = pd.date_range("2020-01-01", periods=1, freq="D")

        data_list = []
        for date in dates_long:
            data_list.append(
                {
                    "date": date,
                    "ticker": "AAPL",
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1000000.0,
                }
            )
        for date in dates_short:
            data_list.append(
                {
                    "date": date,
                    "ticker": "MSFT",
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 1000000.0,
                }
            )

        mixed_df = pd.DataFrame(data_list)
        mixed_df.to_csv(mixed_file, index=False)

        output_file = tmp_path / "ticker_split.parquet"
        split_tickers_train_test(
            train_ratio=0.8,
            input_file=str(mixed_file),
            output_file=str(output_file),
        )

        split_df = pd.read_parquet(output_file)
        assert split_df["tickers"].nunique() == 1
        assert "AAPL" in split_df["tickers"].values
        assert "MSFT" not in split_df["tickers"].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
