"""Unit tests for data_conversion module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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

# Now import pandas (will get the real one if we restored it)
import pandas as pd

# Import and reload the module to ensure it uses real pandas
from src.data_conversion import data_conversion

# Reload the module to pick up the real pandas
importlib.reload(data_conversion)

from src.data_conversion.data_conversion import (
    compute_liquidity_weights,
    compute_log_returns,
    compute_weighted_aggregated_returns,
    compute_weighted_log_returns,
    compute_weighted_prices,
    load_filtered_dataset,
    save_liquidity_weights,
    save_weighted_returns,
)


class TestLoadFilteredDataset:
    """Tests for load_filtered_dataset function."""

    @patch("src.data_conversion.data_conversion.pd.read_parquet")
    @patch("src.data_conversion.data_conversion.pd.read_csv")
    @patch("src.data_conversion.data_conversion.Path")
    @patch("src.data_conversion.data_conversion.logger")
    def test_load_filtered_dataset_success(
        self,
        mock_logger: MagicMock,
        mock_path: MagicMock,
        mock_read_csv: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test successful loading of filtered dataset from explicit path without fallback."""
        mock_df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "ticker": ["AAPL", "AAPL"],
                "open": [100.0, 102.0],
                "close": [101.0, 103.0],
                "volume": [1000000, 1100000],
            }
        )
        # Mock reading from CSV (explicit path), no fallback to parquet
        mock_read_csv.return_value = mock_df.copy(deep=True)
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".csv"
        mock_path.return_value = mock_path_instance

        result = load_filtered_dataset("test.csv")

        mock_path.assert_called_once_with("test.csv")
        mock_read_csv.assert_called_once_with(mock_path_instance)
        assert isinstance(result, pd.DataFrame)
        assert {"date", "ticker", "open", "closing", "volume"}.issubset(result.columns)
        assert len(result) == 2

    @patch("src.data_conversion.data_conversion.Path")
    def test_load_filtered_dataset_file_not_found(self, mock_path: MagicMock) -> None:
        """Test error handling when file does not exist."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            load_filtered_dataset("nonexistent.csv")

    @patch("src.data_conversion.data_conversion.pd.read_parquet")
    @patch("src.data_conversion.data_conversion.pd.read_csv")
    @patch("src.data_conversion.data_conversion.Path")
    def test_load_filtered_dataset_empty_file(
        self, mock_path: MagicMock, mock_read_csv: MagicMock, mock_read_parquet: MagicMock
    ) -> None:
        """Test error handling when explicit parquet file is empty."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".parquet"
        mock_path.return_value = mock_path_instance
        mock_read_parquet.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Loaded dataset is empty"):
            load_filtered_dataset("empty.parquet")

    @patch("src.data_conversion.data_conversion.pd.read_parquet")
    @patch("src.data_conversion.data_conversion.pd.read_csv")
    @patch("src.data_conversion.data_conversion.Path")
    def test_load_filtered_dataset_unsupported_extension(
        self, mock_path: MagicMock, mock_read_csv: MagicMock, mock_read_parquet: MagicMock
    ) -> None:
        """Test error handling when file extension is unsupported."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".txt"
        mock_path.return_value = mock_path_instance

        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_filtered_dataset("test.txt")


class TestComputeLiquidityWeights:
    """Tests for compute_liquidity_weights function."""

    def test_compute_liquidity_weights_success(self, sample_raw_df: pd.DataFrame) -> None:
        """Test successful computation of liquidity weights."""
        result = compute_liquidity_weights(sample_raw_df)

        assert isinstance(result, pd.DataFrame)
        assert "weight" in result.columns
        assert "liquidity_score" in result.columns
        # Check that weights sum to approximately 1.0
        assert abs(result["weight"].sum() - 1.0) < 1e-10

    @pytest.mark.parametrize(
        "df,expected_error",
        [
            (pd.DataFrame(), "Input DataFrame is empty"),
            (pd.DataFrame({"ticker": ["AAPL"], "volume": [1000.0]}), "Missing required columns"),
            (
                pd.DataFrame(
                    {
                        "ticker": ["AAPL", "MSFT"],
                        "volume": [0.0, 0.0],
                        "closing": [0.0, 0.0],
                    }
                ),
                "Sum of liquidity scores is zero",
            ),
        ],
    )
    def test_compute_liquidity_weights_errors(self, df: pd.DataFrame, expected_error: str) -> None:
        """Test error handling for various invalid inputs."""
        with pytest.raises((ValueError, KeyError), match=expected_error):
            compute_liquidity_weights(df)


class TestSaveLiquidityWeights:
    """Tests for save_liquidity_weights function."""

    @patch("src.data_conversion.data_conversion.Path")
    @patch("src.data_conversion.data_conversion.logger")
    def test_save_liquidity_weights_success(
        self, mock_logger: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test successful saving of liquidity weights."""
        liquidity_metrics = pd.DataFrame(
            {
                "mean_volume": [1000000.0],
                "mean_price": [100.0],
                "liquidity_score": [100000000.0],
                "weight": [1.0],
            },
            index=pd.Index(["AAPL"]),
        )
        mock_path_instance = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path_instance.to_csv = MagicMock()
        mock_path.return_value = mock_path_instance

        save_liquidity_weights(liquidity_metrics, "test_weights.csv")

        mock_path.assert_called_once_with("test_weights.csv")
        mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_logger.info.assert_called_once()

    def test_save_liquidity_weights_missing_columns(self) -> None:
        """Test error handling when required columns are missing."""
        liquidity_metrics = pd.DataFrame(
            {
                "mean_volume": [1000000.0],
                "mean_price": [100.0],
            },
            index=pd.Index(["AAPL"]),
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            save_liquidity_weights(liquidity_metrics, "test_weights.csv")


class TestComputeLogReturns:
    """Tests for compute_log_returns function."""

    @patch("src.data_conversion.data_conversion.logger")
    def test_compute_log_returns_success(self, mock_logger: MagicMock) -> None:
        """Test successful computation of log returns."""
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL"],
                "closing": [100.0, 102.0, 104.0],
            }
        )

        result = compute_log_returns(raw_df)

        assert isinstance(result, pd.DataFrame)
        assert "log_return" in result.columns
        assert len(result) == 2  # Two rows after dropping first NaN
        assert result["log_return"].isna().sum() == 0
        mock_logger.info.assert_called_once()

    @pytest.mark.parametrize(
        "df,expected_error",
        [
            (pd.DataFrame(), "Input DataFrame is empty"),
            (pd.DataFrame({"ticker": ["AAPL"]}), "Missing required columns"),
        ],
    )
    def test_compute_log_returns_errors(self, df: pd.DataFrame, expected_error: str) -> None:
        """Test error handling for various invalid inputs."""
        with pytest.raises((ValueError, KeyError), match=expected_error):
            compute_log_returns(df)

    def test_compute_log_returns_zero_or_negative_prices(self) -> None:
        """Test handling of zero or negative prices (should be excluded)."""
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
                "closing": [100.0, 0.0, -10.0, 102.0, 104.0],
            }
        )

        result = compute_log_returns(raw_df)

        assert len(result) >= 1
        assert result["log_return"].notna().all()


class TestComputeWeightedAggregatedReturns:
    """Tests for compute_weighted_aggregated_returns function."""

    @patch("src.data_conversion.data_conversion.logger")
    def test_compute_weighted_aggregated_returns_success(
        self,
        mock_logger: MagicMock,
        sample_returns_df: pd.DataFrame,
        sample_liquidity_metrics: pd.DataFrame,
    ) -> None:
        """Test successful computation of weighted aggregated returns."""
        aggregated, daily_totals = compute_weighted_aggregated_returns(
            sample_returns_df, sample_liquidity_metrics
        )

        assert isinstance(aggregated, pd.DataFrame)
        assert isinstance(daily_totals, pd.DataFrame)
        assert {"weighted_log_return", "date"}.issubset(aggregated.columns)
        assert "weight_sum" in daily_totals.columns
        assert len(aggregated) > 0
        mock_logger.info.assert_called_once()

    @pytest.mark.parametrize(
        "returns_df,liquidity_metrics,expected_error",
        [
            (pd.DataFrame(), pd.DataFrame({"weight": [0.5]}, index=["AAPL"]), "Returns"),
            (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2020-01-01"]),
                        "ticker": ["AAPL"],
                        "log_return": [0.01],
                    }
                ),
                pd.DataFrame(),
                "Liquidity metrics",
            ),
            (
                pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ticker": ["AAPL"]}),
                pd.DataFrame({"weight": [0.5]}, index=["AAPL"]),
                "Missing required columns",
            ),
            (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2020-01-01"]),
                        "ticker": ["AAPL"],
                        "log_return": [0.01],
                    }
                ),
                pd.DataFrame({"other_col": [0.5]}, index=["AAPL"]),
                "Missing required columns",
            ),
        ],
    )
    def test_compute_weighted_aggregated_returns_errors(
        self,
        returns_df: pd.DataFrame,
        liquidity_metrics: pd.DataFrame,
        expected_error: str,
    ) -> None:
        """Test error handling for various invalid inputs."""
        with pytest.raises((ValueError, KeyError), match=expected_error):
            compute_weighted_aggregated_returns(returns_df, liquidity_metrics)

    def test_compute_weighted_aggregated_returns_empty_after_filtering(self) -> None:
        """Test handling when weighted returns are empty after filtering."""
        returns_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "ticker": ["AAPL"],
                "log_return": [0.01],
            }
        )
        liquidity_metrics = pd.DataFrame({"weight": [0.5]}, index=pd.Index(["MSFT"]))

        aggregated, _ = compute_weighted_aggregated_returns(returns_df, liquidity_metrics)

        assert aggregated.empty
        assert {"date", "weighted_log_return"}.issubset(aggregated.columns)


class TestComputeWeightedPrices:
    """Tests for compute_weighted_prices function."""

    @patch("src.data_conversion.data_conversion.logger")
    def test_compute_weighted_prices_success(
        self,
        mock_logger: MagicMock,
        sample_returns_df: pd.DataFrame,
        sample_liquidity_metrics: pd.DataFrame,
        sample_daily_weight_totals: pd.DataFrame,
    ) -> None:
        """Test successful computation of weighted prices."""
        result = compute_weighted_prices(
            sample_returns_df, sample_liquidity_metrics, sample_daily_weight_totals
        )

        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        mock_logger.info.assert_called_once()

    @pytest.mark.parametrize(
        "returns_df,liquidity_metrics,daily_totals,expected_error",
        [
            (
                pd.DataFrame(),
                pd.DataFrame({"weight": [0.5]}, index=["AAPL"]),
                pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "weight_sum": [1.0]}),
                "Returns",
            ),
            (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2020-01-01"]),
                        "ticker": ["AAPL"],
                        "open": [100.0],
                        "closing": [101.0],
                    }
                ),
                pd.DataFrame(),
                pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "weight_sum": [1.0]}),
                "Liquidity metrics",
            ),
            (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2020-01-01"]),
                        "ticker": ["AAPL"],
                        "open": [100.0],
                        "closing": [101.0],
                    }
                ),
                pd.DataFrame({"weight": [0.5]}, index=["AAPL"]),
                pd.DataFrame(),
                "Daily weight totals",
            ),
        ],
    )
    def test_compute_weighted_prices_errors(
        self,
        returns_df: pd.DataFrame,
        liquidity_metrics: pd.DataFrame,
        daily_totals: pd.DataFrame,
        expected_error: str,
    ) -> None:
        """Test error handling for various invalid inputs."""
        with pytest.raises(ValueError, match=expected_error):
            compute_weighted_prices(returns_df, liquidity_metrics, daily_totals)


class TestSaveWeightedReturns:
    """Tests for save_weighted_returns function."""

    @patch("src.data_conversion.data_conversion.save_parquet_and_csv")
    @patch("src.data_conversion.data_conversion.Path")
    @patch("src.data_conversion.data_conversion.logger")
    def test_save_weighted_returns_success(
        self,
        mock_logger: MagicMock,
        mock_path: MagicMock,
        mock_save_parquet_and_csv: MagicMock,
        sample_aggregated_returns: pd.DataFrame,
    ) -> None:
        """Test successful saving of weighted returns."""
        mock_path_instance = MagicMock()
        mock_path_instance.suffix = ".csv"
        mock_path_instance.with_suffix.return_value = MagicMock()
        mock_path.return_value = mock_path_instance

        save_weighted_returns(sample_aggregated_returns, "test_returns.csv")

        mock_path.assert_called_once_with("test_returns.csv")
        mock_save_parquet_and_csv.assert_called_once()
        assert mock_logger.info.call_count >= 1

    @pytest.mark.parametrize(
        "aggregated,expected_error",
        [
            (pd.DataFrame(), "Aggregated DataFrame is empty"),
            (pd.DataFrame({"weighted_log_return": [0.01, 0.02]}), "Missing 'date' column"),
        ],
    )
    def test_save_weighted_returns_errors(
        self, aggregated: pd.DataFrame, expected_error: str
    ) -> None:
        """Test error handling for various invalid inputs."""
        with pytest.raises(ValueError, match=expected_error):
            save_weighted_returns(aggregated, "test_returns.csv")


class TestComputeWeightedLogReturns:
    """Tests for compute_weighted_log_returns orchestrator function."""

    @patch(
        "src.data_conversion.data_conversion.DATASET_FILTERED_PARQUET_FILE", Path("mock.parquet")
    )
    @patch("src.data_conversion.data_conversion.save_weighted_returns")
    @patch("src.data_conversion.data_conversion.compute_weighted_prices")
    @patch("src.data_conversion.data_conversion.compute_weighted_aggregated_returns")
    @patch("src.data_conversion.data_conversion.compute_log_returns")
    @patch("src.data_conversion.data_conversion.compute_liquidity_weights")
    @patch("src.data_conversion.data_conversion.load_filtered_dataset")
    def test_compute_weighted_log_returns_success(
        self,
        mock_load: MagicMock,
        mock_compute_weights: MagicMock,
        mock_compute_returns: MagicMock,
        mock_compute_aggregated: MagicMock,
        mock_compute_prices: MagicMock,
        mock_save_returns: MagicMock,
    ) -> None:
        """Test successful computation of weighted log returns."""
        mock_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "ticker": ["AAPL"],
                "open": [99.5],
                "closing": [100.0],
                "volume": [1_000.0],
            }
        )
        mock_load.return_value = mock_df
        mock_compute_weights.return_value = pd.DataFrame(
            {"weight": [1.0]}, index=pd.Index(["AAPL"])
        )
        # Mock compute_log_returns to return DataFrame with log_return column
        mock_returns_df = mock_df.copy()
        mock_returns_df["log_return"] = [0.01]
        mock_compute_returns.return_value = mock_returns_df
        aggregated_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "weighted_log_return": [0.01],
            }
        )
        daily_totals_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "weight_sum": [1.0],
            }
        )
        mock_compute_aggregated.return_value = (aggregated_df, daily_totals_df)
        mock_compute_prices.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "weighted_open": [100.0],
                "weighted_closing": [101.0],
            }
        )

        compute_weighted_log_returns()

        mock_load.assert_called_once_with(Path("mock.parquet"))
        mock_compute_weights.assert_called_once()
        # We no longer save weights in the orchestrator
        mock_compute_returns.assert_called_once()
        mock_compute_aggregated.assert_called_once()
        mock_compute_prices.assert_called_once()
        mock_save_returns.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
