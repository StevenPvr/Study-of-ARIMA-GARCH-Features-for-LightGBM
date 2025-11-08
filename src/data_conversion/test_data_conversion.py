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
    compute_liquidity_weights_timevarying,
    compute_log_returns,
    compute_weighted_aggregated_returns,
    compute_weighted_log_returns,
    compute_weighted_log_returns_no_lookahead,
    compute_weighted_prices,
    load_filtered_dataset,
    save_liquidity_weights,
    save_weighted_returns,
)


class TestLoadFilteredDataset:
    """Tests for load_filtered_dataset function."""

    @patch("src.data_conversion.data_conversion.pd.read_csv")
    @patch("src.data_conversion.data_conversion.Path")
    @patch("src.data_conversion.data_conversion.logger")
    def test_load_filtered_dataset_success(
        self, mock_logger: MagicMock, mock_path: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test successful loading of filtered dataset."""
        mock_df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "ticker": ["AAPL", "AAPL"],
                "open": [100.0, 102.0],
                "closing": [101.0, 103.0],
                "volume": [1000000, 1100000],
            }
        )
        # Return a deep copy to avoid any issues with pandas internal operations
        mock_read_csv.return_value = mock_df.copy(deep=True)

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        result = load_filtered_dataset("test.csv")

        mock_path.assert_called_once_with("test.csv")
        mock_path_instance.exists.assert_called_once()
        mock_read_csv.assert_called_once_with(mock_path_instance)
        # Verify result is a DataFrame-like object with expected structure
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "ticker" in result.columns
        assert len(result) == 2
        mock_logger.info.assert_called_once()

    @patch("src.data_conversion.data_conversion.Path")
    def test_load_filtered_dataset_file_not_found(self, mock_path: MagicMock) -> None:
        """Test error handling when file does not exist."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            load_filtered_dataset("nonexistent.csv")

    @patch("src.data_conversion.data_conversion.pd.read_csv")
    @patch("src.data_conversion.data_conversion.Path")
    def test_load_filtered_dataset_empty_file(
        self, mock_path: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test error handling when file is empty."""
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_read_csv.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Loaded dataset is empty"):
            load_filtered_dataset("empty.csv")


class TestComputeLiquidityWeights:
    """Tests for compute_liquidity_weights function."""

    def test_compute_liquidity_weights_success(self) -> None:
        """Test successful computation of liquidity weights."""
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "volume": [1000000, 1100000, 2000000, 2100000],
                "closing": [100.0, 102.0, 200.0, 204.0],
            }
        )

        result = compute_liquidity_weights(raw_df)

        # Verify result structure
        assert hasattr(result, "columns")
        assert hasattr(result, "sum")
        # Check that weights sum to approximately 1.0
        weight_sum = result["weight"].sum()
        assert abs(weight_sum - 1.0) < 1e-10

    def test_compute_liquidity_weights_empty_dataframe(self) -> None:
        """Test error handling when DataFrame is empty."""
        raw_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            compute_liquidity_weights(raw_df)

    def test_compute_liquidity_weights_missing_columns(self) -> None:
        """Test error handling when required columns are missing."""
        raw_df = pd.DataFrame({"ticker": ["AAPL"], "volume": [1000.0]})

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_liquidity_weights(raw_df)

    def test_compute_liquidity_weights_zero_liquidity(self) -> None:
        """Test error handling when liquidity scores sum to zero."""
        # Create DataFrame with all zeros to ensure zero liquidity score
        raw_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "volume": [0.0, 0.0, 0.0, 0.0],
                "closing": [0.0, 0.0, 0.0, 0.0],
            }
        )

        with pytest.raises(ValueError, match="Sum of liquidity scores is zero"):
            compute_liquidity_weights(raw_df)


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

        # Verify result structure - should be a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert "log_return" in result.columns
        # Verify log returns were computed (should have 2 rows after dropping NaN)
        # Note: first row has NaN log return, so it's dropped
        assert len(result) == 2  # Two rows after dropping NaN from first row
        # Verify all log returns are non-null (since NaN rows were dropped)
        assert result["log_return"].isna().sum() == 0
        mock_logger.info.assert_called_once()

    def test_compute_log_returns_empty_dataframe(self) -> None:
        """Test error handling when DataFrame is empty."""
        raw_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            compute_log_returns(raw_df)

    def test_compute_log_returns_missing_columns(self) -> None:
        """Test error handling when required columns are missing."""
        raw_df = pd.DataFrame({"ticker": ["AAPL"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_log_returns(raw_df)


class TestTimeVaryingWeights:
    """Tests for no-look-ahead time-varying weights."""

    def test_compute_liquidity_weights_timevarying_shift(self) -> None:
        import pandas as pd

        # Construct simple series where liquidity equals day index for clarity
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])  # 3 days
        raw_df = pd.DataFrame(
            {
                "date": dates.tolist() * 1,
                "ticker": ["AAA", "AAA", "AAA"],
                "closing": [1.0, 2.0, 3.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        # window=2, weight on day 2 should use day1; day3 uses mean(day1,day2)
        weights = compute_liquidity_weights_timevarying(raw_df, window=2, min_periods=1)
        # Expect two weights (no weight on first day because of shift)
        assert len(weights) == 2
        # Day 2024-01-02 weight equals liquidity of 2024-01-01: 1*1=1
        w_day2 = float(weights.loc[weights["date"] == pd.Timestamp("2024-01-02"), "weight"].iloc[0])
        assert abs(w_day2 - 1.0) < 1e-12
        # Day 2024-01-03 weight equals mean(liq day1, day2) = mean(1, 2) = 1.5
        w_day3 = float(weights.loc[weights["date"] == pd.Timestamp("2024-01-03"), "weight"].iloc[0])
        assert abs(w_day3 - 1.5) < 1e-12

    @patch("src.data_conversion.data_conversion.save_weighted_returns")
    @patch("src.data_conversion.data_conversion.compute_weighted_prices")
    @patch("src.data_conversion.data_conversion.compute_weighted_aggregated_returns")
    @patch("src.data_conversion.data_conversion.compute_log_returns")
    @patch("src.data_conversion.data_conversion.compute_liquidity_weights_timevarying")
    @patch("src.data_conversion.data_conversion.load_filtered_dataset")
    def test_compute_weighted_log_returns_no_lookahead_orchestrator(
        self,
        mock_load: MagicMock,
        mock_tv_weights: MagicMock,
        mock_compute_returns: MagicMock,
        mock_compute_aggregated: MagicMock,
        mock_compute_prices: MagicMock,
        mock_save_returns: MagicMock,
    ) -> None:
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "ticker": ["AAPL"],
                "closing": [100.0],
                "volume": [1_000.0],
            }
        )
        mock_load.return_value = mock_df
        mock_tv_weights.return_value = pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-02"]), "ticker": ["AAPL"], "weight": [1.0]}
        )
        mock_compute_returns.return_value = mock_df
        aggregated_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"]),
                "weighted_log_return": [0.01],
            }
        )
        daily_totals_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"]),
                "weight_sum": [1.0],
            }
        )
        mock_compute_aggregated.return_value = (aggregated_df, daily_totals_df)
        mock_compute_prices.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"]),
                "weighted_open": [100.0],
                "weighted_closing": [101.0],
            }
        )

        compute_weighted_log_returns_no_lookahead()

        mock_load.assert_called_once()
        mock_tv_weights.assert_called_once()
        mock_compute_returns.assert_called_once()
        mock_compute_aggregated.assert_called_once()
        mock_compute_prices.assert_called_once()
        mock_save_returns.assert_called_once()


class TestComputeWeightedAggregatedReturns:
    """Tests for compute_weighted_aggregated_returns function."""

    @patch("src.data_conversion.data_conversion.logger")
    def test_compute_weighted_aggregated_returns_success(self, mock_logger: MagicMock) -> None:
        """Test successful computation of weighted aggregated returns."""
        returns_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "ticker": ["AAPL", "MSFT"],
                "log_return": [0.01, 0.02],
            }
        )
        liquidity_metrics = pd.DataFrame({"weight": [0.5, 0.5]}, index=pd.Index(["AAPL", "MSFT"]))

        aggregated, daily_totals = compute_weighted_aggregated_returns(
            returns_df, liquidity_metrics
        )

        # Verify result structure - all checks combined
        expected_columns = {"weighted_log_return", "date"}
        assert (
            isinstance(aggregated, pd.DataFrame)
            and isinstance(daily_totals, pd.DataFrame)
            and expected_columns.issubset(aggregated.columns)
            and "weight_sum" in daily_totals.columns
            and len(aggregated) == 1
            and len(daily_totals) == 1
        )
        mock_logger.info.assert_called_once()


class TestComputeWeightedPrices:
    """Tests for compute_weighted_prices function."""

    @patch("src.data_conversion.data_conversion.logger")
    def test_compute_weighted_prices_success(self, mock_logger: MagicMock) -> None:
        """Test successful computation of weighted prices."""
        returns_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "ticker": ["AAPL", "MSFT"],
                "open": [100.0, 200.0],
                "closing": [101.0, 201.0],
            }
        )
        liquidity_metrics = pd.DataFrame({"weight": [0.5, 0.5]}, index=pd.Index(["AAPL", "MSFT"]))
        daily_weight_totals = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "weight_sum": [1.0, 1.0],
            },
            index=pd.RangeIndex(2),
        )

        result = compute_weighted_prices(returns_df, liquidity_metrics, daily_weight_totals)

        # Verify result structure
        assert hasattr(result, "columns")
        mock_logger.info.assert_called_once()


class TestSaveWeightedReturns:
    """Tests for save_weighted_returns function."""

    @patch("src.data_conversion.data_conversion.Path")
    @patch("src.data_conversion.data_conversion.logger")
    def test_save_weighted_returns_success(
        self, mock_logger: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test successful saving of weighted returns."""
        aggregated = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "weighted_log_return": [0.01, 0.02],
            }
        )
        mock_path_instance = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path_instance.to_csv = MagicMock()
        mock_path.return_value = mock_path_instance

        save_weighted_returns(aggregated, "test_returns.csv")

        mock_path.assert_called_once_with("test_returns.csv")
        mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert mock_logger.info.call_count == 2


class TestComputeWeightedLogReturns:
    """Tests for compute_weighted_log_returns orchestrator function."""

    @patch("src.data_conversion.data_conversion.save_weighted_returns")
    @patch("src.data_conversion.data_conversion.compute_weighted_prices")
    @patch("src.data_conversion.data_conversion.compute_weighted_aggregated_returns")
    @patch("src.data_conversion.data_conversion.compute_log_returns")
    @patch("src.data_conversion.data_conversion.save_liquidity_weights")
    @patch("src.data_conversion.data_conversion.compute_liquidity_weights")
    @patch("src.data_conversion.data_conversion.load_filtered_dataset")
    def test_compute_weighted_log_returns_success(
        self,
        mock_load: MagicMock,
        mock_compute_weights: MagicMock,
        mock_save_weights: MagicMock,
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
                "closing": [100.0],
            }
        )
        mock_load.return_value = mock_df
        mock_compute_weights.return_value = pd.DataFrame(
            {"weight": [1.0]}, index=pd.Index(["AAPL"])
        )
        mock_compute_returns.return_value = mock_df
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

        mock_load.assert_called_once()
        mock_compute_weights.assert_called_once()
        mock_save_weights.assert_called_once()
        mock_compute_returns.assert_called_once()
        mock_compute_aggregated.assert_called_once()
        mock_compute_prices.assert_called_once()
        mock_save_returns.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
