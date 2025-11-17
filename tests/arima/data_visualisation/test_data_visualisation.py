"""Unit tests for data_visualisation module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Dependencies are mocked in conftest.py before imports
from src.arima.data_visualisation.pre_modeling import (
    plot_acf_pacf,
    plot_stationarity,
    plot_stationarity_timeseries_with_bands,
    plot_weighted_series,
)
from src.arima.stationnarity_check.stationnarity_check import StationarityReport

# Seasonality decomposition constants are unused since legacy plots were removed


class MockAxesArray:
    """Mock class that simulates numpy array indexing for matplotlib axes.

    This allows us to use axes[i, j] syntax while returning MagicMock objects.
    """

    def __init__(self, axes_grid: list[list[MagicMock]]) -> None:
        """Initialize with a 2D grid of MagicMock axes.

        Args:
            axes_grid: 2D list of MagicMock axes objects.
        """
        self._axes = axes_grid

    def __getitem__(self, key: tuple[int, int] | int) -> MagicMock | list[MagicMock]:
        """Enable indexing with tuple (i, j) or single index.

        Args:
            key: Tuple (i, j) or single index.

        Returns:
            MagicMock axes object for tuple key, or list of MagicMock for single index.
        """
        if isinstance(key, tuple):
            i, j = key
            return self._axes[i][j]
        # For single index, return the row (list of MagicMock)
        return self._axes[key]


class TestPlotWeightedSeries:
    """Tests for plot_weighted_series function."""

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_weighted_series_success(
        self,
        mock_logger: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
    ) -> None:
        """Test successful plotting of weighted series."""
        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_load_df.return_value = mock_df

        # Mock create_standard_figure to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        # Execute
        plot_weighted_series(
            data_file="test_data.csv",
            output_file="test_output.png",
        )

        # Verify
        mock_load_df.assert_called_once()
        mock_create_figure.assert_called_once()
        mock_save_plot.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_weighted_series_custom_paths(
        self,
        mock_logger: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test plotting with custom file paths."""

        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_load_df.return_value = mock_df

        # Mock create_standard_figure to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        # Use tmp_path instead of relative path to avoid creating permanent files
        custom_output = tmp_path / "custom_output.png"
        plot_weighted_series(data_file="custom_data.csv", output_file=str(custom_output))

        # Verify load_dataframe was called (it internally uses read_csv/read_parquet)
        mock_load_df.assert_called_once()
        # Verify _save_plot was called with the figure and output path
        mock_save_plot.assert_called_once()
        # Verify the output file path was passed
        call_args = mock_save_plot.call_args
        assert str(custom_output) in str(call_args)


class TestPlotAcfPacf:
    """Tests for plot_acf_pacf function."""

    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.plot_acf")
    @patch("src.arima.data_visualisation.pre_modeling.plot_pacf")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_acf_pacf_success(
        self,
        mock_logger: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_load_df: MagicMock,
        mock_create_figure: MagicMock,
        mock_save_plot: MagicMock,
    ) -> None:
        """Test successful plotting of ACF/PACF."""
        # Setup mocks

        # Mock load_dataframe to return test data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_load_df.return_value = mock_df

        mock_axes = [MagicMock(), MagicMock()]
        mock_create_figure.return_value = (MagicMock(), mock_axes)

        # Execute
        plot_acf_pacf(
            data_file="test_data.csv",
            output_file="test_output.png",
        )

        # Verify
        mock_load_df.assert_called_once()
        mock_create_figure.assert_called_once()
        mock_plot_acf.assert_called_once()
        mock_plot_pacf.assert_called_once()
        mock_save_plot.assert_called_once()

    @patch("src.arima.data_visualisation.pre_modeling.plt")
    @patch("src.arima.data_visualisation.data_loading.load_dataframe")
    @patch("src.arima.data_visualisation.pre_modeling.plot_acf")
    @patch("src.arima.data_visualisation.pre_modeling.plot_pacf")
    @patch("src.arima.data_visualisation.pre_modeling.logger")
    def test_plot_acf_pacf_custom_lags(
        self,
        mock_logger: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_load_df: MagicMock,
        mock_plt: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test plotting with custom number of lags.

        When requesting more lags than available (50 lags with 50 data points),
        the function should automatically adjust to the maximum possible (49 lags).
        """

        # Create 50 data points - can only compute up to 49 lags (need lags+1 points)
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_load_df.return_value = mock_df

        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        # Use tmp_path instead of relative path to avoid creating permanent files
        test_output = tmp_path / "test_output.png"
        plot_acf_pacf(
            data_file="test_data.csv",
            output_file=str(test_output),
            lags=50,  # Request 50 lags, but will be adjusted to 49
        )

        mock_plot_acf.assert_called_once()
        args, kwargs = mock_plot_acf.call_args
        # With 50 points, max lags is 49 (need lags+1 points to compute lag 'lags')
        assert kwargs["lags"] == 49
        # Verify warning was logged about adjustment
        mock_logger.warning.assert_called_once()


class TestPlotStationarity:
    """Tests for plot_stationarity function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling.plot_stationarity_timeseries_with_bands")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path")
    def test_plot_stationarity_success(
        self,
        mock_path: MagicMock,
        mock_mkdir: MagicMock,
        mock_plot_timeseries: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        # Mock Path to return a path object
        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.__truediv__ = MagicMock()
        mock_path_instance.mkdir = mock_mkdir

        plot_stationarity("input.csv", "output_dir", rolling_window=10, alpha=0.05)

        # Verify that the plotting function is called
        mock_plot_timeseries.assert_called_once()

        # Verify that mkdir is called to create output directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify logging
        assert mock_logger.info.call_count >= 2


class TestPlotStationarityTimeseriesWithBands:
    """Tests for plot_stationarity_timeseries_with_bands function."""

    @patch("src.arima.data_visualisation.pre_modeling.logger")
    @patch("src.arima.data_visualisation.pre_modeling._save_plot")
    @patch("src.arima.data_visualisation.pre_modeling.add_stationarity_text_box")
    @patch("src.arima.data_visualisation.pre_modeling.format_stationarity_test_text")
    @patch("src.arima.data_visualisation.pre_modeling.plot_stationarity_series_with_bands")
    @patch("src.arima.data_visualisation.pre_modeling.evaluate_stationarity")
    @patch("src.arima.data_visualisation.pre_modeling.load_and_validate_data")
    @patch("src.arima.data_visualisation.pre_modeling.create_standard_figure")
    def test_plot_stationarity_timeseries_with_bands_success(
        self,
        mock_create_figure: MagicMock,
        mock_load: MagicMock,
        mock_evaluate: MagicMock,
        mock_plot_series: MagicMock,
        mock_format_text: MagicMock,
        mock_add_box: MagicMock,
        mock_save: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        dates = pd.date_range("2020-01-01", periods=260, freq="B")
        values = np.linspace(-0.1, 0.1, len(dates))
        dataframe = pd.DataFrame({"weighted_log_return": values}, index=dates)
        dataframe.index.name = "date"
        mock_load.return_value = dataframe

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_create_figure.return_value = (mock_fig, mock_ax)

        report = StationarityReport(
            stationary=True,
            alpha=0.05,
            adf={
                "statistic": -2.5,
                "p_value": 0.04,
                "lags": 2,
                "nobs": len(dates),
                "critical_values": {},
            },
            kpss={
                "statistic": 0.12,
                "p_value": 0.1,
                "lags": 3,
                "nobs": len(dates),
                "critical_values": {},
            },
            zivot_andrews=None,
        )
        mock_evaluate.return_value = report

        plot_stationarity_timeseries_with_bands(
            "input.csv", "output.png", rolling_window=10, alpha=0.05
        )

        mock_plot_series.assert_called_once()
        mock_add_box.assert_called_once()
        mock_save.assert_called_once_with("output.png")


# Legacy seasonality decomposition plot tests removed.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
