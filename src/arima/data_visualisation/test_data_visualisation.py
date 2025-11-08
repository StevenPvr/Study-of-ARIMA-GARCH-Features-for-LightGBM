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

# Dependencies are mocked in src/conftest.py before imports
from src.arima.data_visualisation.data_visualisation import (
    analyze_residuals_sarima_000,
    plot_acf_pacf,
    plot_rolling_forecast_sarima_000,
    plot_seasonality_for_year,
    plot_weighted_series,
)


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

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.Path")
    @patch("src.arima.data_visualisation.utils.pd.read_csv")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.data_visualisation.logger")
    def test_plot_weighted_series_success(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_read_csv: MagicMock,
        mock_path: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test successful plotting of weighted series."""
        # Setup mocks
        mock_plots_dir.mkdir = MagicMock()
        mock_plots_dir.__truediv__ = MagicMock(return_value=Path("test_output.png"))

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_read_csv.return_value = mock_df

        # Mock plt.subplots to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Execute
        plot_weighted_series(
            data_file="test_data.csv",
            output_file=str(mock_plots_dir / "weighted_log_returns_series.png"),
        )

        # Verify
        mock_read_csv.assert_called_once()
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.Path")
    @patch("src.arima.data_visualisation.utils.pd.read_csv")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.data_visualisation.logger")
    def test_plot_weighted_series_custom_paths(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_read_csv: MagicMock,
        mock_path: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test plotting with custom file paths."""
        mock_plots_dir.mkdir = MagicMock()

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_read_csv.return_value = mock_df

        # Mock plt.subplots to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_weighted_series(data_file="custom_data.csv", output_file="custom_output.png")

        mock_read_csv.assert_called_once_with("custom_data.csv", parse_dates=["date"])
        mock_plt.savefig.assert_called_once_with("custom_output.png", dpi=300, bbox_inches="tight")


class TestPlotAcfPacf:
    """Tests for plot_acf_pacf function."""

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.Path")
    @patch("src.arima.data_visualisation.data_visualisation.plot_acf")
    @patch("src.arima.data_visualisation.data_visualisation.plot_pacf")
    @patch("src.arima.data_visualisation.utils.pd.read_csv")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.data_visualisation.logger")
    def test_plot_acf_pacf_success(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_read_csv: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_path: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test successful plotting of ACF/PACF."""
        # Setup mocks
        mock_plots_dir.mkdir = MagicMock()
        mock_plots_dir.__truediv__ = MagicMock(return_value=Path("test_output.png"))

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "weighted_log_return": np.random.randn(100),
            }
        )
        mock_read_csv.return_value.set_index.return_value = mock_df.set_index("date")

        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        # Execute
        plot_acf_pacf(
            data_file="test_data.csv",
            output_file=str(mock_plots_dir / "acf_pacf.png"),
        )

        # Verify
        mock_read_csv.assert_called_once()
        mock_plt.subplots.assert_called_once()
        mock_plot_acf.assert_called_once()
        mock_plot_pacf.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.Path")
    @patch("src.arima.data_visualisation.data_visualisation.plot_acf")
    @patch("src.arima.data_visualisation.data_visualisation.plot_pacf")
    @patch("src.arima.data_visualisation.utils.pd.read_csv")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.data_visualisation.logger")
    def test_plot_acf_pacf_custom_lags(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_read_csv: MagicMock,
        mock_plot_pacf: MagicMock,
        mock_plot_acf: MagicMock,
        mock_path: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test plotting with custom number of lags."""
        mock_plots_dir.mkdir = MagicMock()

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="D"),
                "weighted_log_return": np.random.randn(50),
            }
        )
        mock_read_csv.return_value.set_index.return_value = mock_df.set_index("date")

        mock_axes = [MagicMock(), MagicMock()]
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)

        plot_acf_pacf(
            data_file="test_data.csv",
            output_file="test_output.png",
            lags=50,
        )

        mock_plot_acf.assert_called_once()
        args, kwargs = mock_plot_acf.call_args
        assert kwargs["lags"] == 50


class TestPlotSeasonalityForYear:
    """Tests for plot_seasonality_for_year function."""

    @patch("src.arima.data_visualisation.utils.plt")
    @patch("src.arima.data_visualisation.utils.Path")
    @patch("src.arima.data_visualisation.utils.seasonal_decompose")
    @patch("src.arima.data_visualisation.utils.pd.read_csv")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.utils.logger")
    def test_plot_seasonality_for_year(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_read_csv: MagicMock,
        mock_decompose: MagicMock,
        mock_path: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        mock_plots_dir.mkdir = MagicMock()
        mock_plots_dir.__truediv__ = MagicMock(return_value=Path("season_year.png"))

        # Mock Path.exists() to return True
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Build a year of business-day data
        idx = pd.bdate_range("2023-01-02", "2023-12-29")
        mock_df = pd.DataFrame({"date": idx, "weighted_log_return": np.random.randn(len(idx))})
        mock_read_csv.return_value.set_index.return_value = mock_df.set_index("date")

        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        mock_res = MagicMock()
        mock_res.seasonal = pd.Series(np.random.randn(len(idx)), index=idx)
        mock_decompose.return_value = mock_res

        plot_seasonality_for_year(
            2023,
            data_file="test_data.csv",
            output_file=str(mock_plots_dir / "seasonal_year_2023.png"),
        )

        mock_decompose.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()


class TestPlotRollingForecastSarima000:
    """Tests for plot_rolling_forecast_sarima_000 function."""

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.plt")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.utils.logger")
    def test_plot_rolling_forecast_success(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_plt_utils: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test successful plotting of rolling forecast."""
        # Setup
        mock_plots_dir.mkdir = MagicMock()
        mock_plots_dir.__truediv__ = MagicMock(return_value=Path("test_output.png"))

        test_series = pd.Series(
            np.random.randn(50),
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        actuals = np.random.randn(50)
        predictions = np.random.randn(50)
        sarima_order = (1, 1, 1)
        metrics = {"RMSE": 0.01, "MAE": 0.008}

        # Mock plt.subplots to return (figure, axes)
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        # Mock plt for utils._save_plot
        mock_plt_utils.savefig = MagicMock()
        mock_plt_utils.close = MagicMock()

        # Execute
        plot_rolling_forecast_sarima_000(
            test_series,
            actuals,
            predictions,
            sarima_order,
            metrics,
            output_file=str(mock_plots_dir / "rolling_forecast_sarima_000.png"),
        )

        # Verify
        mock_plt.subplots.assert_called_once()
        mock_plt_utils.savefig.assert_called_once()
        mock_plt_utils.close.assert_called_once()
        mock_logger.info.assert_called()


class TestAnalyzeResidualsSarima000:
    """Tests for analyze_residuals_sarima_000 function."""

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.plt")
    @patch("src.arima.data_visualisation.utils.scipy_stats.probplot")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.utils.logger")
    def test_analyze_residuals_success(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_probplot: MagicMock,
        mock_plt_utils: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test successful residual analysis."""
        # Setup
        mock_plots_dir.mkdir = MagicMock()
        mock_plots_dir.__truediv__ = MagicMock(return_value=Path("test_output.png"))

        test_series = pd.Series(
            np.random.randn(50),
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        actuals = np.random.randn(50)
        predictions = np.random.randn(50)
        sarima_order = (1, 1, 1)

        # Create a proper mock axes array that supports tuple indexing
        mock_ax00 = MagicMock()
        mock_ax01 = MagicMock()
        mock_ax10 = MagicMock()
        mock_ax11 = MagicMock()
        mock_axes = MockAxesArray(
            [
                [mock_ax00, mock_ax01],
                [mock_ax10, mock_ax11],
            ]
        )
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)
        # Mock plt for utils._save_plot
        mock_plt_utils.savefig = MagicMock()
        mock_plt_utils.close = MagicMock()

        # Execute
        residuals = analyze_residuals_sarima_000(
            test_series,
            actuals,
            predictions,
            sarima_order,
            output_file=str(mock_plots_dir / "residuals_analysis_sarima_000.png"),
        )

        # Verify
        assert residuals is not None
        assert len(residuals) == 50
        mock_plt.subplots.assert_called_once()
        mock_probplot.assert_called_once()
        mock_plt_utils.savefig.assert_called_once()
        mock_plt_utils.close.assert_called_once()
        mock_logger.info.assert_called()

    @patch("src.arima.data_visualisation.data_visualisation.plt")
    @patch("src.arima.data_visualisation.utils.plt")
    @patch("src.arima.data_visualisation.utils.scipy_stats.probplot")
    @patch("src.arima.data_visualisation.utils.SARIMA_DATA_VISU_PLOTS_DIR")
    @patch("src.arima.data_visualisation.utils.logger")
    def test_analyze_residuals_custom_output(
        self,
        mock_logger: MagicMock,
        mock_plots_dir: MagicMock,
        mock_probplot: MagicMock,
        mock_plt_utils: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test residual analysis with custom output path."""
        mock_plots_dir.mkdir = MagicMock()

        test_series = pd.Series(
            np.random.randn(30),
            index=pd.date_range("2020-01-01", periods=30, freq="D"),
        )
        actuals = np.random.randn(30)
        predictions = np.random.randn(30)
        sarima_order = (2, 1, 2)

        # Create a proper mock axes array that supports tuple indexing
        mock_ax00 = MagicMock()
        mock_ax01 = MagicMock()
        mock_ax10 = MagicMock()
        mock_ax11 = MagicMock()
        mock_axes = MockAxesArray(
            [
                [mock_ax00, mock_ax01],
                [mock_ax10, mock_ax11],
            ]
        )
        mock_plt.subplots.return_value = (MagicMock(), mock_axes)
        # Mock plt for utils._save_plot
        mock_plt_utils.savefig = MagicMock()
        mock_plt_utils.close = MagicMock()

        analyze_residuals_sarima_000(
            test_series,
            actuals,
            predictions,
            sarima_order,
            output_file="custom_residuals.png",
        )

        mock_plt_utils.savefig.assert_called_once_with(
            "custom_residuals.png", dpi=300, bbox_inches="tight"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
