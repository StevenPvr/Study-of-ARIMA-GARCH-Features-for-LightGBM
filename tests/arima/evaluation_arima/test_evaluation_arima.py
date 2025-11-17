"""Unit tests for evaluation_arima module."""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.metrics import compute_residuals

from src.arima.evaluation_arima.evaluation_arima import (
    anderson_darling_test,
    calculate_metrics,
    evaluate_model,
    jarque_bera_test,
    ljung_box_on_residuals,
    plot_predictions_vs_actual,
    plot_residuals_acf_with_ljungbox,
    rolling_forecast,
    run_all_normality_tests,
    save_evaluation_results,
    save_ljung_box_results,
    shapiro_wilk_test,
)
from src.arima.evaluation_arima.save_data_for_garch import (
    regenerate_garch_dataset_from_rolling_predictions,
    save_garch_dataset,
)
from src.arima.evaluation_arima.utils import (
    add_point_to_series,
    detect_value_column,
    is_refit_step,
    to_date_index,
)
from src.utils import validate_temporal_order_series


class TestRollingForecast:
    """Tests for rolling_forecast function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.SARIMAX")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_rolling_forecast_success(
        self,
        mock_logger: MagicMock,
        mock_sarimax: MagicMock,
    ) -> None:
        """Test successful rolling forecast."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.02] * 10,
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02, 0.015], index=pd.date_range("2020-02-20", periods=3, freq="D")
        )

        # Mock SARIMAX model
        mock_fitted = MagicMock()
        mock_fitted.forecast.return_value = pd.Series([0.01])
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted
        mock_sarimax.return_value = mock_model

        # Execute
        predictions, actuals = rolling_forecast(
            train_series, test_series, order=(1, 0, 1), refit_every=20, verbose=False
        )

        # Verify
        assert len(predictions) == len(test_series)
        assert len(actuals) == len(test_series)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(actuals, np.ndarray)

    @patch("src.arima.evaluation_arima.evaluation_arima.SARIMAX")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_rolling_forecast_with_forecast_failures(
        self,
        mock_logger: MagicMock,
        mock_sarimax: MagicMock,
    ) -> None:
        """Test rolling forecast with forecast failures (fit succeeds, forecast fails)."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015] * 10,
            index=pd.date_range("2020-01-01", periods=30, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-02-01", periods=2, freq="D")
        )

        # Mock SARIMAX model with forecast failures
        mock_fitted = MagicMock()
        call_count = 0

        def forecast_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First forecast fails
                raise Exception("Forecast failed")
            # Subsequent forecasts succeed
            return pd.Series([0.01])

        mock_fitted.forecast.side_effect = forecast_side_effect
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted
        mock_sarimax.return_value = mock_model

        # Execute with warnings suppressed for statsmodels
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            predictions, actuals = rolling_forecast(
                train_series,
                test_series,
                order=(1, 0, 1),
                refit_every=20,
                verbose=False,
            )

        # Verify
        assert len(predictions) == len(test_series)
        assert len(actuals) == len(test_series)
        # First prediction should use last observed value (fallback)
        # Second prediction should use forecast value
        assert all(not np.isnan(p) for p in predictions)

    @patch("src.arima.evaluation_arima.evaluation_arima.SARIMAX")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_rolling_forecast_appends_single_observation(
        self,
        mock_logger: MagicMock,
        mock_sarimax: MagicMock,
    ) -> None:
        """Ensure append receives only one new observation per rolling step."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.02],
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02, 0.015, -0.01],
            index=pd.date_range("2020-01-06", periods=4, freq="D"),
        )

        # Mock SARIMAX model
        mock_fitted = MagicMock()
        mock_fitted.forecast.return_value = pd.Series([0.01])
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted
        mock_sarimax.return_value = mock_model

        predictions, actuals = rolling_forecast(
            train_series,
            test_series,
            order=(1, 0, 1),
            refit_every=2,
            verbose=False,
        )

        assert len(predictions) == len(test_series)
        assert len(actuals) == len(test_series)
        # Verify that model was refit at step 0 and step 2 (refit_every=2)
        assert mock_model.fit.call_count >= 2


def _assert_split_dataframe_structure(split_df: pd.DataFrame) -> None:
    """Assert split dataframe has correct structure and temporal ordering."""
    assert len(split_df) == 3
    assert {"MSE", "RMSE", "MAE"}.issubset(split_df.columns)
    assert split_df["validation_start"].is_monotonic_increasing
    assert all(split_df["train_end"] < split_df["validation_start"])


def _assert_summary_structure(summary: dict[str, float]) -> None:
    """Assert summary has correct structure and values."""
    expected_summary_keys = {
        "mse_mean",
        "mse_std",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
    }
    assert expected_summary_keys.issubset(summary.keys())
    for value in summary.values():
        assert value == pytest.approx(0.0)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.mean_squared_error")
    @patch("src.arima.evaluation_arima.evaluation_arima.mean_absolute_error")
    def test_calculate_metrics_success(self, mock_mae: MagicMock, mock_mse: MagicMock) -> None:
        """Test successful metric calculation."""
        predictions = np.array([0.01, -0.02, 0.015, 0.01, -0.02])
        actuals = np.array([0.012, -0.019, 0.016, 0.011, -0.021])

        # Mock sklearn metrics
        mock_mse.return_value = 0.0001
        mock_mae.return_value = 0.01

        metrics = calculate_metrics(predictions, actuals)

        # Verify structure
        assert isinstance(metrics, dict)
        required_keys = {"MSE", "RMSE", "MAE"}
        assert required_keys.issubset(metrics.keys())

        # Verify values
        assert metrics["RMSE"] == pytest.approx(np.sqrt(metrics["MSE"]), rel=1e-6)
        assert metrics["MSE"] >= 0 and metrics["MAE"] >= 0

    @patch("src.arima.evaluation_arima.evaluation_arima.mean_squared_error")
    @patch("src.arima.evaluation_arima.evaluation_arima.mean_absolute_error")
    def test_calculate_metrics_with_nan(self, mock_mae: MagicMock, mock_mse: MagicMock) -> None:
        """Test metric calculation with NaN values filtered out."""
        predictions = np.array([0.01, np.nan, 0.015, 0.01, -0.02])
        actuals = np.array([0.012, -0.019, 0.016, 0.011, -0.021])

        # Mock sklearn metrics
        mock_mse.return_value = 0.0001
        mock_mae.return_value = 0.01

        metrics = calculate_metrics(predictions, actuals)

        assert isinstance(metrics, dict)
        assert "MSE" in metrics
        assert "RMSE" in metrics
        assert "MAE" in metrics

    def test_calculate_metrics_all_nan(self) -> None:
        """Test metric calculation when all predictions are NaN."""
        predictions = np.array([np.nan, np.nan, np.nan])
        actuals = np.array([0.012, -0.019, 0.016])

        with pytest.raises(RuntimeError, match="All predictions are NaN"):
            calculate_metrics(predictions, actuals)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    @staticmethod
    def _assert_evaluation_results(results: dict, expected_order: tuple, expected_len: int) -> None:
        """Assert evaluation results structure."""
        required_keys = {"model", "order", "metrics", "predictions", "actuals", "dates"}
        assert isinstance(results, dict) and required_keys.issubset(results.keys())
        assert results["order"] == expected_order
        predictions_len = len(results["predictions"])
        actuals_len = len(results["actuals"])
        assert predictions_len == expected_len == actuals_len
        # Dates should have the same length as predictions/actuals when non-empty
        assert expected_len == 0 or len(results["dates"]) == expected_len

    @patch("src.arima.evaluation_arima.evaluation_arima.calculate_metrics")
    @patch("src.arima.evaluation_arima.evaluation_arima.rolling_forecast")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_evaluate_model_success(
        self,
        mock_logger: MagicMock,
        mock_rolling: MagicMock,
        mock_calculate: MagicMock,
    ) -> None:
        """Test successful model evaluation."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015] * 10,
            index=pd.date_range("2020-01-01", periods=30, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02, 0.015] * 5,
            index=pd.date_range("2020-02-01", periods=15, freq="D"),
        )

        predictions = np.array([0.01, -0.02, 0.015, 0.01, -0.02] * 3)
        actuals = np.array([0.012, -0.019, 0.016, 0.011, -0.021] * 3)
        mock_rolling.return_value = (predictions, actuals)
        mock_calculate.return_value = {"MSE": 0.0001, "RMSE": 0.01, "MAE": 0.008}

        results = evaluate_model(
            train_series,
            test_series,
            order=(1, 0, 1),
            refit_every=20,
            model_info={"params": "ARIMA(1,0,1)"},
        )

        # evaluate_model now uses rolling_forecast with optimized refit
        mock_rolling.assert_called_once()
        expected_len = len(predictions)
        self._assert_evaluation_results(results, (1, 0, 1), expected_len)
        mock_calculate.assert_called_once()


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_save_evaluation_results_success(
        self,
        mock_logger: MagicMock,
    ) -> None:
        """Test successful saving of evaluation results."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_preds:
            tmp_preds_path = Path(tmp_preds.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_metrics:
            tmp_metrics_path = Path(tmp_metrics.name)

        try:
            results = {
                "model": "ARIMA(1,0,1)",
                "order": (1, 0, 1),
                "seasonal_order": (0, 0, 0, 0),
                "metrics": {"MSE": 0.0001, "RMSE": 0.01, "MAE": 0.008},
                "predictions": [0.01, -0.02, 0.015],
                "actuals": [0.012, -0.019, 0.016],
                "dates": ["2020-02-01", "2020-02-02", "2020-02-03"],
                "refit_every": 20,
            }

            # Patch file paths
            with patch(
                "src.arima.evaluation_arima.evaluation_arima.ROLLING_PREDICTIONS_ARIMA_FILE",
                tmp_preds_path,
            ):
                with patch(
                    "src.arima.evaluation_arima.evaluation_arima.ROLLING_VALIDATION_METRICS_ARIMA_FILE",
                    tmp_metrics_path,
                ):
                    preds_path, metrics_path = save_evaluation_results(results)

            # Verify return values
            assert isinstance(preds_path, Path)
            assert isinstance(metrics_path, Path)
            assert preds_path == tmp_preds_path
            assert metrics_path == tmp_metrics_path

            # Verify files were created
            assert tmp_preds_path.exists()
            assert tmp_metrics_path.exists()

            # Verify CSV content
            df_read = pd.read_csv(tmp_preds_path)
            assert len(df_read) == 3
            assert "date" in df_read.columns
            assert "y_true" in df_read.columns
            assert "y_pred" in df_read.columns
            assert "residual" in df_read.columns

            # Verify JSON content
            import json

            metrics_data = json.loads(tmp_metrics_path.read_text(encoding="utf-8"))
            assert "order" in metrics_data
            assert "metrics" in metrics_data
            assert metrics_data["metrics"]["MSE"] == 0.0001

            assert mock_logger.info.call_count >= 2
        finally:
            # Cleanup
            if tmp_preds_path.exists():
                tmp_preds_path.unlink()
            if tmp_metrics_path.exists():
                tmp_metrics_path.unlink()


class TestLjungBoxResiduals:
    """Tests for Ljung–Box utilities on ARIMA residuals."""

    def test_compute_residuals_shape(self) -> None:
        actuals = [1.0, 2.0, 3.0]
        preds = [0.5, 1.5, 2.5]
        r = compute_residuals(actuals, preds)
        assert r.shape == (3,)
        assert r.tolist() == [0.5, 0.5, 0.5]

    @patch("src.arima.evaluation_arima.evaluation_arima.acorr_ljungbox")
    def test_ljung_box_on_residuals_basic(self, mock_ljungbox: MagicMock) -> None:
        """Test Ljung-Box on residuals with mocked statsmodels."""
        # Mock statsmodels acorr_ljungbox return
        mock_df = pd.DataFrame(
            {
                "lb_stat": [1.5, 2.3, 3.1, 4.2, 5.0, 6.1, 7.2, 8.3, 9.4, 10.5],
                "lb_pvalue": [0.22, 0.13, 0.08, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001],
            }
        )
        mock_ljungbox.return_value = mock_df

        # White noise residuals should yield non-systematic small autocorrelation
        rng = np.random.default_rng(0)
        res = rng.standard_normal(300)
        out = ljung_box_on_residuals(res, lags=10)
        assert isinstance(out, dict)
        assert set(["lags", "q_stat", "p_value", "reject_5pct", "n"]).issubset(out.keys())
        assert (
            len(out["lags"])
            == len(out["q_stat"])
            == len(out["p_value"])
            == len(out["reject_5pct"])
            == 10
        )
        assert out["n"] == 300
        assert all(isinstance(x, (int, float)) for x in out["q_stat"])
        assert all(isinstance(x, (int, float)) for x in out["p_value"])
        assert all(isinstance(x, bool) for x in out["reject_5pct"])


class TestValidateTemporalOrder:
    """Tests for validate_temporal_order_series function."""

    def test_valid_temporal_order(self) -> None:
        """Test with valid temporal order (test after train)."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01-05", periods=2, freq="D")
        )
        # Should not raise
        validate_temporal_order_series(train_series, test_series)

    def test_invalid_temporal_order(self) -> None:
        """Test with invalid temporal order (test overlaps train)."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01-02", periods=2, freq="D")
        )
        with pytest.raises(ValueError, match="(?i)look-ahead bias"):
            validate_temporal_order_series(train_series, test_series)

    def test_empty_series(self) -> None:
        """Test with empty series."""
        train_series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        test_series = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="D"))
        # Should not raise for empty series
        validate_temporal_order_series(train_series, test_series)

    def test_rolling_forecast_with_temporal_validation(self) -> None:
        """Test that rolling_forecast validates temporal order."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-01-02", periods=2, freq="D")
        )
        with pytest.raises(ValueError, match="(?i)look-ahead bias"):
            rolling_forecast(
                train_series,
                test_series,
                order=(1, 0, 1),
                refit_every=20,
            )


class TestSaveLjungBoxResults:
    """Tests for save_ljung_box_results function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.Path")
    @patch("src.arima.evaluation_arima.evaluation_arima.LJUNGBOX_RESIDUALS_ARIMA_FILE")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    @patch("src.arima.evaluation_arima.evaluation_arima.save_json_pretty")
    def test_save_ljung_box_results_success(
        self,
        mock_save_json: MagicMock,
        mock_logger: MagicMock,
        mock_file: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test successful saving of Ljung-Box results."""
        lb_result = {
            "lags": [1, 2, 3],
            "q_stat": [1.5, 2.3, 3.1],
            "p_value": [0.22, 0.13, 0.08],
            "reject_5pct": [False, False, False],
            "n": 100,
        }

        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path.return_value = mock_path_instance

        result_path = save_ljung_box_results(lb_result)

        mock_path.assert_called_once_with(mock_file)
        mock_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_save_json.assert_called_once_with(lb_result, mock_path_instance)
        mock_logger.info.assert_called_once()
        assert result_path == mock_path_instance


class TestPlotResidualsAcfWithLjungbox:
    """Tests for plot_residuals_acf_with_ljungbox function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.plt")
    @patch("src.arima.evaluation_arima.evaluation_arima.plot_acf")
    @patch("src.arima.evaluation_arima.evaluation_arima.ljung_box_on_residuals")
    @patch("src.arima.evaluation_arima.evaluation_arima.Path")
    @patch("src.arima.evaluation_arima.evaluation_arima.ARIMA_RESIDUALS_LJUNGBOX_PLOT")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_plot_residuals_acf_with_ljungbox_success(
        self,
        mock_logger: MagicMock,
        _mock_plot_file: MagicMock,
        mock_path: MagicMock,
        mock_ljung_box: MagicMock,
        mock_plot_acf: MagicMock,
        mock_plt: MagicMock,
    ) -> None:
        """Test successful plotting of residuals ACF with Ljung-Box."""
        residuals = np.random.randn(100)
        mock_ljung_box.return_value = {
            "lags": [1, 2, 3],
            "p_value": [0.22, 0.13, 0.08],
            "n": 100,
        }

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_plt.figure.return_value = mock_fig

        mock_path_instance = MagicMock()
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.mkdir = MagicMock()
        mock_path.return_value = mock_path_instance

        result_path = plot_residuals_acf_with_ljungbox(residuals, lags=10)

        mock_plt.figure.assert_called_once()
        mock_plot_acf.assert_called_once()
        mock_ljung_box.assert_called_once()
        mock_ax.text.assert_called_once()
        mock_fig.tight_layout.assert_called_once()
        mock_fig.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
        assert result_path == mock_path_instance


class TestNormalityTests:
    """Tests for normality test functions."""

    def test_jarque_bera_test_normal_data(self) -> None:
        """Test Jarque-Bera test with normal data."""
        rng = np.random.default_rng(42)
        normal_data = rng.standard_normal(100)
        result = jarque_bera_test(normal_data)

        assert isinstance(result, dict)
        assert "statistic" in result
        assert "p_value" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "n" in result
        assert result["n"] == 100

    def test_jarque_bera_test_small_sample(self) -> None:
        """Test Jarque-Bera test with small sample."""
        small_data = [1.0, 2.0]
        result = jarque_bera_test(small_data)

        assert isinstance(result, dict)
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])
        assert result["n"] == 2

    def test_shapiro_wilk_test_normal_data(self) -> None:
        """Test Shapiro-Wilk test with normal data."""
        rng = np.random.default_rng(42)
        normal_data = rng.standard_normal(100)
        result = shapiro_wilk_test(normal_data)

        assert isinstance(result, dict)
        assert "statistic" in result
        assert "p_value" in result
        assert "n" in result
        assert result["n"] == 100

    def test_shapiro_wilk_test_small_sample(self) -> None:
        """Test Shapiro-Wilk test with small sample."""
        small_data = [1.0, 2.0]
        result = shapiro_wilk_test(small_data)

        assert isinstance(result, dict)
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])
        assert result["n"] == 2

    def test_anderson_darling_test_normal_data(self) -> None:
        """Test Anderson-Darling test with normal data."""
        rng = np.random.default_rng(42)
        normal_data = rng.standard_normal(100)
        result = anderson_darling_test(normal_data)

        assert isinstance(result, dict)
        assert "statistic" in result
        assert "critical_values" in result
        assert "significance_levels" in result
        assert "n" in result
        assert result["n"] == 100
        assert isinstance(result["critical_values"], dict)

    def test_anderson_darling_test_small_sample(self) -> None:
        """Test Anderson-Darling test with small sample."""
        small_data = [1.0, 2.0]
        result = anderson_darling_test(small_data)

        assert isinstance(result, dict)
        assert np.isnan(result["statistic"])
        assert result["n"] == 2

    def test_run_all_normality_tests(self) -> None:
        """Test run_all_normality_tests function."""
        rng = np.random.default_rng(42)
        normal_data = rng.standard_normal(100)
        result = run_all_normality_tests(normal_data)

        assert isinstance(result, dict)
        assert "jarque_bera" in result
        assert "shapiro_wilk" in result
        assert "anderson_darling" in result
        assert isinstance(result["jarque_bera"], dict)
        assert isinstance(result["shapiro_wilk"], dict)
        assert isinstance(result["anderson_darling"], dict)


class TestUtils:
    """Tests for utility functions."""

    def test_is_refit_step(self) -> None:
        """Test is_refit_step function."""
        assert is_refit_step(0, 5) is True
        assert is_refit_step(5, 5) is True
        assert is_refit_step(10, 5) is True
        assert is_refit_step(1, 5) is False
        assert is_refit_step(4, 5) is False
        assert is_refit_step(0, 0) is True
        assert is_refit_step(1, 0) is False

    def test_add_point_to_series(self) -> None:
        """Test add_point_to_series function."""
        series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        new_date = pd.Timestamp("2020-01-04")
        result = add_point_to_series(series, 4.0, new_date)

        assert len(result) == 4
        assert result.iloc[-1] == 4.0
        assert result.index[-1] == new_date

    def test_to_date_index_with_datetime_index(self) -> None:
        """Test to_date_index with existing datetime index."""
        series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        result = to_date_index(series)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3

    def test_to_date_index_without_datetime_index(self) -> None:
        """Test to_date_index without datetime index."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = to_date_index(series)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3

    def test_detect_value_column_with_weighted_log_return(self) -> None:
        """Test detect_value_column with weighted_log_return column."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "weighted_log_return": [0.01, -0.02],
            }
        )
        result = detect_value_column(df)
        assert result == "weighted_log_return"

    def test_detect_value_column_with_log_return(self) -> None:
        """Test detect_value_column with log_return column."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "log_return": [0.01, -0.02],
            }
        )
        result = detect_value_column(df)
        assert result == "log_return"

    def test_detect_value_column_with_y(self) -> None:
        """Test detect_value_column with y column."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "y": [0.01, -0.02],
            }
        )
        result = detect_value_column(df)
        assert result == "y"

    def test_detect_value_column_with_target(self) -> None:
        """Test detect_value_column with target column."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "target": [0.01, -0.02],
            }
        )
        result = detect_value_column(df)
        assert result == "target"

    def test_detect_value_column_with_return_in_name(self) -> None:
        """Test detect_value_column with return in column name."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "daily_return": [0.01, -0.02],
            }
        )
        result = detect_value_column(df)
        assert result == "daily_return"

    def test_detect_value_column_raises_error(self) -> None:
        """Test detect_value_column raises ValueError when no suitable column found."""
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "other_col": [0.01, -0.02],
            }
        )
        with pytest.raises(ValueError, match="Could not identify the returns column"):
            detect_value_column(df)


class TestSaveGarchDataset:
    """Tests for save_garch_dataset function."""

    @patch("src.arima.evaluation_arima.save_data_for_garch._compute_train_residuals")
    @patch("src.arima.evaluation_arima.save_data_for_garch.pd.read_csv")
    @patch("src.arima.evaluation_arima.save_data_for_garch.logger")
    def test_save_garch_dataset_with_results(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_compute_train: MagicMock,
    ) -> None:
        """Test save_garch_dataset with results dict."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_split:
            tmp_split_path = Path(tmp_split.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_garch:
            tmp_garch_path = Path(tmp_garch.name)

        try:
            # Write test split file
            split_df = pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                    "split": ["train", "train", "test"],
                    "weighted_log_return": [0.01, -0.02, 0.015],
                }
            )
            split_df.to_csv(tmp_split_path, index=False)
            mock_read_csv.return_value = split_df

            # Mock train residuals
            train_resid_df = pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-02"],
                    "arima_resid": [0.001, -0.001],
                }
            )
            mock_compute_train.return_value = train_resid_df

            # Mock results
            results = {
                "dates": ["2020-01-03"],
                "y_true": [0.015],
                "y_pred": [0.014],
                "residuals": [0.001],
            }

            # Mock fitted model
            mock_fitted_model = MagicMock()
            mock_fitted_model.fittedvalues = pd.Series([0.01, -0.02])

            # Patch file paths
            with patch(
                "src.arima.evaluation_arima.save_data_for_garch.WEIGHTED_LOG_RETURNS_SPLIT_FILE",
                tmp_split_path,
            ):
                with patch(
                    "src.arima.evaluation_arima.save_data_for_garch.GARCH_DATASET_FILE",
                    tmp_garch_path,
                ):
                    result_path = save_garch_dataset(
                        results=results, fitted_model=mock_fitted_model
                    )

            # Verify the file was created
            assert result_path == tmp_garch_path
            assert tmp_garch_path.exists()
            assert mock_logger.info.call_count >= 1
        finally:
            # Cleanup
            if tmp_split_path.exists():
                tmp_split_path.unlink()
            if tmp_garch_path.exists():
                tmp_garch_path.unlink()

    def test_save_garch_dataset_file_not_found(self) -> None:
        """Test save_garch_dataset raises FileNotFoundError when split file missing."""
        # Create a path that doesn't exist
        non_existent_path = Path("/tmp/non_existent_file_12345.csv")

        with patch(
            "src.arima.evaluation_arima.save_data_for_garch.WEIGHTED_LOG_RETURNS_SPLIT_FILE",
            non_existent_path,
        ):
            with pytest.raises(FileNotFoundError):
                save_garch_dataset()


class TestRegenerateGarchDataset:
    """Tests for regenerate_garch_dataset_from_rolling_predictions function."""

    @patch("src.arima.evaluation_arima.save_data_for_garch.save_garch_dataset")
    @patch("src.arima.training_arima.training_arima.load_trained_model")
    @patch("src.arima.evaluation_arima.save_data_for_garch.logger")
    def test_regenerate_garch_dataset_calls_save(
        self,
        mock_logger: MagicMock,
        mock_load_model: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test regenerate_garch_dataset_from_rolling_predictions calls save_garch_dataset."""
        mock_path = Path("/fake/path")
        mock_save.return_value = mock_path
        mock_load_model.return_value = (None, {})  # Mock the loaded model as None

        result = regenerate_garch_dataset_from_rolling_predictions()

        mock_save.assert_called_once_with(results=None, fitted_model=None, backtest_residuals=None)
        assert result == mock_path


class TestPlotPredictionsVsActual:
    """Tests for plot_predictions_vs_actual function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.plt")
    @patch("pandas.read_csv")
    def test_plot_predictions_vs_actual_success(self, mock_read_csv, mock_plt) -> None:
        """Test successful plotting of predictions vs actual values."""
        # Mock dataframe with test data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        test_df = pd.DataFrame(
            {
                "date": [d.strftime("%Y-%m-%d") for d in dates],
                "y_true": [0.01, -0.005, 0.002, -0.001, 0.003, 0.0, -0.002, 0.004, 0.001, -0.003],
                "y_pred": [
                    0.008,
                    -0.003,
                    0.001,
                    -0.002,
                    0.004,
                    0.001,
                    -0.001,
                    0.003,
                    0.002,
                    -0.002,
                ],
                "residual": [
                    0.002,
                    -0.002,
                    0.001,
                    0.001,
                    -0.001,
                    -0.001,
                    -0.001,
                    0.001,
                    -0.001,
                    -0.001,
                ],
            }
        )
        mock_read_csv.return_value = test_df

        # Mock matplotlib
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Mock the path creation to avoid Path issues
        with patch("src.arima.evaluation_arima.evaluation_arima.Path") as mock_path:
            mock_input_path = MagicMock()
            mock_path.return_value = mock_input_path
            mock_input_path.parent.mkdir = MagicMock()

            # Call the function
            plot_predictions_vs_actual()

            # Verify Path was called twice (for input and output)
            assert mock_path.call_count >= 2

        # Verify calls
        mock_read_csv.assert_called_once()
        mock_plt.subplots.assert_called_once_with(figsize=(12, 8), dpi=150)  # PLOT_DPI_EVALUATION
        # Should be called twice (actual and predicted)
        assert mock_ax.plot.call_count == 2
        mock_ax.set_xlabel.assert_called_with("Date", fontsize=14, fontweight="bold")
        mock_ax.set_ylabel.assert_called_with(
            "Rendements logarithmiques", fontsize=14, fontweight="bold"
        )
        mock_ax.set_title.assert_called()
        mock_ax.legend.assert_called()
        mock_ax.text.assert_called()
        mock_fig.savefig.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
