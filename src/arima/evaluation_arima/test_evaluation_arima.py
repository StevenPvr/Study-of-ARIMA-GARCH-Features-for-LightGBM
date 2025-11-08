"""Unit tests for evaluation_arima module."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.evaluation_arima.evaluation_arima import (
    calculate_metrics,
    compute_residuals,
    evaluate_model,
    ljung_box_on_residuals,
    rolling_forecast,
    save_evaluation_results,
)


class TestRollingForecast:
    """Tests for rolling_forecast function."""

    @patch("statsmodels.tsa.arima.model.ARIMA")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_rolling_forecast_success(
        self,
        mock_logger: MagicMock,
        mock_arima: MagicMock,
    ) -> None:
        """Test successful rolling forecast."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015, -0.01, 0.02] * 10,
            index=pd.date_range("2020-01-01", periods=50, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02, 0.015], index=pd.date_range("2020-02-20", periods=3, freq="D")
        )

        # Mock ARIMA model
        mock_fitted = MagicMock()
        mock_fitted.forecast.return_value = pd.Series([0.01])
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_fitted
        mock_arima.return_value = mock_model

        # Execute
        predictions, actuals = rolling_forecast(
            train_series, test_series, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12), verbose=False
        )

        # Verify
        assert len(predictions) == len(test_series)
        assert len(actuals) == len(test_series)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(actuals, np.ndarray)

    @patch("statsmodels.tsa.arima.model.ARIMA")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_rolling_forecast_with_failures(
        self,
        mock_logger: MagicMock,
        mock_arima: MagicMock,
    ) -> None:
        """Test rolling forecast with some forecast failures."""
        train_series = pd.Series(
            [0.01, -0.02, 0.015] * 10,
            index=pd.date_range("2020-01-01", periods=30, freq="D"),
        )
        test_series = pd.Series(
            [0.01, -0.02], index=pd.date_range("2020-02-01", periods=2, freq="D")
        )

        call_count = 0

        def arima_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_model = MagicMock()
            if call_count == 1:
                # First call fails during fit
                mock_model.fit.side_effect = Exception("Forecast failed")
            else:
                # Subsequent calls succeed
                mock_fitted = MagicMock()
                mock_fitted.forecast.return_value = pd.Series([0.01])
                mock_model.fit.return_value = mock_fitted
            return mock_model

        mock_arima.side_effect = arima_side_effect

        # Execute with warnings suppressed for statsmodels
        # (FutureWarning and ValueWarning from mocks)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            predictions, actuals = rolling_forecast(
                train_series,
                test_series,
                order=(1, 0, 1),
                seasonal_order=(0, 0, 0, 12),
                verbose=False,
            )

        # Verify
        assert len(predictions) == len(test_series)
        assert len(actuals) == len(test_series)


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
            seasonal_order=(0, 0, 0, 12),
            model_info={"params": "ARIMA(1,0,1)"},
        )

        # Use length of predictions from mock (15 elements)
        expected_len = len(predictions)
        self._assert_evaluation_results(results, (1, 0, 1), expected_len)
        mock_rolling.assert_called_once()
        mock_calculate.assert_called_once()


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    @patch("src.arima.evaluation_arima.evaluation_arima.Path")
    @patch("src.arima.evaluation_arima.evaluation_arima.ROLLING_PREDICTIONS_SARIMA_FILE")
    @patch("src.arima.evaluation_arima.evaluation_arima.ROLLING_VALIDATION_METRICS_SARIMA_FILE")
    @patch("src.arima.evaluation_arima.evaluation_arima.RESULTS_DIR")
    @patch("src.arima.evaluation_arima.evaluation_arima.pd.DataFrame")
    @patch("src.arima.evaluation_arima.evaluation_arima.logger")
    def test_save_evaluation_results_success(
        self,
        mock_logger: MagicMock,
        mock_df: MagicMock,
        mock_results_dir: MagicMock,
        mock_metrics_file: MagicMock,
        mock_predictions_file: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test successful saving of evaluation results."""
        results = {
            "model": "ARIMA(1,0,1)",
            "order": (1, 0, 1),
            "seasonal_order": (0, 0, 0, 12),
            "metrics": {"MSE": 0.0001, "RMSE": 0.01, "MAE": 0.008},
            "predictions": [0.01, -0.02, 0.015],
            "actuals": [0.012, -0.019, 0.016],
            "dates": ["2020-02-01", "2020-02-02", "2020-02-03"],
        }

        # Setup mocks
        mock_df_instance = MagicMock()
        mock_df.return_value = mock_df_instance

        # Mock Path operations
        mock_path_instance = MagicMock()
        mock_path_instance.open = mock_open()
        mock_path.return_value = mock_path_instance

        # Mock RESULTS_DIR.mkdir
        mock_results_dir.mkdir = MagicMock()

        save_evaluation_results(results)

        # Verify calls
        mock_df.assert_called_once()
        mock_df_instance.to_csv.assert_called_once_with(mock_predictions_file, index=False)
        mock_results_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_path.assert_called_once_with(mock_metrics_file)
        mock_path_instance.open.assert_called_once_with("w")
        assert mock_logger.info.call_count >= 2


class TestLjungBoxResiduals:
    """Tests for Ljung–Box utilities on ARIMA residuals."""

    def test_compute_residuals_shape(self) -> None:
        actuals = [1.0, 2.0, 3.0]
        preds = [0.5, 1.5, 2.5]
        r = compute_residuals(actuals, preds)
        assert r.shape == (3,)
        assert r.tolist() == [0.5, 0.5, 0.5]

    def test_ljung_box_on_residuals_basic(self) -> None:
        import numpy as np

        # White noise residuals should yield non-systematic small autocorrelation
        rng = np.random.default_rng(0)
        res = rng.standard_normal(300)
        out = ljung_box_on_residuals(res, lags=10)
        assert isinstance(out, dict)
        assert set(["lags", "q_stat", "p_value", "reject_5pct", "n"]).issubset(out.keys())
        assert len(out["lags"]) == len(out["q_stat"]) == len(out["p_value"]) == 10
        assert out["n"] == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
