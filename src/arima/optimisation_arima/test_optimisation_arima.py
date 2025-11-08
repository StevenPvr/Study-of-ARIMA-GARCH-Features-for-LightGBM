"""Unit tests for optimisation_arima module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.optimisation_arima.optimisation_arima import (
    load_train_test_data,
    optimize_sarima_models,
)
from src.arima.optimisation_arima.utils import _build_best_model_dict


class TestOptimizeSarimaModels:
    """Tests for optimize_sarima_models function."""

    @staticmethod
    def _assert_has_columns(results_df: pd.DataFrame) -> None:
        """Assert that DataFrame has required columns."""
        if hasattr(results_df, "columns"):
            required_cols = ["p", "d", "q", "P", "D", "Q", "s", "aic", "bic"]
            for col in required_cols:
                assert col in results_df.columns

    @staticmethod
    def _assert_results_dataframe(results_df: pd.DataFrame) -> None:
        """Assert that results DataFrame has expected structure."""
        assert hasattr(results_df, "columns") or hasattr(results_df, "__len__")
        TestOptimizeSarimaModels._assert_has_columns(results_df)
        assert len(results_df) > 0

    @staticmethod
    def _assert_model_dict(model_dict: dict, required_keys: list[str]) -> None:
        """Assert that model dictionary has required keys."""
        assert isinstance(model_dict, dict)
        for key in required_keys:
            assert key in model_dict

    @patch("src.arima.optimisation_arima.optimisation_arima._save_optimization_results")
    @patch("src.arima.optimisation_arima.utils.fit_sarima_model")
    @patch("src.arima.optimisation_arima.optimisation_arima.logger")
    def test_optimize_sarima_models_success(
        self,
        mock_logger: MagicMock,
        mock_fit_sarima: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test successful SARIMA optimization."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        test_series = pd.Series([0.01, -0.02] * 10)

        # Setup mock fitted model
        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_fitted.bic = -95.0
        # Provide white-noise residuals so Ljung–Box does not reject
        import numpy as _np

        mock_fitted.resid = _np.random.default_rng(123).normal(0.0, 1.0, size=200)
        mock_fit_sarima.return_value = mock_fitted

        results_df, best_aic, best_bic = optimize_sarima_models(
            train_series,
            test_series,
            p_range=range(2),
            d_range=range(1),
            q_range=range(2),
        )

        self._assert_results_dataframe(results_df)
        required_keys = ["p", "d", "q", "P", "D", "Q", "s", "aic", "bic", "params"]
        self._assert_model_dict(best_aic, required_keys)
        self._assert_model_dict(best_bic, required_keys)

        mock_save.assert_called_once()
        assert mock_logger.info.call_count > 0
        # Whiteness columns exist
        assert "lb_reject_5pct" in results_df.columns
        assert "lb_pvalue_last" in results_df.columns

    @patch("src.arima.optimisation_arima.utils._test_sarima_model")
    def test_optimize_sarima_models_no_convergence(
        self,
        mock_test_sarima: MagicMock,
    ) -> None:
        """Test optimization when no models converge."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        test_series = pd.Series([0.01, -0.02] * 5)

        # Mock _test_sarima_model to always return None (failed)
        mock_test_sarima.return_value = None

        # Execute and verify exception
        # Use n_jobs=1 to force sequential mode (mocks don't work well with multiprocessing)
        with pytest.raises(RuntimeError, match="No SARIMA models converged"):
            optimize_sarima_models(
                train_series,
                test_series,
                p_range=range(1),
                d_range=range(1),
                q_range=range(1),
                n_jobs=1,  # Force sequential mode for reliable mocking
            )

    def test_optimize_sarima_models_invalid_inputs(self) -> None:
        """Test optimization with invalid inputs."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        test_series = pd.Series([0.01, -0.02] * 5)

        # Test empty train series
        with pytest.raises(ValueError, match="train_series cannot be empty"):
            optimize_sarima_models(
                pd.Series(dtype=float),
                test_series,
                p_range=range(1),
                d_range=range(1),
                q_range=range(1),
            )

        # Test invalid seasonal period
        with pytest.raises(ValueError, match="Seasonal period s must be >= 1"):
            optimize_sarima_models(
                train_series,
                test_series,
                p_range=range(1),
                d_range=range(1),
                q_range=range(1),
                s=0,
            )

    @patch("src.arima.optimisation_arima.optimisation_arima._save_optimization_results")
    @patch("src.arima.optimisation_arima.utils.fit_sarima_model")
    def test_optimize_sarima_models_partial_convergence(
        self,
        mock_fit_sarima: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test optimization when some models fail."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 50)
        test_series = pd.Series([0.01, -0.02] * 5)

        # Mock fit_sarima_model to succeed for some, fail for others
        call_count = 0

        def fit_sarima_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                msg = "Model failed"
                raise RuntimeError(msg)
            mock_fitted = MagicMock()
            mock_fitted.aic = -100.0 - call_count
            mock_fitted.bic = -95.0 - call_count
            import numpy as _np

            mock_fitted.resid = _np.random.default_rng(42 + call_count).normal(0.0, 1.0, size=120)
            return mock_fitted

        mock_fit_sarima.side_effect = fit_sarima_side_effect

        # Execute
        results_df, best_aic, best_bic = optimize_sarima_models(
            train_series,
            test_series,
            p_range=range(2),
            d_range=range(1),
            q_range=range(2),
        )

        # Verify some results were saved
        assert len(results_df) > 0
        self._assert_results_dataframe(results_df)
        required_keys = ["p", "d", "q", "P", "D", "Q", "s", "aic", "bic", "params"]
        self._assert_model_dict(best_aic, required_keys)
        self._assert_model_dict(best_bic, required_keys)
        mock_save.assert_called_once()


class TestLoadTrainTestData:
    """Tests for load_train_test_data function."""

    @staticmethod
    def _create_mock_split_dataframe() -> pd.DataFrame:
        """Create a mock DataFrame for testing."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "split": ["train"] * 8 + ["test"] * 2,
                "weighted_log_return": [0.01] * 10,
            }
        )

    @staticmethod
    def _assert_series_type_and_length(series: pd.Series, expected_len: int) -> None:
        """Assert that a series has correct type and length."""
        assert isinstance(series, pd.Series)
        assert len(series) == expected_len

    @staticmethod
    def _assert_series_index(series: pd.Series) -> None:
        """Assert that a series has correct index."""
        assert series.index.name == "date" or isinstance(series.index, pd.DatetimeIndex)

    @staticmethod
    def _assert_series_structure(
        train_series: pd.Series,
        test_series: pd.Series,
        expected_train_len: int,
        expected_test_len: int,
    ) -> None:
        """Assert that series have correct structure."""
        TestLoadTrainTestData._assert_series_type_and_length(train_series, expected_train_len)
        TestLoadTrainTestData._assert_series_type_and_length(test_series, expected_test_len)
        TestLoadTrainTestData._assert_series_index(train_series)
        TestLoadTrainTestData._assert_series_index(test_series)

    @patch("src.arima.optimisation_arima.optimisation_arima.WEIGHTED_LOG_RETURNS_SPLIT_FILE")
    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_csv")
    @patch("src.arima.optimisation_arima.optimisation_arima.logger")
    def test_load_train_test_data_success(
        self,
        mock_logger: MagicMock,
        mock_read_csv: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """Test successful loading of train/test data."""
        mock_file.exists.return_value = True
        mock_read_csv.return_value = self._create_mock_split_dataframe()

        train_series, test_series = load_train_test_data()

        self._assert_series_structure(train_series, test_series, 8, 2)
        mock_read_csv.assert_called_once_with(mock_file, parse_dates=["date"])
        assert mock_logger.info.call_count >= 2

    @patch("src.arima.optimisation_arima.optimisation_arima.WEIGHTED_LOG_RETURNS_SPLIT_FILE")
    def test_load_train_test_data_file_not_found(
        self,
        mock_file: MagicMock,
    ) -> None:
        """Test when split data file doesn't exist."""
        mock_file.exists.return_value = False

        # Execute and verify exception
        with pytest.raises(FileNotFoundError, match="Split data file not found"):
            load_train_test_data()

    @patch("src.arima.optimisation_arima.optimisation_arima.WEIGHTED_LOG_RETURNS_SPLIT_FILE")
    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_csv")
    def test_load_train_test_data_missing_columns(
        self,
        mock_read_csv: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """Test when data file is missing required columns."""
        mock_file.exists.return_value = True
        mock_df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5)})
        mock_read_csv.return_value = mock_df

        with pytest.raises(ValueError, match="Missing required columns"):
            load_train_test_data()

    @patch("src.arima.optimisation_arima.optimisation_arima.WEIGHTED_LOG_RETURNS_SPLIT_FILE")
    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_csv")
    def test_load_train_test_data_empty_train_set(
        self,
        mock_read_csv: MagicMock,
        mock_file: MagicMock,
    ) -> None:
        """Test when train set is empty (no train split values)."""
        mock_file.exists.return_value = True
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "split": ["test"] * 5,
                "weighted_log_return": [0.01] * 5,
            }
        )
        mock_read_csv.return_value = mock_df

        # When train split is missing, _validate_split_values raises first
        with pytest.raises(ValueError, match="Invalid split values"):
            load_train_test_data()


class TestBuildBestModelDict:
    """Tests for _build_best_model_dict function."""

    def test_build_best_model_dict_missing_seasonal_params(self) -> None:
        """Test that ValueError is raised when seasonal parameters are missing."""
        # Create a Series with only non-seasonal parameters
        incomplete_row = pd.Series(
            {
                "p": 1,
                "d": 1,
                "q": 1,
                "aic": -100.0,
                "bic": -95.0,
            }
        )

        with pytest.raises(ValueError, match="Missing required SARIMA parameters"):
            _build_best_model_dict(incomplete_row)

    def test_build_best_model_dict_success(self) -> None:
        """Test successful building of model dict with all parameters."""
        complete_row = pd.Series(
            {
                "p": 1,
                "d": 1,
                "q": 1,
                "P": 0,
                "D": 0,
                "Q": 0,
                "s": 12,
                "aic": -100.0,
                "bic": -95.0,
            }
        )

        result = _build_best_model_dict(complete_row)

        assert isinstance(result, dict)
        expected_values = {
            "p": 1,
            "d": 1,
            "q": 1,
            "P": 0,
            "D": 0,
            "Q": 0,
            "s": 12,
            "aic": -100.0,
            "bic": -95.0,
        }
        for key, expected_value in expected_values.items():
            assert result[key] == expected_value
        assert "params" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
