"""Unit tests for optimisation_arima module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.arima.optimisation_arima.model_evaluation import (  # noqa: E402
    ArimaParams,
    walk_forward_backtest,
)
from src.arima.optimisation_arima.optimisation_arima import (  # noqa: E402
    load_train_data,
    optimize_arima_models,
)
from src.arima.optimisation_arima.results_processing import (  # noqa: E402
    build_best_model_dict,
    determine_sort_columns,
    pick_best,
)
from src.arima.optimisation_arima.validation import validate_backtest_config  # noqa: E402


class TestOptimizeArimaModels:
    """Tests for optimize_arima_models function."""

    @staticmethod
    def _assert_has_columns(results_df: pd.DataFrame) -> None:
        """Assert that DataFrame has required columns."""
        if hasattr(results_df, "columns"):
            # to_dataframe creates columns with param_ prefix
            required_param_cols = [
                "param_p",
                "param_d",
                "param_q",
                "param_refit_every",
            ]
            required_metric_cols = [
                "aic",
                "bic",
            ]
            for col in required_param_cols + required_metric_cols:
                assert col in results_df.columns, f"Missing column: {col}"

            # Backtest columns are optional - backtest is only for , not metrics

    @staticmethod
    def _assert_results_dataframe(results_df: pd.DataFrame) -> None:
        """Assert that results DataFrame has expected structure."""
        assert hasattr(results_df, "columns") or hasattr(results_df, "__len__")
        TestOptimizeArimaModels._assert_has_columns(results_df)
        assert len(results_df) > 0

    @staticmethod
    def _assert_model_dict(model_dict: dict, required_keys: list[str]) -> None:
        """Assert that model dictionary has required keys."""
        assert isinstance(model_dict, dict)
        for key in required_keys:
            assert key in model_dict

    @staticmethod
    def _create_mock_fitted_model() -> MagicMock:
        """Create a mock fitted ARIMA model."""
        import numpy as _np

        mock_fitted = MagicMock()
        mock_fitted.aic = -100.0
        mock_fitted.bic = -95.0
        mock_fitted.resid = _np.random.default_rng(123).normal(0.0, 1.0, size=200)
        return mock_fitted

    @patch("src.arima.optimisation_arima.results_processing.save_results")
    @patch("src.arima.optimisation_arima.optimisation_arima._run_optuna")
    def test_optimize_arima_models_success(
        self,
        mock_run_optuna: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test successful ARIMA optimization."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        test_series = pd.Series([0.01, -0.02] * 10)

        # Mock _run_optuna to return results directly from Optuna trials
        # Results are extracted from trial user_attrs, so they include refit_every
        def mock_run_optuna_side_effect(
            train: pd.Series,
            n_trials: int,
            criterion: str = "aic",
            seed: int = 42,
            backtest_cfg: Optional[Dict[str, int]] = None,
        ) -> list[dict[str, Any]]:
            """Mock _run_optuna to return test data with correct structure."""
            results = []
            for i in range(2):  # Return 2 results
                result = {
                    "params": {
                        "p": 1 + i,
                        "d": 0,
                        "q": 1,
                        "trend": "c",
                        "refit_every": 5,
                    },
                    "aic": -100.0 - i,
                    "bic": -95.0 - i,
                }
                results.append(result)
            return results

        mock_run_optuna.side_effect = mock_run_optuna_side_effect

        results_df, best_aic, best_bic = optimize_arima_models(
            train_series,
            test_series,
            n_trials=10,  # Use small number for testing
            backtest_n_splits=5,
            backtest_test_size=20,
            backtest_refit_every=20,
            out_dir=None,  # Don't save results during testing
        )

        self._assert_results_dataframe(results_df)
        # pick_best returns {"params": {...}, "aic": ..., "bic": ...}
        assert "params" in best_aic
        assert "aic" in best_aic
        assert "bic" in best_aic
        # BIC optimization removed - best_bic is now None
        assert best_bic is None
        # Check params structure (including refit_every)
        param_keys = ["p", "d", "q", "trend", "refit_every"]
        for key in param_keys:
            assert key in best_aic["params"]

        # Verify _run_optuna was called once for AIC (BIC removed)
        assert mock_run_optuna.call_count == 1
        # Verify call was for AIC
        assert mock_run_optuna.call_args_list[0].kwargs.get("criterion") == "aic"

    @patch("src.arima.optimisation_arima.optimisation_arima._run_optuna")
    def test_optimize_arima_models_no_convergence(
        self,
        mock_run_optuna: MagicMock,
    ) -> None:
        """Test optimization when no models converge."""
        # Need enough data to pass validation
        train_series = pd.Series([0.01, -0.02, 0.015] * 50)  # 150 observations
        test_series = pd.Series([0.01, -0.02] * 5)

        # Mock _run_optuna to return empty list (no converged models)
        mock_run_optuna.return_value = []

        # Execute and verify exception
        with pytest.raises(RuntimeError, match="No successful models"):
            optimize_arima_models(
                train_series,
                test_series,
                n_trials=10,
                backtest_n_splits=5,
                backtest_test_size=20,
                backtest_refit_every=20,
                out_dir=None,
            )

    def test_optimize_arima_models_invalid_inputs(self) -> None:
        """Test optimization with invalid inputs."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 10)
        test_series = pd.Series([0.01, -0.02] * 5)

        # Test empty train series
        with pytest.raises(ValueError, match="Series too short for backtest"):
            optimize_arima_models(
                pd.Series(dtype=float),
                test_series,
                backtest_n_splits=5,
                backtest_test_size=20,
                backtest_refit_every=20,
                out_dir=None,
            )

        # Note: seasonal_period parameter has been removed as S is now optimized automatically

        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            optimize_arima_models(
                train_series,
                test_series,
                backtest_cfg={"n_splits": 0, "test_size": 20},
                out_dir=None,
            )

        with pytest.raises(ValueError, match="test_size must be >= 1"):
            optimize_arima_models(
                train_series,
                test_series,
                backtest_cfg={"n_splits": 5, "test_size": 0},
                out_dir=None,
            )

    @patch("src.arima.optimisation_arima.results_processing.save_results")
    @patch("src.arima.optimisation_arima.optimisation_arima._run_optuna")
    def test_optimize_arima_models_partial_convergence(
        self,
        mock_run_optuna: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test optimization when some models fail."""
        train_series = pd.Series([0.01, -0.02, 0.015] * 50)
        test_series = pd.Series([0.01, -0.02] * 5)

        # Mock _run_optuna to return some successful results and some errors
        def mock_run_optuna_side_effect(
            train: pd.Series,
            n_trials: int,
            criterion: str = "aic",
            seed: int = 42,
            backtest_cfg: Optional[Dict[str, int]] = None,
        ) -> list[dict[str, Any]]:
            """Mock _run_optuna to return some results with errors."""
            results = [
                {
                    "params": {
                        "p": 1,
                        "d": 0,
                        "q": 1,
                        "P": 0,
                        "D": 0,
                        "Q": 0,
                        "s": 1,  # Default seasonal period
                        "refit_every": 5,
                    },
                    "error": "Convergence failed",
                },
                {
                    "params": {
                        "p": 2,
                        "d": 0,
                        "q": 1,
                        "P": 0,
                        "D": 0,
                        "Q": 0,
                        "s": 1,  # Default seasonal period
                        "refit_every": 5,
                    },
                    "aic": -100.0,
                    "bic": -95.0,
                },
            ]
            return results  # type: ignore[return-value]

        mock_run_optuna.side_effect = mock_run_optuna_side_effect

        results_df, best_aic, best_bic = optimize_arima_models(
            train_series,
            test_series,
            n_trials=10,
            backtest_cfg={"n_splits": 5, "test_size": 20, "refit_every": 5},
            out_dir=None,
        )

        # Verify some results were saved
        assert len(results_df) > 0
        self._assert_results_dataframe(results_df)
        # Verify _run_optuna was called once for AIC (BIC removed)
        assert mock_run_optuna.call_count == 1
        # pick_best returns {"params": {...}, "aic": ..., "bic": ...}
        assert "params" in best_aic and "aic" in best_aic and "bic" in best_aic
        # BIC optimization removed - best_bic is now None
        assert best_bic is None
        param_keys = ["p", "d", "q", "P", "D", "Q", "s", "refit_every"]
        for key in param_keys:
            assert key in best_aic["params"]

    # test_optimize_arima_models_bic_failure removed - BIC optimization no longer exists

    @patch("src.arima.optimisation_arima.results_processing.save_results")
    @patch("src.arima.optimisation_arima.optimisation_arima._run_optuna")
    def test_optimize_arima_models_random_state(
        self,
        mock_run_optuna: MagicMock,
        mock_save: MagicMock,
    ) -> None:
        """Test that random_state parameter is properly used."""
        train_series = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        test_series = pd.Series([0.01, -0.02] * 10)

        # Track the seeds used in calls
        seeds_used: list[int] = []

        def mock_run_optuna_side_effect(
            train: pd.Series,
            n_trials: int,
            criterion: str = "aic",
            seed: int = 42,
            backtest_cfg: Optional[Dict[str, int]] = None,
        ) -> list[dict[str, Any]]:
            """Mock _run_optuna to track seed usage."""
            seeds_used.append(seed)
            return [
                {
                    "params": {
                        "p": 1,
                        "d": 0,
                        "q": 1,
                        "P": 0,
                        "D": 0,
                        "Q": 0,
                        "s": 1,  # Default seasonal period
                        "refit_every": 5,
                    },
                    "aic": -100.0,
                    "bic": -95.0,
                }
            ]

        mock_run_optuna.side_effect = mock_run_optuna_side_effect

        # Test with custom random_state
        custom_random_state = 123
        results_df, best_aic, best_bic = optimize_arima_models(
            train_series,
            test_series,
            n_trials=10,
            backtest_n_splits=5,
            backtest_test_size=20,
            backtest_refit_every=20,
            random_state=custom_random_state,
            out_dir=None,
        )

        # Verify _run_optuna was called once for AIC (BIC removed)
        assert mock_run_optuna.call_count == 1
        # Verify the call used custom_random_state
        assert seeds_used[0] == custom_random_state
        # BIC optimization removed - best_bic is now None
        assert best_bic is None


class TestLoadTrainData:
    """Tests for load_train_data function."""

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
        # load_train_data returns RangeIndex (reset_index(drop=True))
        assert (
            isinstance(series.index, (pd.RangeIndex, pd.DatetimeIndex))
            or series.index.name == "date"
        )

    @staticmethod
    def _assert_series_structure(
        train_series: pd.Series,
        expected_train_len: int,
    ) -> None:
        """Assert that series have correct structure."""
        TestLoadTrainData._assert_series_type_and_length(train_series, expected_train_len)
        TestLoadTrainData._assert_series_index(train_series)

    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_parquet")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_load_train_data_success(
        self,
        mock_is_file: MagicMock,
        mock_exists: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test successful loading of train data."""
        from pathlib import Path

        mock_df = self._create_mock_split_dataframe()
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Use a real Path object instead of MagicMock
        test_path = Path("test_data.csv")

        train_series = load_train_data(
            csv_path=test_path,
            value_col="weighted_log_return",
            date_col="date",
        )

        self._assert_series_structure(train_series, 8)
        # Verify read_parquet was called with the parquet version of the path
        expected_parquet_path = test_path.with_suffix(".parquet")
        mock_read_parquet.assert_called_once_with(expected_parquet_path)
        # Verify the series have correct values
        assert len(train_series) == 8

    @patch("pathlib.Path.exists")
    def test_load_train_data_file_not_found(
        self,
        mock_exists: MagicMock,
    ) -> None:
        """Test when Parquet file doesn't exist."""
        from pathlib import Path

        # Mock exists to return False for any path check
        mock_exists.return_value = False

        # Use a real Path object
        non_existent_path = Path("non_existent.csv")

        # Execute and verify exception
        with pytest.raises(FileNotFoundError):
            load_train_data(
                csv_path=non_existent_path,
                value_col="weighted_log_return",
            )

    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_parquet")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_load_train_data_missing_columns(
        self,
        mock_is_file: MagicMock,
        mock_exists: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test when data file is missing required columns."""
        from pathlib import Path

        mock_df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5)})
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Use a real Path object
        test_path = Path("test_data.csv")

        with pytest.raises(KeyError):
            load_train_data(
                csv_path=test_path,
                value_col="nonexistent_column",
            )

    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_parquet")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_load_train_data_no_split_column(
        self,
        mock_is_file: MagicMock,
        mock_exists: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test when data file is missing 'split' column."""
        from pathlib import Path

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "weighted_log_return": [0.01] * 5,
            }
        )
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Use a real Path object
        test_path = Path("test_data.csv")

        with pytest.raises(KeyError, match="split"):
            load_train_data(
                csv_path=test_path,
                value_col="weighted_log_return",
            )

    @patch("src.arima.optimisation_arima.optimisation_arima.pd.read_parquet")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_file")
    def test_load_train_data_empty_train_set(
        self,
        mock_is_file: MagicMock,
        mock_exists: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test when train set is empty (no 'train' split found)."""
        from pathlib import Path

        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "split": ["test"] * 5,  # Only test data, no train
                "weighted_log_return": [0.01] * 5,
            }
        )
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True
        mock_is_file.return_value = True

        # Use a real Path object
        test_path = Path("test_data.csv")

        # When no train data found, should raise ValueError
        with pytest.raises(ValueError, match="No 'train' data found"):
            load_train_data(
                csv_path=test_path,
                value_col="weighted_log_return",
            )


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

        with pytest.raises(ValueError, match="Missing required ARIMA parameters"):
            build_best_model_dict(incomplete_row)

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

        result = build_best_model_dict(complete_row)

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


class TestDetermineSortColumns:
    """Tests for _determine_sort_columns function."""

    def test_with_backtest_metrics(self) -> None:
        """Test sorting with Ljung-Box test available."""
        results_df = pd.DataFrame(
            {
                "p": [1, 2],
                "d": [0, 0],
                "q": [1, 1],
                "P": [0, 0],
                "D": [0, 0],
                "Q": [0, 0],
                "s": [12, 12],
                "aic": [100.0, 95.0],
                "lb_reject_5pct": [False, False],
            }
        )
        sort_cols = determine_sort_columns(results_df, "aic")
        assert sort_cols == ["lb_reject_5pct", "aic"]

    def test_without_backtest_metrics(self) -> None:
        """Test sorting without backtest metrics."""
        results_df = pd.DataFrame(
            {
                "p": [1, 2],
                "d": [0, 0],
                "q": [1, 1],
                "P": [0, 0],
                "D": [0, 0],
                "Q": [0, 0],
                "s": [12, 12],
                "aic": [100.0, 95.0],
                "lb_reject_5pct": [False, True],
            }
        )
        sort_cols = determine_sort_columns(results_df, "aic")
        assert sort_cols == ["lb_reject_5pct", "aic"]

    def test_without_lb_reject(self) -> None:
        """Test sorting without lb_reject_5pct."""
        results_df = pd.DataFrame(
            {
                "p": [1, 2],
                "d": [0, 0],
                "q": [1, 1],
                "P": [0, 0],
                "D": [0, 0],
                "Q": [0, 0],
                "s": [12, 12],
                "aic": [100.0, 95.0],
            }
        )
        sort_cols = determine_sort_columns(results_df, "aic")
        assert sort_cols == ["aic"]


class TestSelectBestModels:
    """Tests for select_best_models function."""

    def test_select_with_backtest_metrics(self) -> None:
        """Test selection using AIC/BIC criteria."""
        results_df = pd.DataFrame(
            {
                "param_p": [1, 2, 3],
                "param_d": [0, 0, 0],
                "param_q": [1, 1, 1],
                "param_P": [0, 0, 0],
                "param_D": [0, 0, 0],
                "param_Q": [0, 0, 0],
                "param_s": [12, 12, 12],
                "aic": [100.0, 95.0, 90.0],
                "bic": [105.0, 100.0, 95.0],
                "backtest_rmse_mean": [0.3, 0.2, 0.25],
                "lb_reject_5pct": [False, False, True],
            }
        )
        best_aic, best_bic = pick_best(results_df)
        # Should select model with lowest AIC (index 2, AIC=90.0)
        assert best_aic["params"]["p"] == 3
        assert best_aic["aic"] == 90.0
        # BIC optimization removed - best_bic uses BIC column
        assert best_bic["params"]["p"] == 3
        assert best_bic["bic"] == 95.0


class TestValidateBacktestParameters:
    """Tests for validate_backtest_config function."""

    def test_valid_parameters(self) -> None:
        """Test with valid parameters."""
        validate_backtest_config(
            n_splits=3,
            test_size=10,
            refit_every=5,
            train_len=100,
        )

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        with pytest.raises(ValueError, match="Series too short"):
            validate_backtest_config(
                n_splits=10,
                test_size=10,
                refit_every=5,
                train_len=50,
            )

    def test_invalid_n_splits(self) -> None:
        """Test with invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            validate_backtest_config(
                n_splits=0,
                test_size=10,
                refit_every=5,
                train_len=100,
            )

    def test_invalid_test_size(self) -> None:
        """Test with invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be >= 1"):
            validate_backtest_config(
                n_splits=3,
                test_size=0,
                refit_every=5,
                train_len=100,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestWalkForwardBacktest:
    """Tests for strict block-wise refit logic in walk_forward_backtest."""

    def test_strict_blockwise_refit_and_no_leakage(self) -> None:
        """Ensure training window lengths are correct for blockwise refitting.

        This test verifies that when refit_every > 1, the training windows
        have the correct lengths at each refit point.
        """
        import numpy as _np

        # Synthetic series with datetime index to test temporal ordering
        dates = pd.date_range("2020-01-01", periods=40, freq="D")
        y = pd.Series(_np.arange(40, dtype=float), index=dates)

        # Backtest configuration
        n_splits = 2
        test_size = 6
        refit_every = 2  # override default to exercise block logic

        # Expected training lengths used at each internal refit
        # train_end = 40 - 2*6 = 28
        # split 0: test_start=28 -> blocks start at 28, 30, 32
        # split 1: test_start=34 -> blocks start at 34, 36, 38
        expected_train_lengths = [28, 30, 32, 34, 36, 38]
        observed_train_lengths: list[int] = []

        class _StubAppliedResults:
            """Stub for results returned by apply()."""

            def __init__(self, n_obs: int):
                # forecasts should be shape (1, n_obs) with zeros
                self.forecasts = _np.zeros((1, n_obs), dtype=float)

        class _StubFitted:
            def forecast(self, steps: int) -> _np.ndarray:
                # Return zeros of the correct length to keep metrics finite and simple
                return _np.zeros(int(steps), dtype=float)

            def apply(self, endog: _np.ndarray) -> _StubAppliedResults:
                """Mock apply() that returns forecasts for each observation."""
                n_obs = len(endog) if hasattr(endog, "__len__") else 1
                return _StubAppliedResults(n_obs)

        def _mock_fit(y_train: pd.Series, *args: object, **kwargs: object) -> Any:
            observed_train_lengths.append(int(len(y_train)))
            return _StubFitted()

        # Patch the internal fit function to capture training window boundaries
        with patch("src.arima.optimisation_arima.model_evaluation._fit_arima", new=_mock_fit):
            test_params = ArimaParams(p=0, d=0, q=0, trend="n", refit_every=refit_every)
            metrics = walk_forward_backtest(
                y=y,
                params=test_params,
                n_splits=n_splits,
                test_size=test_size,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

        # Verify the training lengths match the expected block starts exactly
        assert observed_train_lengths == expected_train_lengths

        # Verify metrics structure - backtest returns validation metrics
        assert isinstance(metrics, dict)
        assert len(metrics) == 3  # Returns rmse, mae, mean_error
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mean_error" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
