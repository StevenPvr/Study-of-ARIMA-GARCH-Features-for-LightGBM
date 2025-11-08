"""Tests for random forest data preparation utilities."""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.constants import RF_LAG_WINDOWS
from src.random_forest.data_preparation.utils import add_lag_features, prepare_datasets


def _create_test_dataframe() -> pd.DataFrame:
    """Create test dataframe for lag features testing."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "weighted_log_return": np.arange(5, dtype=float),
        }
    )


def _check_lag_columns_present(result: pd.DataFrame) -> None:
    """Check that lag columns are present."""
    assert "weighted_log_return_lag_1" in result.columns
    assert "weighted_log_return_lag_3" in result.columns


def _check_lag_values(result: pd.DataFrame, original_df: pd.DataFrame) -> None:
    """Check that lag values are correct."""
    assert pd.isna(result.loc[0, "weighted_log_return_lag_1"])
    assert result.loc[3, "weighted_log_return_lag_1"] == pytest.approx(
        original_df.loc[2, "weighted_log_return"]
    )
    assert result.loc[4, "weighted_log_return_lag_3"] == pytest.approx(
        original_df.loc[1, "weighted_log_return"]
    )


def test_add_lag_features_creates_shifted_columns() -> None:
    """Ensure lag features are appended with the expected shift."""
    df = _create_test_dataframe()
    result = add_lag_features(df, feature_columns=["weighted_log_return"], lag_windows=[1, 3])
    _check_lag_columns_present(result)
    _check_lag_values(result, df)


def _create_base_test_dataframe(periods: int) -> pd.DataFrame:
    """Create base test dataframe for dataset preparation."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "weighted_closing": np.linspace(100, 200, periods),
            "weighted_open": np.linspace(99, 199, periods),
            "log_weighted_return": np.linspace(-0.01, 0.01, periods),
            "sigma2_garch": np.linspace(0.1, 0.2, periods),
            "sigma_garch": np.linspace(0.3, 0.4, periods),
            "std_resid_garch": np.linspace(-1.0, 1.0, periods),
            "split": ["train"] * (periods - 5) + ["test"] * 5,
        }
    )


def _fake_add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fake function to add technical indicators for testing."""
    augmented = df.copy()
    augmented["rsi_14"] = np.linspace(30, 70, len(df))
    augmented["sma_20"] = np.linspace(90, 110, len(df))
    augmented["ema_20"] = np.linspace(91, 111, len(df))
    augmented["macd"] = np.linspace(-1, 1, len(df))
    augmented["macd_signal"] = np.linspace(-0.5, 0.5, len(df))
    augmented["macd_histogram"] = np.linspace(-0.2, 0.2, len(df))
    return augmented


def _check_dataset_length(df_complete: pd.DataFrame, expected_length: int) -> None:
    """Check that dataset has expected length."""
    assert len(df_complete) == expected_length


def _check_lag_features_present(df_complete: pd.DataFrame) -> None:
    """Check that lag features are present."""
    assert "weighted_log_return_lag_1" in df_complete.columns
    assert "weighted_log_return_t" in df_complete.columns
    assert "weighted_log_return_t_lag_1" not in df_complete.columns


def _check_non_observable_columns_removed(
    df_complete: pd.DataFrame, df_without: pd.DataFrame
) -> None:
    """Ensure weighted price columns and their lags are removed."""

    for dataset in (df_complete, df_without):
        assert "weighted_closing" not in dataset.columns
        assert "weighted_open" not in dataset.columns
        for lag in RF_LAG_WINDOWS:
            assert f"weighted_closing_lag_{lag}" not in dataset.columns
            assert f"weighted_open_lag_{lag}" not in dataset.columns


def _check_insights_removed(df_without: pd.DataFrame) -> None:
    """Check that insight columns are removed."""
    assert "sigma2_garch_lag_1" not in df_without.columns
    assert "sigma2_garch" not in df_without.columns


def _check_output_files_exist(tmp_path: Path) -> None:
    """Check that output files exist."""
    assert (tmp_path / "rf_dataset_complete.csv").exists()
    assert (tmp_path / "rf_dataset_without_insights.csv").exists()


def test_prepare_datasets_includes_lags_and_drops_insights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify lag features are created and insight columns removed."""
    periods = max(RF_LAG_WINDOWS) + 5
    base_df = _create_base_test_dataframe(periods)

    monkeypatch.setattr(
        "src.random_forest.data_preparation.utils.add_technical_indicators",
        _fake_add_technical_indicators,
    )

    df_complete, df_without = prepare_datasets(df=base_df, output_dir=tmp_path)

    expected_length = periods - max(RF_LAG_WINDOWS) - 1
    _check_dataset_length(df_complete, expected_length)
    _check_lag_features_present(df_complete)
    _check_non_observable_columns_removed(df_complete, df_without)
    _check_insights_removed(df_without)
    _check_output_files_exist(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
