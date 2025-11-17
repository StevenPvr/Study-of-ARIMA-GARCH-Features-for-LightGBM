"""Tests for random forest data preparation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.constants import LIGHTGBM_LAG_WINDOWS, LIGHTGBM_TECHNICAL_FEATURE_COLUMNS
from src.lightgbm.data_preparation.dataset_builders import create_dataset_technical_indicators
from src.lightgbm.data_preparation.dataset_creation import prepare_datasets
from src.lightgbm.data_preparation.features import add_lag_features


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
    """Create ticker-level base test dataframe for dataset preparation."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * periods,
            "open": np.linspace(99, 199, periods, retstep=False),
            "high": np.linspace(101, 201, periods, retstep=False),
            "low": np.linspace(98, 198, periods, retstep=False),
            "closing": np.linspace(100, 200, periods, retstep=False),
            "volume": np.linspace(1_000, 2_000, periods, retstep=False),
            "split": ["train"] * (periods - 5) + ["test"] * 5,
        }
    )


def _create_ticker_base_test_dataframe(periods: int, tickers: list[str]) -> pd.DataFrame:
    """Create ticker-level dataframe with proper ticker-level columns (closing, open, volume)."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    frames = []
    for idx, ticker in enumerate(tickers):
        base_price = 100.0 + idx * 10.0
        ticker_df = pd.DataFrame(
            {
                "date": dates,
                "closing": np.linspace(base_price, base_price + 100, periods, retstep=False),
                "open": np.linspace(base_price - 1, base_price + 99, periods, retstep=False),
                "high": np.linspace(base_price + 1, base_price + 101, periods, retstep=False),
                "low": np.linspace(base_price - 2, base_price + 98, periods, retstep=False),
                "volume": np.linspace(1_000 + idx * 100, 2_000 + idx * 100, periods, retstep=False),
                "split": ["train"] * (periods - 5) + ["test"] * 5,
                "ticker": ticker,
            }
        )
        frames.append(ticker_df)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def _check_lag_features_present(df_complete: pd.DataFrame) -> None:
    """Check that lag features are present and target column is included."""
    # log_volatility is created after lags, so it doesn't have lag features
    # Instead, check that other features have lags (e.g., log_return, log_volume)
    assert "log_return_lag_1" in df_complete.columns or "log_volume_lag_1" in df_complete.columns
    assert "log_volatility" in df_complete.columns, "Target column log_volatility must be present"


def _check_non_observable_columns_removed(
    df_complete: pd.DataFrame, df_without: pd.DataFrame
) -> None:
    """Ensure non-observable price columns and their lags are removed."""
    for dataset in (df_complete, df_without):
        assert "closing" not in dataset.columns
        for lag in LIGHTGBM_LAG_WINDOWS:
            assert f"closing_lag_{lag}" not in dataset.columns
            # Open lags ne sont jamais créés et ne doivent pas apparaître
            assert f"open_lag_{lag}" not in dataset.columns


def _check_no_insights_present(df_: pd.DataFrame) -> None:
    """Ensure ARIMA-GARCH insight columns and their lags are not present."""
    insight_cols = ["sigma2_garch", "sigma_garch", "std_resid_garch", "arima_residual_return"]
    for col in insight_cols:
        assert col not in df_.columns
        for lag in LIGHTGBM_LAG_WINDOWS:
            assert f"{col}_lag_{lag}" not in df_.columns


def _create_base_test_dataframe_with_insights(periods: int) -> pd.DataFrame:
    """Create ticker-level base test dataframe with GARCH insights for dataset preparation."""
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["AAA"] * periods,
            "open": np.linspace(99, 199, periods, retstep=False),
            "high": np.linspace(101, 201, periods, retstep=False),
            "low": np.linspace(98, 198, periods, retstep=False),
            "closing": np.linspace(100, 200, periods, retstep=False),
            "volume": np.linspace(1_000, 2_000, periods, retstep=False),
            "split": ["train"] * (periods - 5) + ["test"] * 5,
            "sigma2_garch": rng.uniform(0.0001, 0.0005, periods),
            "sigma_garch": rng.uniform(0.01, 0.02, periods),
            "std_resid_garch": rng.uniform(-0.02, 0.02, periods),
            "arima_residual_return": rng.uniform(-0.01, 0.01, periods),
        }
    )


def _check_output_files_exist(tmp_path: Path) -> None:
    """Check that output files exist."""
    assert (tmp_path / "lightgbm_dataset_complete.parquet").exists()
    assert (tmp_path / "lightgbm_dataset_without_insights.parquet").exists()


@pytest.fixture
def patch_lightgbm_variant_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    """No longer needed since functions now accept output_dir parameter."""
    return {}


def test_prepare_datasets_builds_new_features_and_lags(
    tmp_path: Path, patch_lightgbm_variant_paths: dict[str, Path]
) -> None:
    """Verify indicators use lags 1-3, avoid spurious insights, and drop the closing column."""
    _ = patch_lightgbm_variant_paths
    periods = 60
    base_df = _create_base_test_dataframe(periods)

    df_complete, df_without = prepare_datasets(df=base_df, output_dir=tmp_path)

    # Some rows should remain after feature engineering and shifts
    assert len(df_complete) > 0
    _check_lag_features_present(df_complete)
    _check_non_observable_columns_removed(df_complete, df_without)
    # Pas d'insights créés à partir de rien
    _check_no_insights_present(df_complete)
    _check_no_insights_present(df_without)

    # Base features present
    assert "log_volatility" in df_complete.columns
    # log_volatility is created after lags are added, so it doesn't have lag features
    # Instead, verify that other base features have lags
    assert "log_return_lag_1" in df_complete.columns
    assert "log_volume_lag_1" in df_complete.columns

    # Calendar features without lags
    for cal in ["day_of_week", "month", "is_month_end", "day_in_month_norm"]:
        assert cal in df_complete.columns
        for lag in LIGHTGBM_LAG_WINDOWS:
            assert f"{cal}_lag_{lag}" not in df_complete.columns

    # No higher lags
    assert not any(col.endswith("lag_4") for col in df_complete.columns)
    _check_output_files_exist(tmp_path)


def test_prepare_datasets_adds_ticker_id_for_ticker_data(
    tmp_path: Path, patch_lightgbm_variant_paths: dict[str, Path]
) -> None:
    """Ensure ticker-level data gains a ticker_id feature."""
    _ = patch_lightgbm_variant_paths
    periods = 70
    ticker_df = _create_ticker_base_test_dataframe(periods, ["AAA", "BBB", "CCC"])

    df_complete, df_without = prepare_datasets(df=ticker_df, output_dir=tmp_path)

    assert "ticker_id" in df_complete.columns
    assert df_complete["ticker_id"].nunique() == 3
    assert "ticker_id" in df_without.columns
    assert df_without["ticker_id"].nunique() == 3


def test_create_dataset_technical_indicators(tmp_path: Path) -> None:
    """Verify custom indicator dataset includes indicators and lags."""
    periods = 60
    base_df = _create_base_test_dataframe(periods)

    output_path = tmp_path / "lightgbm_dataset_technical_indicators.parquet"
    df_technical = create_dataset_technical_indicators(
        df=base_df,
        output_path=output_path,
        include_lags=True,
    )

    assert output_path.exists()
    # Get target column name (exclude from technical indicator checks)
    from src.lightgbm.data_preparation.target_creation import get_target_column_name

    target_col = get_target_column_name(df_technical)
    # Ensure required indicator columns exist (excluding rsi_14 and target which is handled separately)
    for indicator in LIGHTGBM_TECHNICAL_FEATURE_COLUMNS:
        if indicator not in ("rsi_14", target_col):
            assert indicator in df_technical.columns
    # Ensure lag columns exist for each indicator (excluding rsi_14 and target)
    # Target lags are checked separately via select_target_columns
    for indicator in LIGHTGBM_TECHNICAL_FEATURE_COLUMNS:
        if indicator not in ("rsi_14", target_col):
            for lag in LIGHTGBM_LAG_WINDOWS:
                lag_col = f"{indicator}_lag_{lag}"
                assert lag_col in df_technical.columns
    # Note: Target lags (log_volatility_lag_*) may not be included in technical dataset
    # They are handled separately via select_target_columns(include_lags=True)
    # If target is in LIGHTGBM_TECHNICAL_FEATURE_COLUMNS, its lags might be created
    # but may not be selected in the technical dataset due to selection logic
    # Ensure raw price columns removed
    assert "closing" not in df_technical.columns
    assert "open" not in df_technical.columns
    for lag in LIGHTGBM_LAG_WINDOWS:
        assert f"closing_lag_{lag}" not in df_technical.columns
        assert f"open_lag_{lag}" not in df_technical.columns

    # Calendar features must be present without lags in the technical dataset
    for cal in ["day_of_week", "month", "is_month_end", "day_in_month_norm"]:
        assert cal in df_technical.columns
        for lag in LIGHTGBM_LAG_WINDOWS:
            assert f"{cal}_lag_{lag}" not in df_technical.columns


def test_select_calendar_feature_columns() -> None:
    """Ensure calendar selector returns only present calendar columns without lags."""
    from src.lightgbm.data_preparation.columns import select_calendar_feature_columns

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="B"),
            "day_of_week": [0, 1, 2],
            "month": [1, 1, 1],
            "is_month_end": [0, 0, 0],
            # missing day_in_month_norm on purpose to verify filtering
            "log_return": [0.0, 0.1, -0.1],
        }
    )

    selected = select_calendar_feature_columns(df)
    assert set(selected) == {"day_of_week", "month", "is_month_end"}


def test_prepare_datasets_creates_insight_lags(
    tmp_path: Path, patch_lightgbm_variant_paths: dict[str, Path]
) -> None:
    """Verify that GARCH insight columns get lag features created when present."""
    _ = patch_lightgbm_variant_paths
    periods = 70
    base_df = _create_base_test_dataframe_with_insights(periods)

    df_complete, df_without = prepare_datasets(df=base_df, output_dir=tmp_path)

    # Verify insights are present in complete dataset
    insight_cols = ["sigma2_garch", "sigma_garch", "std_resid_garch", "arima_residual_return"]
    for insight_col in insight_cols:
        assert insight_col in df_complete.columns, f"{insight_col} should be present"
        # Verify lags are created for each insight
        for lag in LIGHTGBM_LAG_WINDOWS:
            lag_col = f"{insight_col}_lag_{lag}"
            assert lag_col in df_complete.columns, f"{lag_col} should be created"

    # Dataset "without_insights" ne doit contenir ni insights ni leurs lags
    _check_no_insights_present(df_without)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
