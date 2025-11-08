"""Unit tests for technical indicators calculation."""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.constants import RF_ARIMA_GARCH_INSIGHT_COLUMNS, RF_LAG_WINDOWS
from src.random_forest.data_preparation.calculs_indicators import (
    add_technical_indicators,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from src.random_forest.data_preparation.utils import (
    load_garch_data,
    prepare_datasets,
)


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "date": dates,
            "weighted_closing": prices,
            "weighted_open": prices * (1 + np.random.normal(0, 0.005, 100)),
            "weighted_log_return": returns,
            "arima_pred_return": np.random.normal(0, 0.001, 100),
            "arima_residual_return": np.random.normal(0, 0.001, 100),
            "sigma2_garch": np.random.uniform(0.0001, 0.001, 100),
            "sigma_garch": np.sqrt(np.random.uniform(0.0001, 0.001, 100)),
            "std_resid_garch": np.random.normal(0, 1, 100),
        }
    )


def test_calculate_rsi(sample_price_data: pd.DataFrame) -> None:
    """Test RSI calculation."""
    prices = cast(pd.Series, sample_price_data["weighted_closing"])
    rsi = calculate_rsi(prices, period=14)

    assert len(rsi) == len(prices)
    assert rsi.iloc[0:13].isna().all()  # First 13 values should be NaN
    assert not rsi.iloc[13:].isna().any()  # Rest should be valid
    # Check RSI values are in valid range (excluding NaN)
    rsi_valid = rsi.dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()


def test_calculate_sma(sample_price_data: pd.DataFrame) -> None:
    """Test SMA calculation."""
    prices = cast(pd.Series, sample_price_data["weighted_closing"])
    sma = calculate_sma(prices, period=20)

    assert len(sma) == len(prices)
    assert sma.iloc[0:19].isna().all()  # First 19 values should be NaN
    assert not sma.iloc[19:].isna().any()  # Rest should be valid
    assert sma.iloc[19] == pytest.approx(prices.iloc[0:20].mean())


def test_calculate_ema(sample_price_data: pd.DataFrame) -> None:
    """Test EMA calculation."""
    prices = cast(pd.Series, sample_price_data["weighted_closing"])
    ema = calculate_ema(prices, period=20)

    assert len(ema) == len(prices)
    assert ema.iloc[0:19].isna().all()  # First 19 values should be NaN
    assert not ema.iloc[19:].isna().any()  # Rest should be valid


def test_calculate_macd(sample_price_data: pd.DataFrame) -> None:
    """Test MACD calculation."""
    prices = cast(pd.Series, sample_price_data["weighted_closing"])
    macd_line, signal_line, histogram = calculate_macd(prices, fast=12, slow=26, signal=9)

    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(histogram) == len(prices)
    assert histogram.equals(macd_line - signal_line)


def _check_bollinger_bands_length(
    upper: pd.Series, middle: pd.Series, lower: pd.Series, prices: pd.Series
) -> None:
    """Check that Bollinger Bands have correct length."""
    assert len(upper) == len(prices)
    assert len(middle) == len(prices)
    assert len(lower) == len(prices)


def _check_bollinger_bands_relationships(
    upper: pd.Series, middle: pd.Series, lower: pd.Series
) -> None:
    """Check that Bollinger Bands relationships are correct."""
    valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
    assert (upper[valid_mask] >= middle[valid_mask]).all()
    assert (middle[valid_mask] >= lower[valid_mask]).all()


def test_calculate_bollinger_bands(sample_price_data: pd.DataFrame) -> None:
    """Test Bollinger Bands calculation."""
    prices = cast(pd.Series, sample_price_data["weighted_closing"])
    upper, middle, lower = calculate_bollinger_bands(prices, period=20, num_std=2.0)

    _check_bollinger_bands_length(upper, middle, lower, prices)
    _check_bollinger_bands_relationships(upper, middle, lower)


def test_add_technical_indicators(sample_price_data: pd.DataFrame) -> None:
    """Test adding technical indicators to DataFrame."""
    df = add_technical_indicators(sample_price_data)

    expected_indicators = [
        "rsi_14",
        "sma_20",
        "ema_20",
        "macd",
        "macd_signal",
        "macd_histogram",
    ]

    for indicator in expected_indicators:
        assert indicator in df.columns, f"Missing indicator: {indicator}"

    # Verify BB columns are NOT present
    bb_columns = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"]
    for bb_col in bb_columns:
        assert bb_col not in df.columns, f"BB column {bb_col} should not be present"


def _check_dataset_structure(df_complete: pd.DataFrame, df_without: pd.DataFrame) -> None:
    """Check basic dataset structure."""
    assert len(df_complete) == len(df_without)
    assert len(df_complete.columns) > len(df_without.columns)


def _check_weighted_return_removed(df_complete: pd.DataFrame, df_without: pd.DataFrame) -> None:
    """Check that obsolete columns were removed and targets retained."""

    for dataset in (df_complete, df_without):
        assert "weighted_return" not in dataset.columns
        assert "weighted_closing" not in dataset.columns
        assert "weighted_open" not in dataset.columns
        assert "weighted_log_return" in dataset.columns
        assert "weighted_log_return_t" in dataset.columns
        for lag in RF_LAG_WINDOWS:
            assert f"weighted_closing_lag_{lag}" not in dataset.columns
            assert f"weighted_open_lag_{lag}" not in dataset.columns


def _check_nan_rows_removed(
    df_complete: pd.DataFrame, df_without: pd.DataFrame, original_len: int
) -> None:
    """Check that rows with NaN values were removed."""
    assert len(df_complete) < original_len
    assert len(df_without) < original_len
    assert len(df_complete) == len(df_without)


def _check_no_nan_in_technical_indicators(
    df_complete: pd.DataFrame, df_without: pd.DataFrame
) -> None:
    """Check that there are no NaN values in technical indicator columns."""
    technical_cols = ["rsi_14", "sma_20", "ema_20", "macd", "macd_signal", "macd_histogram"]
    for col in technical_cols:
        if col in df_complete.columns:
            has_nan_complete = bool(df_complete[col].isna().any())
            has_nan_without = bool(df_without[col].isna().any())
            assert not has_nan_complete, f"Column {col} contains NaN values in complete dataset"
            assert (
                not has_nan_without
            ), f"Column {col} contains NaN values in dataset without insights"


def _check_arima_garch_insights(
    df_complete: pd.DataFrame,
    df_without: pd.DataFrame,
    original_data: pd.DataFrame,
) -> None:
    """Check that ARIMA-GARCH insights are present in complete and absent in baseline."""
    for col in RF_ARIMA_GARCH_INSIGHT_COLUMNS:
        if col in original_data.columns:
            assert col in df_complete.columns
            assert col not in df_without.columns


def _check_technical_indicators_present(
    df_complete: pd.DataFrame, df_without: pd.DataFrame
) -> None:
    """Check that technical indicators are in both datasets."""
    assert "rsi_14" in df_complete.columns
    assert "rsi_14" in df_without.columns


def _check_bb_columns_not_present(df_complete: pd.DataFrame, df_without: pd.DataFrame) -> None:
    """Check that BB columns are NOT present."""
    bb_columns = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"]
    for bb_col in bb_columns:
        assert bb_col not in df_complete.columns, f"BB column {bb_col} should not be present"
        assert bb_col not in df_without.columns, f"BB column {bb_col} should not be present"


def _check_output_files_created(tmp_path: Path) -> None:
    """Check that output files were created."""
    assert (tmp_path / "rf_dataset_complete.csv").exists()
    assert (tmp_path / "rf_dataset_without_insights.csv").exists()


def test_prepare_datasets(sample_price_data: pd.DataFrame, tmp_path: Path) -> None:
    """Test dataset preparation."""
    # Add weighted_return to test that it gets removed
    sample_price_data = sample_price_data.copy()
    sample_price_data["weighted_return"] = np.random.randn(len(sample_price_data))
    original_len = len(sample_price_data)

    df_complete, df_without = prepare_datasets(df=sample_price_data, output_dir=tmp_path)

    _check_dataset_structure(df_complete, df_without)
    _check_weighted_return_removed(df_complete, df_without)
    _check_nan_rows_removed(df_complete, df_without, original_len)
    _check_no_nan_in_technical_indicators(df_complete, df_without)
    _check_arima_garch_insights(df_complete, df_without, sample_price_data)
    _check_technical_indicators_present(df_complete, df_without)
    _check_bb_columns_not_present(df_complete, df_without)
    _check_output_files_created(tmp_path)


def test_load_garch_data_missing_file() -> None:
    """Test loading GARCH data with missing file."""
    from pathlib import Path

    fake_path = Path("/nonexistent/path/garch_variance.csv")
    with pytest.raises(FileNotFoundError):
        load_garch_data(fake_path)


def test_prepare_datasets_without_insight_columns(tmp_path: Path) -> None:
    """Test dataset preparation when insight columns are missing (should still work)."""
    # Need enough rows to have data after removing rows with NaN values
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "weighted_closing": np.random.randn(100),
            "weighted_open": np.random.randn(100),
            "weighted_log_return": np.random.randn(100) * 0.01,
        }
    )

    # Should work fine - missing insight columns will just be skipped
    df_complete, df_without = prepare_datasets(df=df, output_dir=tmp_path)
    assert len(df_complete) == len(df_without)
    # Check that rows with NaN were removed
    assert len(df_complete) < len(df)
    assert len(df_complete) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
