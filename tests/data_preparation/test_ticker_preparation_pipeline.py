from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.ticker_preparation import split_tickers_train_test


def _make_ticker_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    df_a = pd.DataFrame(
        {
            "ticker": ["AAA"] * 8,
            "date": dates,
            "open": np.linspace(10, 17, 8),
            "high": np.linspace(11, 18, 8),
            "low": np.linspace(9, 16, 8),
            "close": np.linspace(10, 17, 8),
            "volume": np.arange(100, 108),
        }
    )
    # Precompute a simple log_return to satisfy compute_volatility prerequisites
    df_a = df_a.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_a["log_return"] = np.log(df_a["close"] / df_a["close"].shift(1))
    df_a = df_a.dropna(subset=["log_return"]).reset_index(drop=True)
    return df_a


def test_lgbm_pipeline_outputs_expected_columns(tmp_path):
    df = _make_ticker_df()
    in_path = tmp_path / "dataset_filtered.csv"
    out_path = tmp_path / "data_tickers_full.parquet"
    df.to_csv(in_path, index=False)

    split_tickers_train_test(
        train_ratio=0.75,
        input_file=str(in_path),
        output_file=str(out_path),
    )

    # Load saved output (CSV is also created alongside parquet)
    out_csv = out_path.with_suffix(".csv")
    out_df = pd.read_csv(out_csv, parse_dates=["date"])

    # Columns must match requested schema
    expected_cols = [
        "date",
        "tickers",
        "close",
        "high",
        "low",
        "volume",
        "log_return",
        "log_volatility",
        "split",
    ]
    assert list(out_df.columns) == expected_cols

    # No weighted returns should be present
    assert "weighted_log_return" not in out_df.columns

    # Check sorting and time split per ticker
    for t in out_df["tickers"].unique():
        tdf = out_df[out_df["tickers"] == t].copy()
        tdf_dates = pd.Series(tdf["date"])
        assert tdf_dates.is_monotonic_increasing
        train = pd.DataFrame(tdf[tdf["split"] == "train"])
        test = pd.DataFrame(tdf[tdf["split"] == "test"])
        if not train.empty and not test.empty:
            train_dates = pd.Series(train["date"])
            test_dates = pd.Series(test["date"])
            assert train_dates.max() < test_dates.min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
