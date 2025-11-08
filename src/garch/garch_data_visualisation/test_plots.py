"""Unit tests for visualization step (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.garch_data_visualisation.plots import (
    plot_returns_autocorrelation,
    save_returns_and_squared_plots,
)


def test_save_returns_and_squared(tmp_path: Path) -> None:
    """Test basic returns plotting functionality."""
    r = np.array([0.01, -0.02, 0.03, -0.01] * 50)
    save_returns_and_squared_plots(r, outdir=tmp_path, filename="ret.png")


def test_save_returns_and_squared_with_dates(tmp_path: Path) -> None:
    """Test returns plotting with dates."""
    r = np.array([0.01, -0.02, 0.03, -0.01] * 50)
    dates = pd.date_range("2020-01-01", periods=len(r), freq="D").to_series()
    save_returns_and_squared_plots(r, dates=dates, outdir=tmp_path, filename="ret_dates.png")


def test_save_returns_and_squared_empty(tmp_path: Path) -> None:
    """Test returns plotting with empty array (should handle gracefully)."""
    r = np.array([])
    result = save_returns_and_squared_plots(r, outdir=tmp_path, filename="ret_empty.png")
    # Function returns Path but doesn't create file when data is insufficient
    assert isinstance(result, Path)
    assert not result.exists()


def test_save_returns_and_squared_with_nan(tmp_path: Path) -> None:
    """Test returns plotting with NaN values (should filter them)."""
    r = np.array([0.01, np.nan, 0.03, -0.01, np.inf, -np.inf] * 20)
    save_returns_and_squared_plots(r, outdir=tmp_path, filename="ret_nan.png")


def test_save_returns_and_squared_none_raises(tmp_path: Path) -> None:
    """Test that None input raises ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        save_returns_and_squared_plots(None, outdir=tmp_path, filename="ret.png")  # type: ignore[arg-type]


def test_plot_returns_autocorrelation(tmp_path: Path) -> None:
    """Test returns autocorrelation plot generation."""
    returns = np.array([0.01, -0.02, 0.03, -0.01] * 80)
    plot_returns_autocorrelation(returns, lags=10, outdir=tmp_path, filename="acf.png")


def test_plot_returns_autocorrelation_none_raises(tmp_path: Path) -> None:
    """Test that None input raises ValueError."""
    with pytest.raises(ValueError, match="cannot be None"):
        plot_returns_autocorrelation(
            None, lags=10, outdir=tmp_path, filename="acf.png"  # type: ignore[arg-type]
        )


def test_plot_returns_autocorrelation_invalid_lags(tmp_path: Path) -> None:
    """Test that invalid lags raises ValueError."""
    returns = np.array([0.01, -0.02, 0.03] * 30)
    with pytest.raises(ValueError, match="must be positive"):
        plot_returns_autocorrelation(returns, lags=0, outdir=tmp_path, filename="acf.png")


def test_plot_returns_autocorrelation_insufficient_data(tmp_path: Path) -> None:
    """Test autocorrelation plot with insufficient data (should handle gracefully)."""
    returns = np.array([0.01, -0.02])  # Only 2 values, need at least 3
    result = plot_returns_autocorrelation(
        returns, lags=8, outdir=tmp_path, filename="acf_insufficient.png"
    )
    # Function returns Path but doesn't create file when data is insufficient
    assert isinstance(result, Path)
    assert not result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
