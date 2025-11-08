"""Unit tests for identification step (self-runnable)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.structure_garch.detection import (
    compute_arch_lm_test,
    compute_squared_acf,
    detect_heteroskedasticity,
    plot_arch_diagnostics,
    prepare_residuals,
)


def test_compute_squared_acf_pattern() -> None:
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)
    acf = compute_squared_acf(resid, nlags=5)
    assert len(acf) == 5
    assert abs(acf[1]) > 0.5


def test_arch_lm_pvalue_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)

    def fake_sf(x: float, df: int) -> float:
        return 0.001

    monkeypatch.setattr("src.garch.structure_garch.detection._chi2_sf", fake_sf, raising=True)
    out = compute_arch_lm_test(resid, lags=4)
    assert out["df"] == 4
    assert out["p_value"] == pytest.approx(0.001)


def test_detect_heteroskedasticity_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 100)

    def fake_sf(x: float, df: int) -> float:
        return 0.001

    monkeypatch.setattr("src.garch.structure_garch.detection._chi2_sf", fake_sf, raising=True)
    out = detect_heteroskedasticity(resid, lags=4, acf_lags=10, alpha=0.05)
    assert out["arch_effect_present"] is True
    assert out["acf_significant"] is True
    assert isinstance(out["acf_squared"], list)
    assert len(cast(list, out["acf_squared"])) == 10


def test_prepare_residuals_requires_arima_column() -> None:
    # No 'arima_residual_return' column -> must raise
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "split": ["test"] * 5,
            "weighted_log_return": np.random.randn(5) * 0.01,
        }
    )
    with pytest.raises(ValueError):
        prepare_residuals(df, use_test_only=True)


def test_plot_arch_diagnostics_writes_file(tmp_path: Path) -> None:
    resid = np.array([1.0, -1.0, 2.0, -2.0] * 50)
    out_file = tmp_path / "diag.png"
    path = plot_arch_diagnostics(resid, acf_lags=10, out_path=out_file)
    assert path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
