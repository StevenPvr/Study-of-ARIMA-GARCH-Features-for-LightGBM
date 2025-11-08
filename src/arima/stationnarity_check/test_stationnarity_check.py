from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.arima.stationnarity_check.stationnarity_check import (
    evaluate_stationarity,
    run_stationarity_pipeline,
    save_stationarity_report,
)


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_evaluate_stationarity_on_white_noise() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=600)
    s = pd.Series(y, index=_make_dates(600))
    rep = evaluate_stationarity(s, alpha=0.05)
    assert isinstance(rep.stationary, bool)
    assert rep.adf["p_value"] < 0.05
    # KPSS sometimes returns borderline values; we allow >= 0.05
    assert np.isnan(rep.kpss["p_value"]) or rep.kpss["p_value"] >= 0.05
    assert rep.stationary is True


def test_evaluate_stationarity_on_random_walk() -> None:
    rng = np.random.default_rng(123)
    eps = rng.normal(0.0, 1.0, size=600)
    rw = np.cumsum(eps)
    s = pd.Series(rw, index=_make_dates(600))
    rep = evaluate_stationarity(s, alpha=0.05)
    # Expect non-stationary
    assert rep.stationary is False


def test_pipeline_and_save_json(tmp_path: Path) -> None:
    # Build a synthetic stationary series and persist as CSV
    n = 300
    rng = np.random.default_rng(7)
    y = rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({
        "date": _make_dates(n),
        "weighted_log_return": y,
    })
    csv_path = tmp_path / "mock.csv"
    df.to_csv(csv_path, index=False)

    rep = run_stationarity_pipeline(data_file=str(csv_path), column="weighted_log_return")
    assert rep.stationary is True

    out = tmp_path / "report.json"
    save_path = save_stationarity_report(rep, out)
    assert save_path.exists()
    loaded = json.loads(save_path.read_text())
    assert isinstance(loaded.get("stationary"), bool)


def test_pipeline_raises_for_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"date": _make_dates(10), "x": np.arange(10)})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Column 'weighted_log_return' not found"):
        run_stationarity_pipeline(data_file=str(p), column="weighted_log_return")


def test_evaluate_stationarity_raises_for_invalid_alpha() -> None:
    rng = np.random.default_rng(42)
    y = rng.normal(0.0, 1.0, size=100)
    s = pd.Series(y, index=_make_dates(100))
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=1.5)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
        evaluate_stationarity(s, alpha=-0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])