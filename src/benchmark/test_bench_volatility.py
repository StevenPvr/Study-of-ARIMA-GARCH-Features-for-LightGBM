"""Unit tests for volatility backtest baselines and pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src import constants as C
from src.benchmark.bench_volatility import run_vol_backtest, save_vol_backtest_outputs


def _make_df(n_train: int = 80, n_test: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 0.01, size=n_train)
    test = rng.normal(0.0, 0.02, size=n_test)  # different vol regime
    dates = pd.date_range("2020-01-01", periods=n_train + n_test, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * n_train + ["test"] * n_test,
            "weighted_log_return": np.concatenate([train, test]),
            "arima_pred_return": 0.0,
        }
    )


def _assert_forecasts_structure(forecasts: pd.DataFrame, expected_n: int) -> None:
    """Assert forecasts DataFrame structure."""
    assert isinstance(forecasts, pd.DataFrame)
    assert forecasts.shape[0] == expected_n


def _assert_required_columns(forecasts: pd.DataFrame) -> None:
    """Assert required columns exist and are finite."""
    required_cols = {
        "s2_arima_garch",
        "s2_ewma",
        "s2_roll_var",
        "s2_roll_std",
        "s2_arch1",
        "s2_har3",
    }
    assert required_cols.issubset(set(forecasts.columns))
    assert np.all(np.isfinite(forecasts[list(required_cols)].to_numpy()))


def _assert_metrics_structure(metrics: dict, expected_n: int) -> None:
    """Assert metrics dictionary structure."""
    assert isinstance(metrics, dict)
    assert metrics.get("n_test") == expected_n
    assert "arima_garch" in metrics
    assert "ewma" in metrics
    # Check volatility metrics are present
    vol_metrics = {"qlike", "mse", "mae", "rmse", "r2"}
    assert vol_metrics.issubset(set(metrics["ewma"].keys()))


def _assert_backtest_outputs(forecasts: pd.DataFrame, metrics: dict, df: pd.DataFrame) -> None:
    """Assert backtest outputs structure and content."""
    expected_n = int((df["split"] == "test").sum())
    _assert_forecasts_structure(forecasts, expected_n)
    _assert_required_columns(forecasts)
    _assert_metrics_structure(metrics, expected_n)


def test_run_vol_backtest_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df()

    # Monkeypatch GARCH dataset loader path used inside rolling runner
    monkeypatch.setattr(C, "GARCH_DATASET_FILE", Path("/dev/null"))
    # run_vol_backtest passes df directly to rolling runner via artifacts;
    # we avoid file use by patching. Patch run_from_artifacts to a simple identity
    # baseline to avoid SciPy heavy calls in this test.
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.full_like(
            e_test, float(np.var(df[df["split"] == "train"]["weighted_log_return"]))
        ).astype(float)
        fore = pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2})
        metrics = {"refit_count": 0}
        return fore, metrics

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    forecasts, metrics = run_vol_backtest(df, ewma_lambda=0.94, rolling_window=10)

    _assert_backtest_outputs(forecasts, metrics, df)


def test_save_vol_backtest_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df(n_train=50, n_test=12)

    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        fore = pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2})
        return fore, {"refit_count": 0}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    forecasts, metrics = run_vol_backtest(df)

    monkeypatch.setattr(C, "VOL_BACKTEST_FORECASTS_FILE", tmp_path / "vol.csv", raising=False)
    monkeypatch.setattr(C, "VOL_BACKTEST_METRICS_FILE", tmp_path / "vol.json", raising=False)
    save_vol_backtest_outputs(forecasts, metrics)
    assert (tmp_path / "vol.csv").exists()
    assert (tmp_path / "vol.json").exists()


def test_run_vol_backtest_invalid_ewma_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that invalid ewma_lambda raises ValueError."""
    df = _make_df()
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        return pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    with pytest.raises(ValueError, match="ewma_lambda must be in"):
        run_vol_backtest(df, ewma_lambda=1.5, rolling_window=10)


def test_run_vol_backtest_invalid_rolling_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that invalid rolling_window raises ValueError."""
    df = _make_df()
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        return pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    with pytest.raises(ValueError, match="rolling_window must be"):
        run_vol_backtest(df, ewma_lambda=0.94, rolling_window=0)


def test_run_vol_backtest_invalid_var_alphas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that invalid var_alphas raises ValueError (still validates if provided)."""
    df = _make_df()
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        return pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    # var_alphas is still validated but not used in metrics (VaR removed)
    with pytest.raises(ValueError, match="var_alphas must be"):
        run_vol_backtest(df, var_alphas=[1.5])


def test_run_vol_backtest_missing_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that missing required columns raises ValueError."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D")})
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        return pd.DataFrame({"date": [], "e": [], "sigma2_forecast": []}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    with pytest.raises(ValueError, match="must contain 'date' and 'split'"):
        run_vol_backtest(df)


def test_run_vol_backtest_no_test_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that no test data raises ValueError."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "split": ["train"] * 10,
            "weighted_log_return": np.random.normal(0, 0.01, 10),
        }
    )
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        return pd.DataFrame({"date": [], "e": [], "sigma2_forecast": []}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    with pytest.raises(ValueError, match="No valid test observations"):
        run_vol_backtest(df)


def test_run_vol_backtest_edge_case_small_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test with minimal rolling window."""
    df = _make_df(n_train=20, n_test=5)
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        return pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    forecasts, metrics = run_vol_backtest(df, ewma_lambda=0.95, rolling_window=1)
    _assert_backtest_outputs(forecasts, metrics, df)


def test_run_vol_backtest_edge_case_boundary_ewma(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test with boundary ewma_lambda values."""
    df = _make_df()
    import src.benchmark.bench_volatility as mod

    def fake_run_from_artifacts(**kwargs):
        e_test = np.asarray(df.loc[df["split"] == "test", "weighted_log_return"].values)
        dates = np.asarray(df.loc[df["split"] == "test", "date"].values)
        s2 = np.ones_like(e_test) * 0.0001
        return pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2}), {}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    # Test with very small lambda (close to 0)
    forecasts, metrics = run_vol_backtest(df, ewma_lambda=0.01, rolling_window=10)
    _assert_backtest_outputs(forecasts, metrics, df)

    # Test with large lambda (close to 1)
    forecasts, metrics = run_vol_backtest(df, ewma_lambda=0.99, rolling_window=10)
    _assert_backtest_outputs(forecasts, metrics, df)


def test_run_vol_backtest_auto_load_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When df is None, the function should auto-load the GARCH dataset."""
    # Build a minimal but valid synthetic GARCH dataset
    n_train, n_test = 30, 12
    dates = pd.date_range("2021-01-01", periods=n_train + n_test, freq="D")
    rng = np.random.default_rng(123)
    wl_train = rng.normal(0.0, 0.01, size=n_train)
    wl_test = rng.normal(0.0, 0.02, size=n_test)
    wl_all = np.concatenate([wl_train, wl_test]).astype(float)
    df_auto = pd.DataFrame(
        {
            "date": dates,
            "split": ["train"] * n_train + ["test"] * n_test,
            "weighted_log_return": wl_all,
            "arima_pred_return": 0.0,
            # Provide residuals column to exercise preferred path
            "arima_residual_return": wl_all,
        }
    )

    # Patch loader inside the module to return our synthetic dataset
    import src.benchmark.bench_volatility as mod

    monkeypatch.setattr(mod, "load_garch_dataset", lambda: df_auto)

    # Avoid heavy EGARCH code by patching artifacts runner
    def fake_run_from_artifacts(**kwargs):
        e_test = df_auto.loc[df_auto["split"] == "test", "weighted_log_return"].to_numpy()
        dates_test = df_auto.loc[df_auto["split"] == "test", "date"].to_numpy()
        s2 = np.ones_like(e_test) * 0.0001
        fore = pd.DataFrame({"date": dates_test, "e": e_test, "sigma2_forecast": s2})
        return fore, {"refit_count": 0}

    monkeypatch.setattr(mod, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)

    # Call without df -> should auto-load
    forecasts, metrics = run_vol_backtest()
    _assert_backtest_outputs(forecasts, metrics, df_auto)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
