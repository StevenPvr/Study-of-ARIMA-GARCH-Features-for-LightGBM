"""Integration tests across cleaning → conversion → split.

These tests use synthetic data and tmp paths only (no network).
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import pytest

from src.data_cleaning import data_cleaning as dc
from src.data_conversion import data_conversion as dv
from src.data_preparation import data_preparation as dp


def _make_sample_dataset() -> pd.DataFrame:
    """Build a small, valid synthetic dataset for two tickers.

    Returns:
        DataFrame with columns: date, ticker, open, closing, volume.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    def rows_for(ticker: str, n: int, base: float) -> pd.DataFrame:
        d = pd.DataFrame(
            {
                "date": dates[:n],
                "ticker": ticker,
                "open": base - 1 + pd.RangeIndex(n).astype(float) * 0.1,
                "closing": base + pd.RangeIndex(n).astype(float) * 1.0,
                "volume": 1_000 + pd.RangeIndex(n) * 10,
            }
        )
        return d

    aaa = rows_for("AAA", 5, 100.0)
    bbb = rows_for("BBB", 4, 50.0)
    ccc = rows_for("CCC", 2, 10.0)  # will be filtered out (< min obs)
    return pd.concat([aaa, bbb, ccc], ignore_index=True)


def test_integration_clean_convert_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs cleaning → conversion → temporal split on synthetic data.

    Validates that cleaning filters incomplete tickers, conversion writes
    weighted returns and prices, and preparation splits into train/test.
    """
    # 1) Prepare synthetic raw dataset in tmpdir
    raw_df = _make_sample_dataset()
    dataset_file = tmp_path / "dataset.csv"
    raw_df.to_csv(dataset_file, index=False)

    # 2) Point data_cleaning module constants to tmp files and relax threshold
    filtered_file = tmp_path / "dataset_filtered.csv"
    monkeypatch.setattr(dc, "DATASET_FILE", dataset_file, raising=False)
    monkeypatch.setattr(dc, "DATASET_FILTERED_FILE", filtered_file, raising=False)
    monkeypatch.setattr(dc, "MIN_OBSERVATIONS_PER_TICKER", 3, raising=False)

    # 3) Clean (filters out CCC which has only 2 obs)
    dc.filter_incomplete_tickers()
    assert filtered_file.exists(), "Filtered dataset not created"
    filtered_df = pd.read_csv(filtered_file)
    assert set(filtered_df["ticker"].unique()) == {"AAA", "BBB"}
    # basic sanity: no zero/negative critical fields
    assert (filtered_df[["open", "closing", "volume"]] > 0).all().all()

    # 4) Convert to weighted log returns and prices (explicit output paths)
    weights_file = tmp_path / "liquidity_weights.csv"
    returns_file = tmp_path / "weighted_log_returns.csv"
    dv.compute_weighted_log_returns(
        input_file=str(filtered_file),
        weights_output_file=str(weights_file),
        returns_output_file=str(returns_file),
    )
    assert weights_file.exists(), "Weights CSV was not written"
    assert returns_file.exists(), "Weighted returns CSV was not written"

    returns_df = pd.read_csv(returns_file, parse_dates=["date"])  # has weighted_open/closing
    for col in ("weighted_log_return", "weighted_open", "weighted_closing"):
        assert col in returns_df.columns
        assert returns_df[col].isna().sum() == 0
    # Expect at least 4 dates (AAA contributes 4, BBB 3, combined unique should be 4)
    assert len(returns_df) >= 4
    assert returns_df["date"].is_monotonic_increasing

    # 5) Temporal split with an 80/20 ratio to keep constraints valid
    split_file = tmp_path / "weighted_log_returns_split.csv"
    dp.split_train_test(
        train_ratio=0.8,
        input_file=str(returns_file),
        output_file=str(split_file),
    )
    assert split_file.exists(), "Split CSV was not written"
    split_df = pd.read_csv(split_file)
    assert set(split_df["split"].unique()) == {"train", "test"}
    # Time ordering preserved within each split
    train = split_df[split_df["split"] == "train"]["date"].tolist()
    test = split_df[split_df["split"] == "test"]["date"].tolist()
    assert train == sorted(train)
    assert test == sorted(test)


def test_integration_data_fetching_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: data_fetching with mocked web and yfinance."""
    from src.data_fetching import data_fetching as dfetch

    tickers_file = tmp_path / "sp500_tickers.csv"
    dataset_file = tmp_path / "dataset.csv"
    monkeypatch.setattr(dfetch, "SP500_TICKERS_FILE", tickers_file, raising=False)
    monkeypatch.setattr(dfetch, "DATASET_FILE", dataset_file, raising=False)
    monkeypatch.setattr(dfetch, "DATA_DIR", tmp_path, raising=False)

    html = (
        "<table><thead><tr><th>Symbol</th></tr></thead>"
        "<tbody><tr><td>AAA</td></tr><tr><td>BBB</td></tr></tbody></table>"
    )

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return html.encode("utf-8")

    import urllib.request as _url

    monkeypatch.setattr(_url, "urlopen", lambda *a, **k: _Resp())

    import sys as _sys
    import numpy as np
    import pandas as pd

    yf = _sys.modules.get("yfinance")

    def _yf_download(ticker: str, start=None, end=None, progress=False, auto_adjust=True):
        n = {"AAA": 4, "BBB": 3}[ticker]
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {
                "Open": np.linspace(10.0, 10.0 + n - 1, n),
                "Close": np.linspace(10.1, 10.1 + n - 1, n),
                "Volume": 1000 + np.arange(n),
            },
            index=idx,
        )

    setattr(yf, "download", _yf_download)

    dfetch.fetch_sp500_tickers()
    dfetch.download_sp500_data()
    assert dataset_file.exists() and dataset_file.stat().st_size > 0


def test_integration_sarima_train_evaluate_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: SARIMA optimize→train→evaluate on synthetic returns."""
    # Build synthetic weighted returns
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2022-01-01", periods=20, freq="D")
    vals = np.sin(np.linspace(0, 2 * np.pi, 20)) * 0.01
    returns_df = pd.DataFrame({"date": dates, "weighted_log_return": vals})
    returns_file = tmp_path / "weighted_log_returns.csv"
    returns_df.to_csv(returns_file, index=False)

    split_file = tmp_path / "weighted_log_returns_split.csv"
    dp.split_train_test(train_ratio=0.8, input_file=str(returns_file), output_file=str(split_file))
    train_series, test_series = dp.load_train_test_data(input_file=str(split_file))

    from src.arima.optimisation_arima import optimisation_arima as aopt
    from src.arima.training_arima import training_arima as atrain
    from src.arima.evaluation_arima import evaluation_arima as aeval

    best_models_path = tmp_path / "best_models.json"
    opt_results_path = tmp_path / "arima_opt_results.csv"
    monkeypatch.setattr(aopt, "SARIMA_BEST_MODELS_FILE", best_models_path, raising=False)
    monkeypatch.setattr(aopt, "SARIMA_OPTIMIZATION_RESULTS_FILE", opt_results_path, raising=False)
    monkeypatch.setattr(aopt, "WEIGHTED_LOG_RETURNS_SPLIT_FILE", split_file, raising=False)
    aopt.optimize_sarima_models(train_series, test_series, p_range=range(2), d_range=range(2), q_range=range(2))
    assert best_models_path.exists()

    monkeypatch.setattr(atrain, "SARIMA_BEST_MODELS_FILE", best_models_path, raising=False)
    monkeypatch.setattr(atrain, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(atrain, "TRAINED_MODEL_FILE", tmp_path / "results" / "models" / "arima.pkl", raising=False)
    fitted_model, model_info = atrain.train_best_model(train_series, prefer="aic")
    atrain.save_trained_model(fitted_model, model_info)
    loaded_model, _ = atrain.load_trained_model()
    assert loaded_model is not None

    preds_dir = tmp_path / "results"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(aeval, "RESULTS_DIR", preds_dir, raising=False)
    monkeypatch.setattr(aeval, "ROLLING_PREDICTIONS_SARIMA_FILE", preds_dir / "rolling_predictions.csv", raising=False)
    monkeypatch.setattr(aeval, "ROLLING_VALIDATION_METRICS_SARIMA_FILE", preds_dir / "rolling_metrics.json", raising=False)
    monkeypatch.setattr(aeval, "LJUNGBOX_RESIDUALS_SARIMA_FILE", preds_dir / "ljungbox.json", raising=False)
    monkeypatch.setattr(aeval, "WEIGHTED_LOG_RETURNS_SPLIT_FILE", split_file, raising=False)
    results = aeval.evaluate_model(train_series, test_series, order=(0, 0, 1), seasonal_order=(0, 0, 0, 12))
    aeval.save_evaluation_results(results)
    aeval.save_ljung_box_results({"lags": [1], "q_stat": [0.0], "p_value": [1.0], "reject_5pct": False, "n": 1})
    assert (preds_dir / "rolling_predictions.csv").exists()
    assert (preds_dir / "rolling_metrics.json").exists()
    assert (preds_dir / "ljungbox.json").exists()


def _setup_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install minimal matplotlib stubs for visualization smoke tests."""
    import sys as _sys
    from types import ModuleType

    class _Axes:
        def plot(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

        def legend(self, *a, **k):
            return None

        def axhline(self, *a, **k):
            return None

    class _Fig:
        def __init__(self, *a, **k):
            pass

    def _save(path: str, *a, **k):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    fake_mpl = ModuleType("matplotlib")
    fake_axes = ModuleType("matplotlib.axes")
    setattr(fake_axes, "Axes", _Axes)
    fake_pyplot = ModuleType("matplotlib.pyplot")

    def _subplots(*a, **k):
        nrows = 1
        ncols = 1
        if a and isinstance(a[0], int):
            nrows = int(a[0])
        if len(a) >= 2 and isinstance(a[1], int):
            ncols = int(a[1])
        nrows = int(k.get("nrows", nrows))
        ncols = int(k.get("ncols", ncols))
        fig = _Fig()
        if nrows == 1 and ncols == 1:
            return fig, _Axes()
        if nrows == 1:
            return fig, [_Axes() for _ in range(ncols)]
        if ncols == 1:
            return fig, [_Axes() for _ in range(nrows)]
        return fig, [[_Axes() for _ in range(ncols)] for _ in range(nrows)]

    setattr(fake_pyplot, "subplots", _subplots)
    setattr(fake_pyplot, "tight_layout", lambda *a, **k: None)
    setattr(fake_pyplot, "savefig", _save)
    setattr(fake_pyplot, "close", lambda *a, **k: None)

    monkeypatch.setitem(_sys.modules, "matplotlib", fake_mpl)
    monkeypatch.setitem(_sys.modules, "matplotlib.axes", fake_axes)
    monkeypatch.setitem(_sys.modules, "matplotlib.pyplot", fake_pyplot)


def test_integration_garch_benchmark_visualisation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: GARCH train/eval/rolling + benchmark + viz (offline)."""
    # 1) Build synthetic returns and split
    dates = pd.date_range("2022-01-01", periods=24, freq="D")
    vals = __import__('numpy').sin(__import__('numpy').linspace(0, 2 * __import__('numpy').pi, 24)) * 0.01
    returns_df = pd.DataFrame({"date": dates, "weighted_log_return": vals})
    returns_file = tmp_path / "weighted_log_returns.csv"
    returns_df.to_csv(returns_file, index=False)

    split_file = tmp_path / "weighted_log_returns_split.csv"
    dp.split_train_test(train_ratio=0.8, input_file=str(returns_file), output_file=str(split_file))
    train_series, test_series = dp.load_train_test_data(input_file=str(split_file))

    # 2) SARIMA evaluate (minimal) to get eval_results
    from src.arima.evaluation_arima import evaluation_arima as aeval

    preds_dir = tmp_path / "results"
    preds_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(aeval, "RESULTS_DIR", preds_dir, raising=False)
    monkeypatch.setattr(aeval, "ROLLING_PREDICTIONS_SARIMA_FILE", preds_dir / "rolling_predictions.csv", raising=False)
    monkeypatch.setattr(aeval, "ROLLING_VALIDATION_METRICS_SARIMA_FILE", preds_dir / "rolling_metrics.json", raising=False)
    monkeypatch.setattr(aeval, "LJUNGBOX_RESIDUALS_SARIMA_FILE", preds_dir / "ljungbox.json", raising=False)
    monkeypatch.setattr(aeval, "WEIGHTED_LOG_RETURNS_SPLIT_FILE", split_file, raising=False)

    eval_results = aeval.evaluate_model(train_series, test_series, order=(0, 0, 1))
    aeval.save_evaluation_results(eval_results)

    # Prepare GARCH dataset at tmp path (needs train and test residuals)
    from src.arima.training_arima import training_arima as atrain

    garch_dataset_file = tmp_path / "dataset_garch.csv"
    monkeypatch.setattr(aeval, "GARCH_DATASET_FILE", garch_dataset_file, raising=False)
    fitted_min = atrain.train_sarima_model(train_series, order=(0, 0, 0))
    aeval.save_garch_dataset(eval_results, fitted_model=fitted_min)
    assert garch_dataset_file.exists()
    garch_df = pd.read_csv(garch_dataset_file, parse_dates=["date"])  # type: ignore[arg-type]

    # 3) GARCH training from pre-estimated params
    from src.garch.training_garch import training as gtrain
    from src import constants as C

    garch_model_file = tmp_path / "results" / "models" / "garch_model.joblib"
    garch_metadata_file = tmp_path / "results" / "models" / "garch_model.json"
    garch_variance_file = tmp_path / "results" / "garch_variance.csv"
    monkeypatch.setattr(gtrain, "GARCH_MODEL_FILE", garch_model_file, raising=False)
    monkeypatch.setattr(gtrain, "GARCH_MODEL_METADATA_FILE", garch_metadata_file, raising=False)
    monkeypatch.setattr(gtrain, "GARCH_VARIANCE_OUTPUTS_FILE", garch_variance_file, raising=False)

    garch_estimation_file = tmp_path / "garch_estimation.json"
    monkeypatch.setattr(C, "GARCH_ESTIMATION_FILE", garch_estimation_file, raising=False)
    est = {
        "egarch_normal": {"omega": 1e-6, "alpha": 0.05, "gamma": 0.0, "beta": 0.9, "loglik": -100.0, "converged": True},
        "egarch_student": {"omega": 1e-6, "alpha": 0.06, "gamma": 0.0, "beta": 0.88, "nu": 8.0, "loglik": -99.0, "converged": True},
    }
    garch_estimation_file.write_text(__import__("json").dumps(est))

    gtrain.train_egarch_from_dataset(garch_df)
    assert garch_model_file.exists() and garch_variance_file.exists()

    # 4) GARCH evaluation (forecasts + metrics)
    from src.garch.garch_eval import eval as geval_eval
    from src.garch.garch_eval import metrics as geval_metrics

    garch_forecasts_file = tmp_path / "results" / "garch_forecasts.csv"
    garch_eval_metrics_file = tmp_path / "results" / "garch_eval_metrics.json"
    monkeypatch.setattr(geval_eval, "GARCH_FORECASTS_FILE", garch_forecasts_file, raising=False)
    monkeypatch.setattr(geval_eval, "GARCH_DATASET_FILE", garch_dataset_file, raising=False)
    monkeypatch.setattr(geval_eval, "GARCH_ESTIMATION_FILE", garch_estimation_file, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_EVAL_METRICS_FILE", garch_eval_metrics_file, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_DATASET_FILE", garch_dataset_file, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_VARIANCE_OUTPUTS_FILE", garch_variance_file, raising=False)

    fore_df = geval_eval.forecast_from_artifacts(horizon=3, level=0.95)
    assert isinstance(fore_df, pd.DataFrame) and garch_forecasts_file.exists()
    from src.garch.garch_eval.eval import _choose_best_from_estimation

    params, name, nu = _choose_best_from_estimation(est)
    dist = "student" if "student" in name else "normal"
    metr = geval_metrics.compute_classic_metrics_from_artifacts(
        params=params, model_name=name, dist=dist, nu=nu, alphas=[0.05]
    )
    geval_metrics.save_metrics_json(metr)
    assert garch_eval_metrics_file.exists()

    # 5) Rolling GARCH (minimal, using pre-fit stub)
    import src.garch.rolling_garch.rolling as rmod
    
    def _mock_fit_initial_params(resid_train, dist_preference="auto"):
        from src.garch.rolling_garch.rolling import GarchParams

        best = est["egarch_student"] if est["egarch_student"].get("converged") else est["egarch_normal"]
        dist_name = "student" if best.get("nu") else "normal"
        return GarchParams(
            omega=float(best["omega"]),
            alpha=float(best["alpha"]),
            beta=float(best["beta"]),
            gamma=float(best.get("gamma", 0.0)),
            nu=float(best["nu"]) if best.get("nu") else None,
            dist=dist_name,
            model="egarch",
        )

    monkeypatch.setattr(rmod, "_fit_initial_params", _mock_fit_initial_params)
    # save_rolling_outputs reads from src.constants at call-time
    monkeypatch.setattr(C, "GARCH_ROLLING_FORECASTS_FILE", tmp_path / "results" / "garch_rolling_forecasts.csv", raising=False)
    monkeypatch.setattr(C, "GARCH_ROLLING_EVAL_FILE", tmp_path / "results" / "garch_rolling_eval.json", raising=False)

    rolling_forecasts, rolling_metrics = rmod.run_rolling_garch(
        garch_df,
        refit_every=10,
        window="expanding",
        window_size=50,
        dist_preference="auto",
        keep_nu_between_refits=True,
        var_alphas=[0.05],
    )
    rmod.save_rolling_outputs(rolling_forecasts, rolling_metrics)
    assert C.GARCH_ROLLING_FORECASTS_FILE.exists()
    assert C.GARCH_ROLLING_EVAL_FILE.exists()

    # 6) Benchmark backtest (stubbed rolling)
    import src.benchmark.bench_volatility as bmk

    def fake_run_from_artifacts(**kwargs):
        import numpy as _np

        e_test = _np.asarray(garch_df.loc[garch_df["split"] == "test", "weighted_log_return"].values)
        dates = _np.asarray(garch_df.loc[garch_df["split"] == "test", "date"].values)
        s2 = _np.ones_like(e_test) * float(_np.var(train_series))
        fore = pd.DataFrame({"date": dates, "e": e_test, "sigma2_forecast": s2})
        return fore, {"refit_count": 0}

    monkeypatch.setattr(bmk, "run_rolling_garch_from_artifacts", fake_run_from_artifacts)
    fore_df2, metrics2 = bmk.run_vol_backtest(garch_df, var_alphas=[0.05])
    monkeypatch.setattr(C, "VOL_BACKTEST_FORECASTS_FILE", tmp_path / "vol_forecasts.csv", raising=False)
    monkeypatch.setattr(C, "VOL_BACKTEST_METRICS_FILE", tmp_path / "vol_metrics.json", raising=False)
    bmk.save_vol_backtest_outputs(fore_df2, metrics2)
    assert (tmp_path / "vol_forecasts.csv").exists()
    assert (tmp_path / "vol_metrics.json").exists()

    # 7) Visualization smoke (data_visualisation)
    _setup_fake_matplotlib(monkeypatch)
    import src.arima.data_visualisation.data_visualisation as dviz

    plots_dir = tmp_path / "plots"
    monkeypatch.setattr(dviz, "PLOTS_DIR", plots_dir, raising=False)
    monkeypatch.setattr(dviz, "plot_acf", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dviz, "plot_pacf", lambda *a, **k: None, raising=False)
    dviz.plot_weighted_series(
        data_file=str(returns_file),
        output_file=str(plots_dir / "weighted_log_returns_series.png"),
    )
    dviz.plot_acf_pacf(
        data_file=str(returns_file),
        output_file=str(plots_dir / "acf_pacf.png"),
        lags=5,
    )


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-q", "-x"])


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-q", "-x"])
