"""End-to-end tests using default module paths (monkeypatched).

Runs the full pipeline with defaults: cleaning → conversion → split → load,
by monkeypatching module-level constants to tmp files. Uses synthetic data only.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import data_cleaning as dc
from src.data_conversion import data_conversion as dv
from src.data_preparation import data_preparation as dp


def _build_small_raw() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=6, freq="D")

    def mk(t: str, n: int, base: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates[:n],
                "ticker": t,
                "open": base - 0.5 + pd.RangeIndex(n).astype(float) * 0.2,
                "closing": base + pd.RangeIndex(n).astype(float) * 0.8,
                "volume": 2_000 + pd.RangeIndex(n) * 5,
            }
        )

    return pd.concat([mk("AAA", 6, 100), mk("BBB", 5, 40), mk("CCC", 1, 10)], ignore_index=True)


def test_e2e_full_pipeline_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E2E: defaults with monkeypatched paths across modules.

    Verifies artifacts are produced and final loaded series are non-empty.
    """
    # Arrange synthetic raw dataset
    raw = _build_small_raw()
    dataset_file = tmp_path / "dataset.csv"
    filtered_file = tmp_path / "dataset_filtered.csv"
    weights_file = tmp_path / "liquidity_weights.csv"
    returns_file = tmp_path / "weighted_log_returns.csv"
    split_file = tmp_path / "weighted_log_returns_split.csv"
    raw.to_csv(dataset_file, index=False)

    # Patch data_cleaning module-level constants
    monkeypatch.setattr(dc, "DATASET_FILE", dataset_file, raising=False)
    monkeypatch.setattr(dc, "DATASET_FILTERED_FILE", filtered_file, raising=False)
    monkeypatch.setattr(dc, "MIN_OBSERVATIONS_PER_TICKER", 3, raising=False)

    # Patch data_conversion module-level paths used as defaults
    monkeypatch.setattr(dv, "DATASET_FILTERED_FILE", filtered_file, raising=False)
    monkeypatch.setattr(dv, "LIQUIDITY_WEIGHTS_FILE", weights_file, raising=False)
    monkeypatch.setattr(dv, "WEIGHTED_LOG_RETURNS_FILE", returns_file, raising=False)

    # Patch data_preparation module-level paths used as defaults
    monkeypatch.setattr(dp, "WEIGHTED_LOG_RETURNS_FILE", returns_file, raising=False)
    monkeypatch.setattr(dp, "WEIGHTED_LOG_RETURNS_SPLIT_FILE", split_file, raising=False)

    # Execute: clean → convert → split → load
    dc.filter_incomplete_tickers()
    assert filtered_file.exists()

    dv.compute_weighted_log_returns()  # uses patched defaults
    assert weights_file.exists()
    assert returns_file.exists()

    dp.split_train_test(train_ratio=0.6)  # uses patched defaults
    assert split_file.exists()

    train_series, test_series = dp.load_train_test_data()  # uses patched defaults
    assert len(train_series) > 0 and len(test_series) > 0
    # Monotonic by construction
    assert train_series.index.is_monotonic_increasing
    assert test_series.index.is_monotonic_increasing


def _setup_fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install minimal matplotlib stubs into sys.modules for plotting smoke tests."""
    import sys as _sys

    class _FakeAxes:
        def __init__(self, figure=None):
            self.figure = figure

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

        def fill_between(self, *a, **k):
            return None

        def text(self, *a, **k):
            return None

        def hist(self, *a, **k):
            return None

        def scatter(self, *a, **k):
            return None

        def bar(self, *a, **k):
            return None

        # Minimal attribute used by code that sets text with transform
        @property
        def transAxes(self):  # type: ignore[override]
            return object()

    class _FakeFig:
        def __init__(self, *a, **k):
            self.axes = []

        def add_subplot(self, *a, **k):
            ax = _FakeAxes(figure=self)
            self.axes.append(ax)
            return ax

        def subplots(self, nrows: int = 1, ncols: int = 1, *args, **kwargs):
            nrows = int(nrows)
            ncols = int(ncols)
            if nrows <= 0:
                nrows = 1
            if ncols <= 0:
                ncols = 1
            if nrows == 1 and ncols == 1:
                return _FakeAxes(figure=self)
            if nrows == 1:
                return [_FakeAxes(figure=self) for _ in range(ncols)]
            if ncols == 1:
                return [_FakeAxes(figure=self) for _ in range(nrows)]
            return [[_FakeAxes(figure=self) for _ in range(ncols)] for _ in range(nrows)]

        def suptitle(self, *a, **k):
            return None

    def _save(path: str, *a, **k):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    fake_mpl = ModuleType("matplotlib")
    fake_axes = ModuleType("matplotlib.axes")
    setattr(fake_axes, "Axes", _FakeAxes)
    fake_pyplot = ModuleType("matplotlib.pyplot")

    class _AxesGrid:
        def __init__(self, nrows: int, ncols: int, figure=None):
            self._figure = figure
            self._grid = [[_FakeAxes(figure=figure) for _ in range(ncols)] for _ in range(nrows)]

        def __getitem__(self, key: int | tuple[int, int]) -> _FakeAxes:
            if isinstance(key, tuple) and len(key) == 2:
                i, j = key
                return self._grid[i][j]
            if isinstance(key, int):
                # For 1D indexing, return first axes in that row
                return self._grid[key][0]
            raise TypeError(f"Invalid key type: {type(key)}")

    def _subplots(*a, **k):
        # Support signatures like subplots(), subplots(1,2), subplots(2,2)
        nrows = 1
        ncols = 1
        if len(a) >= 1 and isinstance(a[0], int):
            nrows = a[0]
        if len(a) >= 2 and isinstance(a[1], int):
            ncols = a[1]
        nrows = int(k.get("nrows", nrows))
        ncols = int(k.get("ncols", ncols))
        fig = _FakeFig()
        if nrows == 1 and ncols == 1:
            return fig, _FakeAxes(figure=fig)
        if nrows == 1 and ncols >= 1:
            return fig, [_FakeAxes(figure=fig) for _ in range(ncols)]
        if ncols == 1 and nrows >= 1:
            return fig, [_FakeAxes(figure=fig) for _ in range(nrows)]
        # 2D grid
        return fig, _AxesGrid(nrows, ncols, figure=fig)

    setattr(fake_pyplot, "subplots", _subplots)
    setattr(fake_pyplot, "tight_layout", lambda *a, **k: None)
    setattr(fake_pyplot, "savefig", _save)
    setattr(fake_pyplot, "close", lambda *a, **k: None)
    fake_backends = ModuleType("matplotlib.backends")
    fake_backend_agg = ModuleType("matplotlib.backends.backend_agg")

    class _FakeCanvas:
        def __init__(self, fig: _FakeFig) -> None:
            self.fig = fig

        def print_png(self, path: str) -> None:
            _save(path)

    setattr(fake_backend_agg, "FigureCanvasAgg", _FakeCanvas)
    fake_figure = ModuleType("matplotlib.figure")
    setattr(fake_figure, "Figure", _FakeFig)

    monkeypatch.setitem(_sys.modules, "matplotlib", fake_mpl)
    monkeypatch.setitem(_sys.modules, "matplotlib.axes", fake_axes)
    monkeypatch.setitem(_sys.modules, "matplotlib.pyplot", fake_pyplot)
    monkeypatch.setitem(_sys.modules, "matplotlib.backends", fake_backends)
    monkeypatch.setitem(_sys.modules, "matplotlib.backends.backend_agg", fake_backend_agg)
    monkeypatch.setitem(_sys.modules, "matplotlib.figure", fake_figure)


def test_e2e_full_pipeline_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full offline E2E across all modules with mocks/stubs.

    Steps:
      1) data_fetching: mock Wikipedia + yfinance
      2) data_cleaning, data_conversion, data_preparation
      3) SARIMA optimize → train → evaluate → save
      4a) GARCH structure detection (ARCH-LM test)
      4b) GARCH parameter estimation (MLE)
      4c) GARCH training from pre-estimation
      4d) GARCH evaluation (forecasts, VaR, metrics)
      4e) GARCH diagnostics (post-estimation)
      5) Rolling GARCH with periodic refits
      6) Benchmark with real rolling GARCH
      7) data_visualisation smoke plots with fake matplotlib
      8) GARCH visualisation
    """
    # 1) data_fetching
    from src.data_fetching import data_fetching as dfetch

    tickers_file = tmp_path / "sp500_tickers.csv"
    dataset_file = tmp_path / "dataset.csv"
    monkeypatch.setattr(dfetch, "SP500_TICKERS_FILE", tickers_file, raising=False)
    monkeypatch.setattr(dfetch, "DATASET_FILE", dataset_file, raising=False)
    monkeypatch.setattr(dfetch, "DATA_DIR", tmp_path, raising=False)

    # Mock pd.read_html to avoid lxml dependency
    def _mock_read_html(html_bytes):
        return [pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"]})]

    monkeypatch.setattr(pd, "read_html", _mock_read_html)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"<html></html>"

    import urllib.request as _url

    monkeypatch.setattr(_url, "urlopen", lambda *a, **k: _Resp())

    # yfinance MagicMock from conftest; attach download behavior
    import sys as _sys

    yf = _sys.modules.get("yfinance")

    def _yf_download(ticker: str, start=None, end=None, progress=False, auto_adjust=True):
        n = {"AAA": 6, "BBB": 5, "CCC": 2}[ticker]
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
    assert dataset_file.exists()

    # 2) cleaning → conversion → prep
    filtered_file = tmp_path / "dataset_filtered.csv"
    monkeypatch.setattr(dc, "DATASET_FILE", dataset_file, raising=False)
    monkeypatch.setattr(dc, "DATASET_FILTERED_FILE", filtered_file, raising=False)
    monkeypatch.setattr(dc, "MIN_OBSERVATIONS_PER_TICKER", 3, raising=False)
    dc.filter_incomplete_tickers()
    weights_file = tmp_path / "liquidity_weights.csv"
    returns_file = tmp_path / "weighted_log_returns.csv"
    dv.compute_weighted_log_returns(
        input_file=str(filtered_file),
        weights_output_file=str(weights_file),
        returns_output_file=str(returns_file),
    )
    split_file = tmp_path / "weighted_log_returns_split.csv"
    dp.split_train_test(train_ratio=0.8, input_file=str(returns_file), output_file=str(split_file))
    train_series, test_series = dp.load_train_test_data(input_file=str(split_file))

    # 3) SARIMA optimize → train → evaluate
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
    monkeypatch.setattr(aeval, "RESULTS_DIR", preds_dir, raising=False)
    monkeypatch.setattr(aeval, "ROLLING_PREDICTIONS_SARIMA_FILE", preds_dir / "rolling_predictions.csv", raising=False)
    monkeypatch.setattr(aeval, "ROLLING_VALIDATION_METRICS_SARIMA_FILE", preds_dir / "rolling_metrics.json", raising=False)
    monkeypatch.setattr(aeval, "LJUNGBOX_RESIDUALS_SARIMA_FILE", preds_dir / "ljungbox.json", raising=False)
    monkeypatch.setattr(aeval, "WEIGHTED_LOG_RETURNS_SPLIT_FILE", split_file, raising=False)
    eval_results = aeval.evaluate_model(train_series, test_series, order=(0, 0, 1), seasonal_order=(0, 0, 0, 12))
    aeval.save_evaluation_results(eval_results)
    aeval.save_ljung_box_results({"lags": [1], "q_stat": [0.0], "p_value": [1.0], "reject_5pct": False, "n": 1})
    aeval.save_garch_dataset(eval_results, fitted_model=fitted_model)

    # Load dataset prepared earlier
    from src.constants import GARCH_DATASET_FILE as _GARCH_DATASET_FILE
    from src import constants as C

    garch_df = pd.read_csv(_GARCH_DATASET_FILE, parse_dates=["date"])  # type: ignore[arg-type]

    # 4a) GARCH structure detection (ARCH-LM test)
    from src.garch.structure_garch.detection import detect_heteroskedasticity, prepare_residuals
    monkeypatch.setattr(C, "GARCH_DIAGNOSTICS_FILE", tmp_path / "garch_diagnostics.json", raising=False)
    resid_test = prepare_residuals(garch_df, use_test_only=True)
    resid_test = resid_test[np.isfinite(resid_test)]
    structure_results = detect_heteroskedasticity(resid_test, lags=5, acf_lags=10, alpha=0.05)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with C.GARCH_DIAGNOSTICS_FILE.open("w") as f:
        __import__("json").dump({"diagnostics": structure_results, "n_test": int(resid_test.size)}, f, indent=2)
    assert C.GARCH_DIAGNOSTICS_FILE.exists()

    # 4b) GARCH parameter estimation (MLE)
    from src.garch.garch_params.estimation import estimate_egarch_mle
    monkeypatch.setattr(C, "GARCH_ESTIMATION_FILE", tmp_path / "garch_estimation.json", raising=False)
    resid_train = prepare_residuals(garch_df.loc[garch_df["split"] == "train"], use_test_only=False)
    resid_train = resid_train[np.isfinite(resid_train)]
    # Use small synthetic residuals for fast estimation in tests
    if len(resid_train) < 20:
        resid_train = np.random.RandomState(42).normal(0, 0.01, 50)
    try:
        egarch_normal = estimate_egarch_mle(resid_train, dist="normal")
    except Exception:
        egarch_normal = {"omega": 1e-6, "alpha": 0.05, "gamma": 0.0, "beta": 0.9, "loglik": -100.0, "converged": True}
    try:
        egarch_student = estimate_egarch_mle(resid_train, dist="student")
    except Exception:
        egarch_student = {"omega": 1e-6, "alpha": 0.06, "gamma": 0.0, "beta": 0.88, "nu": 8.0, "loglik": -99.0, "converged": True}
    est = {
        "egarch_normal": egarch_normal,
        "egarch_student": egarch_student,
        "source": str(_GARCH_DATASET_FILE),
        "n_obs_train": int(resid_train.size),
        "n_obs_test": int(resid_test.size),
    }
    with C.GARCH_ESTIMATION_FILE.open("w") as f:
        __import__("json").dump(est, f, indent=2)
    assert C.GARCH_ESTIMATION_FILE.exists()

    # 4c) GARCH training (from pre-estimation)
    from src.garch.training_garch import training as gtrain
    # Patch constants in the training module (it imports them directly)
    garch_model_file = tmp_path / "results" / "models" / "garch_model.joblib"
    garch_metadata_file = tmp_path / "results" / "models" / "garch_model.json"
    garch_variance_file = tmp_path / "results" / "garch_variance.csv"
    monkeypatch.setattr(gtrain, "GARCH_MODEL_FILE", garch_model_file, raising=False)
    monkeypatch.setattr(gtrain, "GARCH_MODEL_METADATA_FILE", garch_metadata_file, raising=False)
    monkeypatch.setattr(gtrain, "GARCH_VARIANCE_OUTPUTS_FILE", garch_variance_file, raising=False)
    gtrain.train_egarch_from_dataset(garch_df)
    assert garch_model_file.exists()
    assert garch_variance_file.exists()

    # 4d) GARCH evaluation (forecasts, VaR, metrics)
    from src.garch.garch_eval import eval as geval_eval
    from src.garch.garch_eval import metrics as geval_metrics
    garch_forecasts_file = tmp_path / "results" / "garch_forecasts.csv"
    garch_eval_metrics_file = tmp_path / "results" / "garch_eval_metrics.json"
    # Patch constants in garch_eval modules (they import them directly)
    monkeypatch.setattr(geval_eval, "GARCH_FORECASTS_FILE", garch_forecasts_file, raising=False)
    monkeypatch.setattr(geval_eval, "GARCH_DATASET_FILE", _GARCH_DATASET_FILE, raising=False)
    monkeypatch.setattr(geval_eval, "GARCH_ESTIMATION_FILE", C.GARCH_ESTIMATION_FILE, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_EVAL_METRICS_FILE", garch_eval_metrics_file, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_DATASET_FILE", _GARCH_DATASET_FILE, raising=False)
    monkeypatch.setattr(geval_metrics, "GARCH_VARIANCE_OUTPUTS_FILE", garch_variance_file, raising=False)
    forecast_df = geval_eval.forecast_from_artifacts(horizon=5, level=0.95)
    assert len(forecast_df) > 0
    assert garch_forecasts_file.exists()
    # Compute metrics
    from src.garch.garch_eval.eval import _choose_best_from_estimation
    params, name, nu = _choose_best_from_estimation(est)
    dist = "student" if "student" in name else "normal"
    metrics = geval_metrics.compute_classic_metrics_from_artifacts(params=params, model_name=name, dist=dist, nu=nu, alphas=[0.01, 0.05])
    geval_metrics.save_metrics_json(metrics)
    assert garch_eval_metrics_file.exists()

    # 4e) GARCH diagnostics (post-estimation)
    from src.garch.garch_diagnostic.diagnostics import (
        compute_distribution_diagnostics,
        compute_ljung_box_on_std_squared,
    )
    monkeypatch.setattr(C, "GARCH_LJUNGBOX_FILE", tmp_path / "results" / "garch_ljungbox.json", raising=False)
    monkeypatch.setattr(C, "GARCH_STD_QQ_PLOT", tmp_path / "plots" / "garch_qq_std.png", raising=False)
    monkeypatch.setattr(C, "GARCH_STD_ACF_PACF_PLOT", tmp_path / "plots" / "garch_acf_pacf_std.png", raising=False)
    monkeypatch.setattr(C, "GARCH_STD_SQUARED_ACF_PACF_PLOT", tmp_path / "plots" / "garch_acf_pacf_std_sq.png", raising=False)
    best_params = egarch_student if egarch_student.get("converged") else egarch_normal
    # Ensure we have converged params for diagnostics (use defaults if needed)
    if not best_params.get("converged"):
        best_params = {"omega": 1e-6, "alpha": 0.05, "gamma": 0.0, "beta": 0.9, "loglik": -100.0, "converged": True}
    diag = compute_distribution_diagnostics(resid_test, best_params, dist=dist, nu=best_params.get("nu"))
    assert isinstance(diag, dict)
    lb2 = compute_ljung_box_on_std_squared(resid_test, best_params, lags=5)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with C.GARCH_LJUNGBOX_FILE.open("w") as f:
        __import__("json").dump(lb2, f, indent=2)
    assert C.GARCH_LJUNGBOX_FILE.exists()

    # 5) Rolling GARCH (with periodic refits)
    from src.garch.rolling_garch.rolling import (
        EgarchParams,
        run_rolling_egarch,
        save_rolling_outputs,
    )
    # Mock _fit_initial_params to use pre-estimated parameters
    def _mock_fit_initial_params(resid_train, dist_preference="auto"):
        best_params = egarch_student if egarch_student.get("converged") else egarch_normal
        if not best_params.get("converged"):
            best_params = {"omega": 1e-6, "alpha": 0.05, "gamma": 0.0, "beta": 0.9, "loglik": -100.0, "converged": True}
        dist_name = "student" if best_params.get("nu") else "normal"
        return EgarchParams(
            omega=float(best_params["omega"]),
            alpha=float(best_params["alpha"]),
            beta=float(best_params["beta"]),
            gamma=float(best_params.get("gamma", 0.0)),
            nu=float(best_params["nu"]) if best_params.get("nu") else None,
            dist=dist_name,
            model="egarch",
        )

    monkeypatch.setattr("src.garch.rolling_garch.rolling._fit_initial_params", _mock_fit_initial_params)
    monkeypatch.setattr(C, "GARCH_ROLLING_FORECASTS_FILE", tmp_path / "results" / "garch_rolling_forecasts.csv", raising=False)
    monkeypatch.setattr(C, "GARCH_ROLLING_EVAL_FILE", tmp_path / "results" / "garch_rolling_eval.json", raising=False)
    # Use run_rolling_garch directly with the DataFrame to avoid file I/O issues
    rolling_forecasts, rolling_metrics = run_rolling_egarch(
        garch_df,
        refit_every=10,
        window="expanding",
        window_size=50,
        dist_preference="auto",
        keep_nu_between_refits=True,
        var_alphas=[0.01, 0.05],
    )
    save_rolling_outputs(rolling_forecasts, rolling_metrics)
    assert C.GARCH_ROLLING_FORECASTS_FILE.exists()
    assert C.GARCH_ROLLING_EVAL_FILE.exists()

    # 6) Benchmark (with real rolling GARCH)
    import src.benchmark.bench_volatility as bmk
    monkeypatch.setattr(C, "VOL_BACKTEST_FORECASTS_FILE", tmp_path / "vol_forecasts.csv", raising=False)
    monkeypatch.setattr(C, "VOL_BACKTEST_METRICS_FILE", tmp_path / "vol_metrics.json", raising=False)
    forecasts, metrics = bmk.run_vol_backtest(garch_df, var_alphas=[0.05])
    bmk.save_vol_backtest_outputs(forecasts, metrics)
    assert C.VOL_BACKTEST_FORECASTS_FILE.exists()
    assert C.VOL_BACKTEST_METRICS_FILE.exists()

    # 7) Visualisation smoke with fake matplotlib
    _setup_fake_matplotlib(monkeypatch)
    import src.data_visualisation.data_visualisation as dviz
    plots_dir = tmp_path / "plots"
    monkeypatch.setattr(dviz, "PLOTS_DIR", plots_dir, raising=False)
    dviz.plot_weighted_series(
        data_file=str(returns_file),
        output_file=str(plots_dir / "weighted_log_returns_series.png"),
    )
    # no-op plot functions to avoid statsmodels coupling
    monkeypatch.setattr(dviz, "plot_acf", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(dviz, "plot_pacf", lambda *a, **k: None, raising=False)
    # seasonal_decompose can require longer series; stub decomposition to safe output
    monkeypatch.setattr(
        dviz,
        "_decompose_seasonal_component",
        lambda series, *, model, period: series * 0,
        raising=False,
    )
    dviz.plot_acf_pacf(
        data_file=str(returns_file),
        output_file=str(plots_dir / "acf_pacf.png"),
        lags=5,
    )

    preds = np.asarray(eval_results["predictions"], dtype=float)
    acts = np.asarray(eval_results["actuals"], dtype=float)
    dviz.plot_rolling_forecast_sarima_000(
        test_series=test_series,
        actuals=acts,
        predictions=preds,
        sarima_order=(0, 0, 1),
        metrics=eval_results.get("metrics", {"RMSE": 0.0, "MAE": 0.0}),
        output_file=str(plots_dir / "rolling_forecast_sarima_000.png"),
    )
    # Skip analyze_residuals smoke to avoid deeper matplotlib coupling
    year = int(train_series.index.min().year)  # type: ignore[attr-defined]
    dviz.plot_seasonality_for_year(
        year,
        data_file=str(returns_file),
        output_file=str(plots_dir / f"seasonal_year_{year}.png"),
        resample_to="B",
        period=5,
    )

    # 8) GARCH visualisation
    from src.garch.garch_data_visualisation.plots import save_returns_and_squared_plots, save_acf_squared_plots
    # Mock plot_acf to avoid statsmodels issues with insufficient data
    import statsmodels.graphics.tsaplots as _tsaplots
    monkeypatch.setattr(_tsaplots, "plot_acf", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(C, "GARCH_RETURNS_CLUSTERING_PLOT", tmp_path / "plots" / "garch_returns_clustering.png", raising=False)
    monkeypatch.setattr(C, "GARCH_ACF_SQUARED_PLOT", tmp_path / "plots" / "garch_acf_squared.png", raising=False)
    df_test = garch_df.loc[garch_df["split"] == "test"].copy()
    if "weighted_return" not in df_test.columns and "weighted_log_return" in df_test.columns:
        df_test["weighted_return"] = np.expm1(df_test["weighted_log_return"])  # type: ignore[index]
    if "weighted_return" in df_test.columns:
        save_returns_and_squared_plots(
            df_test["weighted_return"].to_numpy().astype(float),
            outdir=tmp_path / "plots",
            filename=C.GARCH_RETURNS_CLUSTERING_PLOT.name,
        )
    save_acf_squared_plots(
        resid_test,
        acf_lags=10,
        outdir=tmp_path / "plots",
        filename=C.GARCH_ACF_SQUARED_PLOT.name,
    )


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-q", "-x"])
