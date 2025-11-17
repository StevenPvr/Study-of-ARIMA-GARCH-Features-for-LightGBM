"""Tests for GARCH training module."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture(autouse=True)
def patch_garch_file_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatically patch all GARCH file paths to use temporary directory."""
    # Create a temporary subdirectory for GARCH files
    garch_temp_dir = tmp_path / "garch_outputs"
    garch_temp_dir.mkdir(parents=True, exist_ok=True)

    # Define temporary paths for all GARCH constants
    temp_paths = {
        "GARCH_DATASET_FILE": garch_temp_dir / "dataset_garch.csv",
        "GARCH_DIAGNOSTICS_FILE": garch_temp_dir / "diagnostics.json",
        "GARCH_NUMERICAL_TESTS_FILE": garch_temp_dir / "numerical_tests.json",
        "GARCH_ESTIMATION_FILE": garch_temp_dir / "estimation.json",
        "GARCH_OPTIMIZATION_RESULTS_FILE": garch_temp_dir / "hyperparameters.json",
        "GARCH_MODEL_FILE": garch_temp_dir / "model.joblib",
        "GARCH_MODEL_METADATA_FILE": garch_temp_dir / "model_metadata.json",
        "GARCH_RESIDUALS_OUTPUTS_FILE": garch_temp_dir / "residuals_outputs.csv",
        "GARCH_VARIANCE_OUTPUTS_FILE": garch_temp_dir / "variance_outputs.csv",
        "GARCH_LJUNGBOX_FILE": garch_temp_dir / "ljungbox.json",
        "GARCH_DISTRIBUTION_DIAGNOSTICS_FILE": garch_temp_dir / "distribution_diagnostics.json",
        "GARCH_FORECASTS_FILE": garch_temp_dir / "garch_forecasts.parquet",
        "GARCH_EVAL_METRICS_FILE": garch_temp_dir / "metrics.json",
        "GARCH_ROLLING_FORECASTS_FILE": garch_temp_dir / "forecasts.parquet",
        "GARCH_ROLLING_EVAL_FILE": garch_temp_dir / "rolling_metrics.json",
        "GARCH_ROLLING_VARIANCE_FILE": garch_temp_dir / "variance.parquet",
        "GARCH_ML_DATASET_FILE": garch_temp_dir / "ml_dataset.parquet",
    }

    # Patch in all modules that might use these constants
    modules_to_patch = [
        "src.garch.training_garch.training",
        "src.garch.training_garch.utils",
        "src.garch.training_garch.orchestration",
        "src.garch.training_garch.predictions_io",
        "src.garch.garch_params.main",
        "src.garch.garch_params.optimization.optuna",
        "src.garch.structure_garch.detection",
        "src.garch.structure_garch.utils",
        "src.garch.garch_eval.main",
        "src.constants",
        "src.path",
    ]

    for module_name in modules_to_patch:
        try:
            module = __import__(module_name, fromlist=[""])
            for const_name, temp_path in temp_paths.items():
                if hasattr(module, const_name):
                    monkeypatch.setattr(module, const_name, temp_path)
        except (ImportError, AttributeError):
            # Module might not exist or constant might not be there, skip
            continue


from src.garch.training_garch.forecaster import EGARCHForecaster, ForecastResult
from src.garch.training_garch.orchestration import (
    generate_full_sample_forecasts,
    load_garch_dataset,
    load_optimized_hyperparameters,
)
from src.garch.training_garch.predictions_io import (
    load_estimation_results,
    load_garch_forecasts,
    save_estimation_results,
    save_garch_forecasts,
    save_ml_dataset,
)
from src.garch.training_garch.training import (
    compute_diagnostics_with_filter,
    create_egarch_forecaster,
    create_training_summary,
    load_hyperparameters,
    save_model_and_metadata,
    train_egarch_from_dataset,
)
from src.garch.training_garch.utils import _compute_std_resid_diagnostics, _prepare_training_data


def _simulate_garch11(
    n: int, omega: float, alpha: float, beta: float, seed: int = 123
) -> np.ndarray:
    """Simulate GARCH(1,1) process for testing.

    Args:
        n: Number of observations.
        omega: Constant term.
        alpha: ARCH coefficient.
        beta: GARCH coefficient.
        seed: Random seed.

    Returns:
        Simulated GARCH residuals.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    e = np.empty(n)
    e[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * (e[t - 1] ** 2) + beta * sigma2[t - 1]
        e[t] = np.sqrt(sigma2[t]) * z[t]
    return e


def _create_test_dataframe(n: int = 50) -> pd.DataFrame:
    """Create a minimal test dataframe with required columns."""
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    split = np.where(np.arange(n) < n - 10, "train", "test")
    rng = np.random.default_rng(0)
    resid = rng.standard_normal(n) * 0.01
    return pd.DataFrame(
        {
            "date": dates,
            "split": split,
            "sarima_resid": resid,
        }
    )


def _create_test_optimization_results(tmp_path: Path) -> Path:
    """Create test optimization results file."""
    opt_file = tmp_path / "hyperparameters.json"
    opt_results = {
        "best_params": {
            "o": 1,
            "p": 1,
            "distribution": "normal",
            "window_type": "expanding",
            "refit_freq": 20,
            "window_size": None,
        }
    }
    opt_file.write_text(json.dumps(opt_results))
    return opt_file


def _patch_training_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, Path]:
    """Patch training module paths to use temporary directory.

    Returns:
        Tuple of (model_path, meta_path, variance_path, opt_path).
    """
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "model_metadata.json"
    variance_path = tmp_path / "variance_outputs.csv"
    opt_path = tmp_path / "hyperparameters.json"

    # Patch in the training module where they are used
    from src.garch.training_garch import training

    monkeypatch.setattr(training, "GARCH_MODEL_FILE", model_path)
    monkeypatch.setattr(training, "GARCH_MODEL_METADATA_FILE", meta_path)
    monkeypatch.setattr(training, "GARCH_VARIANCE_OUTPUTS_FILE", variance_path)

    # Also patch optimization results file used by various modules
    from src.garch.garch_params import main
    from src.garch.garch_params.optimization import optuna
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)
    monkeypatch.setattr(main, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path, raising=False)
    monkeypatch.setattr(optuna, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path, raising=False)

    return model_path, meta_path, variance_path, opt_path


# ============================================================================
# Tests for training.py
# ============================================================================


def test_load_hyperparameters_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading hyperparameters from file.

    Note: load_hyperparameters is actually load_optimized_hyperparameters from utils.py.
    The function uses GARCH_OPTIMIZATION_RESULTS_FILE which is imported from constants.py.
    We need to patch it in utils.py where it's used, not in constants.py.
    """
    opt_path = tmp_path / "hyperparameters.json"
    opt_results = {
        "best_params": {
            "o": 1,
            "p": 1,
            "distribution": "normal",
            "window_type": "expanding",
            "refit_freq": 20,
        }
    }
    opt_path.write_text(json.dumps(opt_results))

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    result = load_hyperparameters()
    assert result == opt_results["best_params"]


def test_load_hyperparameters_file_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading hyperparameters when file doesn't exist."""
    opt_path = Path("/nonexistent/hyperparameters.json")

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    with pytest.raises(FileNotFoundError, match="Optimization results not found"):
        load_hyperparameters()


def test_load_hyperparameters_invalid_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test loading hyperparameters with invalid format."""
    opt_path = tmp_path / "hyperparameters.json"
    opt_path.write_text(json.dumps({"invalid": "format"}))

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    with pytest.raises(ValueError, match="No 'best_params' in optimization results"):
        load_hyperparameters()


def test_create_egarch_forecaster() -> None:
    """Test creating EGARCH forecaster from hyperparameters."""
    hyperparams = {
        "o": 1,
        "p": 1,
        "distribution": "normal",
        "window_type": "expanding",
        "refit_freq": 20,
        "window_size": None,
    }
    forecaster = create_egarch_forecaster(hyperparams)
    assert forecaster.o == 1
    assert forecaster.p == 1
    assert forecaster.dist == "normal"
    assert forecaster.window_type == "expanding"
    assert forecaster.refit_frequency == 20


def test_create_egarch_forecaster_rolling_window() -> None:
    """Test creating EGARCH forecaster with rolling window."""
    hyperparams = {
        "o": 1,
        "p": 1,
        "distribution": "normal",
        "window_type": "rolling",
        "refit_freq": 20,
        "window_size": 100,
    }
    forecaster = create_egarch_forecaster(hyperparams)
    assert forecaster.window_type == "rolling"
    assert forecaster.window_size == 100


def test_compute_diagnostics_with_filter() -> None:
    """Test computing diagnostics with EGARCH variance filter."""
    resid_train = _simulate_garch11(200, 0.02, 0.05, 0.9, seed=42)
    final_params_dict = {
        "omega": 0.02,
        "alpha": 0.05,
        "gamma": 0.0,
        "beta": 0.9,
    }
    hyperparams = {
        "o": 1,
        "p": 1,
        "distribution": "normal",
        "window_type": "expanding",
        "refit_freq": 20,
    }
    forecaster = create_egarch_forecaster(hyperparams)

    sigma2_filtered, z_standardized, diagnostics = compute_diagnostics_with_filter(
        forecaster, resid_train, final_params_dict
    )

    assert sigma2_filtered.shape == resid_train.shape
    assert z_standardized.shape == resid_train.shape
    assert isinstance(diagnostics, dict)
    assert "n" in diagnostics
    assert "mean" in diagnostics
    assert "var" in diagnostics


def test_save_model_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving model and metadata."""
    model_path, meta_path, _, _ = _patch_training_paths(monkeypatch, tmp_path)

    hyperparams = {
        "o": 1,
        "p": 1,
        "distribution": "normal",
        "window_type": "expanding",
        "refit_freq": 20,
    }
    forecaster = create_egarch_forecaster(hyperparams)
    final_params = {
        "omega": 0.02,
        "alpha": 0.05,
        "gamma": 0.0,
        "beta": 0.9,
    }
    diagnostics = {"n": 100, "mean": 0.0, "var": 1.0, "std": 1.0}

    # Mock ForecastResult
    result = MagicMock()
    result.n_refits = 5
    result.convergence_rate = 0.8

    save_model_and_metadata(forecaster, final_params, hyperparams, result, diagnostics, n_train=100)

    assert model_path.exists()
    assert meta_path.exists()

    # Verify metadata
    with open(meta_path) as f:
        metadata = json.load(f)
    assert metadata["o"] == 1
    assert metadata["p"] == 1
    assert metadata["distribution"] == "normal"
    assert metadata["n_train"] == 100


def test_create_training_summary() -> None:
    """Test creating training summary."""
    hyperparams = {
        "o": 1,
        "p": 1,
        "distribution": "normal",
        "window_type": "expanding",
        "refit_freq": 20,
    }
    forecaster = create_egarch_forecaster(hyperparams)
    final_params = {
        "omega": 0.02,
        "alpha": 0.05,
        "gamma": 0.0,
        "beta": 0.9,
    }
    diagnostics = {"n": 100, "mean": 0.0, "var": 1.0, "std": 1.0}

    result = MagicMock()
    result.n_refits = 5
    result.convergence_rate = 0.8

    summary = create_training_summary(
        forecaster=forecaster,
        final_params=final_params,
        hyperparams=hyperparams,
        result=result,
        diagnostics=diagnostics,
        n_train=100,
    )

    assert summary["dist"] == "normal"
    assert summary["n_train"] == 100
    assert summary["n_refits"] == 5
    assert summary["convergence_rate"] == 0.8
    assert "model_file" in summary
    assert "metadata_file" in summary
    assert "outputs_file" in summary


def test_train_egarch_from_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test training EGARCH from dataset."""
    model_path, meta_path, var_path, opt_path = _patch_training_paths(monkeypatch, tmp_path)

    # Patch initial window size to smaller value for testing
    from src.garch.training_garch import training

    monkeypatch.setattr(training, "GARCH_INITIAL_WINDOW_SIZE_DEFAULT", 100)

    # Create optimization results
    _create_test_optimization_results(tmp_path)

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    # Create test dataset with sufficient data
    df = _create_test_dataframe(300)
    train_size = len(df[df["split"] == "train"])
    df.loc[df["split"] == "train", "sarima_resid"] = _simulate_garch11(
        train_size, 0.02, 0.05, 0.9, seed=42
    )

    # Mock estimation results file
    est_path = tmp_path / "estimation.json"
    est_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.garch_params.estimation.common import ConvergenceResult
    from src.garch.garch_params.refit.refit_manager import RefitManager


    # Mock refit_manager.perform_refit to use stable EGARCH parameters
    # This avoids MLE convergence to unstable parameters with small samples
    def mock_perform_refit(self, resid_hist, position):
        params = {
            "omega": 0.02,
            "alpha": 0.05,
            "gamma": -0.01,
            "beta": 0.9,
        }
        convergence = ConvergenceResult(
            converged=True, n_iterations=10, final_loglik=1000.0, message="Mocked convergence"
        )
        return params, convergence

    # Mock refit to avoid actual estimation
    with patch.object(RefitManager, "perform_refit", mock_perform_refit):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            result = train_egarch_from_dataset(df)

    # Distribution should match hyperparams (may vary based on estimation)
    assert result["dist"] in ["normal", "student", "skewt"]
    assert result["n_train"] > 0
    assert "params" in result
    assert "hyperparams" in result
    assert result["hyperparams"]["distribution"] == "normal"  # Check hyperparams not result dist
    assert "std_resid_diagnostics" in result
    assert "model_file" in result
    assert "metadata_file" in result
    assert "outputs_file" in result


def test_train_egarch_from_dataset_no_optimization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test training when optimization results are missing."""
    model_path, meta_path, variance_path, opt_path = _patch_training_paths(monkeypatch, tmp_path)

    # Ensure optimization file does NOT exist
    if opt_path.exists():
        opt_path.unlink()

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    # Create sufficient data so we don't get ValueError for insufficient data
    df = _create_test_dataframe(200)
    train_size = len(df[df["split"] == "train"])
    df.loc[df["split"] == "train", "sarima_resid"] = _simulate_garch11(
        train_size, 0.02, 0.05, 0.9, seed=42
    )

    # Patch initial window size to smaller value
    from src.garch.training_garch import training

    monkeypatch.setattr(training, "GARCH_INITIAL_WINDOW_SIZE_DEFAULT", 50)

    with pytest.raises(FileNotFoundError, match="Optimization results not found"):
        train_egarch_from_dataset(df)


# ============================================================================
# Tests for utils.py
# ============================================================================


def test_compute_std_resid_diagnostics() -> None:
    """Test computing standardized residuals diagnostics."""
    z = np.random.randn(100)
    diagnostics = _compute_std_resid_diagnostics(z)
    assert "n" in diagnostics
    assert "mean" in diagnostics
    assert "var" in diagnostics
    assert "std" in diagnostics
    assert "abs_gt_2" in diagnostics
    assert "abs_gt_3" in diagnostics
    assert diagnostics["n"] == 100


def test_compute_std_resid_diagnostics_empty() -> None:
    """Test computing diagnostics with empty array."""
    diagnostics = _compute_std_resid_diagnostics(np.array([]))
    assert diagnostics["n"] == 0
    assert np.isnan(diagnostics["mean"])


def test_prepare_training_data() -> None:
    """Test preparing training data."""
    df = _create_test_dataframe(100)
    df_train_sorted, resid_train, valid_mask_train, resid_train_full = _prepare_training_data(df)

    assert isinstance(df_train_sorted, pd.DataFrame)
    assert len(resid_train) > 0
    assert len(valid_mask_train) == len(resid_train_full)
    assert np.all(valid_mask_train >= 0)  # Boolean mask


def test_prepare_training_data_no_train() -> None:
    """Test preparing training data when no train split exists."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=10, freq="D"),
            "split": ["test"] * 10,
            "sarima_resid": np.random.randn(10),
        }
    )
    with pytest.raises(ValueError, match="No training data found"):
        _prepare_training_data(df)


# ============================================================================
# Tests for batch estimation (removed)
# ============================================================================


@pytest.mark.skip(reason="Batch estimation functions removed")
def test_estimate_all_distributions() -> None:
    """Test estimating all distributions."""
    pytest.skip("Batch estimation functions removed - this test is no longer relevant")


@pytest.mark.skip(reason="Batch estimation functions removed")
def test_run_batch_estimation_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running batch estimation and saving."""
    pytest.skip("Batch estimation functions removed")


@pytest.mark.skip(reason="Batch estimation functions removed")
def test_run_batch_estimation_and_save_no_train(tmp_path: Path) -> None:
    """Test batch estimation when no train data exists."""
    pytest.skip("Batch estimation functions removed")


# ============================================================================
# Tests for predictions_io.py
# ============================================================================


def test_save_estimation_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving estimation results."""
    est_path = tmp_path / "estimation.json"
    est_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_ESTIMATION_FILE", est_path)

    fits = {
        "normal": {
            "params": {"omega": 0.02, "alpha": 0.05, "gamma": 0.0, "beta": 0.9},
            "converged": True,
            "log_likelihood": 1000.0,
            "iterations": 50,
        }
    }

    save_estimation_results(fits, n_observations=200)

    assert est_path.exists()
    with open(est_path) as f:
        results = json.load(f)
    assert results["n_observations"] == 200
    assert "egarch_normal" in results


def test_load_estimation_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading estimation results."""
    est_path = tmp_path / "estimation.json"
    est_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_ESTIMATION_FILE", est_path)

    fits = {
        "normal": {
            "params": {"omega": 0.02, "alpha": 0.05, "gamma": 0.0, "beta": 0.9},
            "converged": True,
            "log_likelihood": 1000.0,
            "iterations": 50,
        }
    }
    save_estimation_results(fits, n_observations=200)

    results = load_estimation_results()
    assert results["n_observations"] == 200
    assert "egarch_normal" in results


def test_load_estimation_results_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading estimation results when file doesn't exist."""
    est_path = Path("/nonexistent/estimation.json")

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_ESTIMATION_FILE", est_path)

    with pytest.raises(FileNotFoundError, match="Estimation file not found"):
        load_estimation_results()


def test_save_garch_forecasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving GARCH forecasts."""
    forecast_path = tmp_path / "forecasts.csv"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_FORECASTS_FILE", forecast_path)

    df = _create_test_dataframe(100)
    df["garch_forecast_h1"] = np.random.rand(100) * 0.01
    df["garch_vol_h1"] = np.sqrt(df["garch_forecast_h1"])

    save_garch_forecasts(
        df,
        model_type="EGARCH(1,1)",
        distribution="normal",
        refit_frequency=20,
        window_type="expanding",
    )

    # Check both CSV and Parquet exist
    csv_path = forecast_path
    parquet_path = forecast_path.with_suffix(".parquet")
    assert csv_path.exists() or parquet_path.exists()
    if csv_path.exists():
        df_loaded = pd.read_csv(csv_path)
        assert "garch_forecast_h1" in df_loaded.columns


def test_load_garch_forecasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading GARCH forecasts."""
    forecast_path = tmp_path / "forecasts.csv"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_FORECASTS_FILE", forecast_path)

    df = _create_test_dataframe(100)
    df["garch_forecast_h1"] = np.random.rand(100) * 0.01
    save_garch_forecasts(df)

    df_loaded, metadata = load_garch_forecasts()
    assert len(df_loaded) == 100
    assert "garch_forecast_h1" in df_loaded.columns
    # Metadata may be None if not saved


def test_save_ml_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test saving ML dataset."""
    ml_path = tmp_path / "ml_dataset.parquet"
    ml_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import predictions_io

    monkeypatch.setattr(predictions_io, "GARCH_ML_DATASET_FILE", ml_path)

    df = _create_test_dataframe(100)
    df["feature1"] = np.random.rand(100)
    df["target"] = np.random.rand(100)

    save_ml_dataset(df)

    # Check that file was created (either parquet or csv)
    assert ml_path.with_suffix(".parquet").exists() or ml_path.with_suffix(".csv").exists()


# ============================================================================
# Tests for orchestration.py
# ============================================================================


def test_load_optimized_hyperparameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading optimized hyperparameters."""
    opt_path = tmp_path / "hyperparameters.json"
    opt_results = {
        "best_params": {
            "o": 1,
            "p": 1,
            "distribution": "normal",
            "window_type": "expanding",
            "refit_freq": 20,
        }
    }
    opt_path.write_text(json.dumps(opt_results))

    # Patch the constant in utils module where it's used
    from src.garch.training_garch import utils

    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)

    result = load_optimized_hyperparameters()
    assert result == opt_results["best_params"]


def test_load_garch_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading GARCH dataset."""
    dataset_path = tmp_path / "dataset_garch.csv"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import orchestration

    monkeypatch.setattr(orchestration, "GARCH_DATASET_FILE", dataset_path)

    df = _create_test_dataframe(100)
    df.to_csv(dataset_path, index=False)

    df_loaded = load_garch_dataset()
    assert len(df_loaded) == 100
    assert "sarima_resid" in df_loaded.columns
    assert "split" in df_loaded.columns


def test_load_garch_dataset_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading GARCH dataset when file doesn't exist."""
    dataset_path = Path("/nonexistent/dataset_garch.csv")

    from src.garch.training_garch import orchestration

    monkeypatch.setattr(orchestration, "GARCH_DATASET_FILE", dataset_path)

    with pytest.raises(FileNotFoundError, match="GARCH dataset not found"):
        load_garch_dataset()


def test_generate_full_sample_forecasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generating full sample forecasts."""
    # Setup paths
    dataset_path = tmp_path / "dataset_garch.csv"
    opt_path = tmp_path / "hyperparameters.json"
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.garch.training_garch import orchestration, predictions_io, utils

    monkeypatch.setattr(orchestration, "GARCH_DATASET_FILE", dataset_path)
    monkeypatch.setattr(utils, "GARCH_OPTIMIZATION_RESULTS_FILE", opt_path)
    monkeypatch.setattr(orchestration, "GARCH_EVALUATION_DIR", output_dir)
    monkeypatch.setattr(predictions_io, "GARCH_FORECASTS_FILE", output_dir / "forecasts.parquet")

    # Create dataset with sufficient data
    df = _create_test_dataframe(300)
    train_size = len(df[df["split"] == "train"])
    df.loc[df["split"] == "train", "sarima_resid"] = _simulate_garch11(
        train_size, 0.02, 0.05, 0.9, seed=42
    )
    test_size = len(df[df["split"] == "test"])
    df.loc[df["split"] == "test", "sarima_resid"] = _simulate_garch11(
        test_size, 0.02, 0.05, 0.9, seed=43
    )
    df.to_csv(dataset_path, index=False)

    # Create optimization results
    opt_results = {
        "best_params": {
            "o": 1,
            "p": 1,
            "distribution": "normal",
            "window_type": "expanding",
            "refit_freq": 50,
        }
    }
    opt_path.write_text(json.dumps(opt_results))

    # Generate forecasts
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        result_df = generate_full_sample_forecasts(
            use_optimized_params=True,
            output_dir=output_dir,
            initial_window_size=100,
        )

    assert len(result_df) == 300
    assert "garch_forecast_h1" in result_df.columns
    assert "garch_vol_h1" in result_df.columns
    assert "forecast_type" in result_df.columns
    assert "refit_occurred" in result_df.columns


# ============================================================================
# Tests for forecaster.py
# ============================================================================


def test_egarch_forecaster_init() -> None:
    """Test EGARCH forecaster initialization."""
    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=20,
        window_type="expanding",
        initial_window_size=50,
    )
    assert forecaster.o == 1
    assert forecaster.p == 1
    assert forecaster.dist == "normal"
    assert forecaster.refit_frequency == 20
    assert forecaster.window_type == "expanding"


def test_egarch_forecaster_init_invalid_order() -> None:
    """Test EGARCH forecaster with invalid order."""
    with pytest.raises(ValueError, match="ARCH order"):
        EGARCHForecaster(o=3, p=1, dist="normal", refit_frequency=10, window_type="expanding", initial_window_size=50)

    with pytest.raises(ValueError, match="GARCH order"):
        EGARCHForecaster(o=1, p=3, dist="normal", refit_frequency=10, window_type="expanding", initial_window_size=50)


def test_egarch_forecaster_init_invalid_window_type() -> None:
    """Test EGARCH forecaster with invalid window type."""
    with pytest.raises(ValueError, match="window_type"):
        EGARCHForecaster(o=1, p=1, dist="normal", window_type="invalid", refit_frequency=10, initial_window_size=50)


def test_egarch_forecaster_init_rolling_window_no_size() -> None:
    """Test EGARCH forecaster with rolling window but no size."""
    with pytest.raises(ValueError, match="window_size required"):
        EGARCHForecaster(o=1, p=1, dist="normal", window_type="rolling", window_size=None, refit_frequency=10, initial_window_size=50)


def test_egarch_forecaster_forecast_expanding() -> None:
    """Test EGARCH forecaster expanding window forecasts."""
    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=100,
        window_type="expanding",
        initial_window_size=100,
    )

    resid = _simulate_garch11(250, 0.02, 0.05, 0.9, seed=42)
    dates = pd.date_range("2020-01-01", periods=len(resid), freq="D")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        result = forecaster.forecast_expanding(resid, dates=dates)

    assert isinstance(result, ForecastResult)
    assert len(result.forecasts) == len(resid)
    assert len(result.volatility) == len(resid)
    # params_history should have entries for forecasts after initial_window_size
    assert len(result.params_history) == len(resid) - forecaster.initial_window_size
    assert result.n_refits >= 0
    assert 0.0 <= result.convergence_rate <= 1.0


def test_egarch_forecaster_forecast_expanding_insufficient_data() -> None:
    """Test EGARCH forecaster with insufficient data."""
    forecaster = EGARCHForecaster(
        o=1,
        p=1,
        dist="normal",
        refit_frequency=20,
        window_type="expanding",
        initial_window_size=50,
    )

    resid = np.random.randn(10)  # Too small

    with pytest.raises(ValueError, match="Insufficient residuals"):
        forecaster.forecast_expanding(resid)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
