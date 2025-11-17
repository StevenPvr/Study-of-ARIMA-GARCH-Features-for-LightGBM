"""Unit tests for LightGBM optimization module."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Filter warnings for cleaner test output
# Use simplefilter to catch warnings from all modules including subprocesses
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="joblib.externals.loky.backend.context"
)


import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.constants import LIGHTGBM_OPTIMIZATION_N_SPLITS
from src.lightgbm.optimisation import (
    load_dataset,
    optimize_lightgbm,
    run_optimization,
    save_optimization_results,
)


@pytest.fixture
def mock_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Create mock dataset for testing.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "log_volatility_t": np.random.randn(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randn(n_samples), name="log_volatility")
    return X, y


@pytest.fixture
def mock_dataset_file(tmp_path: Path) -> Path:
    """Create mock dataset CSV file for testing.

    Args:
        tmp_path: Pytest tmp_path fixture.

    Returns:
        Path to mock dataset CSV file.
    """
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_samples),
            "log_volatility": np.random.randn(n_samples),
            "log_volatility_t": np.random.randn(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "ticker": ["AAA"] * n_samples,
            "ticker_id": [0] * n_samples,
            "split": ["train"] * n_samples,
        }
    )
    file_path = tmp_path / "test_dataset.csv"
    df.to_csv(file_path, index=False)
    return file_path


def _assert_dataset_basic(X: pd.DataFrame, y: pd.Series, expected_len: int) -> None:
    """Assert basic dataset structure."""
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == expected_len
    assert len(y) == expected_len


def _assert_dataset_columns(X: pd.DataFrame, y: pd.Series, expected_columns: set[str]) -> None:
    """Assert dataset columns."""
    assert y.name == "log_volatility"
    assert "date" not in X.columns
    assert "log_volatility" not in X.columns
    assert set(X.columns) == expected_columns


def test_load_dataset_success(mock_dataset_file: Path) -> None:
    """Test loading dataset from CSV file."""
    X, y = load_dataset(mock_dataset_file, sample_fraction=1.0)

    _assert_dataset_basic(X, y, expected_len=100)
    expected_columns = {"log_volatility_t", "feature1", "feature2", "feature3", "ticker_id"}
    _assert_dataset_columns(X, y, expected_columns)


def test_load_dataset_file_not_found() -> None:
    """Test loading dataset raises error when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(Path("nonexistent.csv"), sample_fraction=1.0)


def test_load_dataset_empty_file(tmp_path: Path) -> None:
    """Test loading empty dataset raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_file, index=False)

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_dataset(empty_file, sample_fraction=1.0)


def test_load_dataset_missing_target(tmp_path: Path) -> None:
    """Test loading dataset without target column raises ValueError."""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    file_path = tmp_path / "no_target.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="must contain 'log_volatility' column"):
        load_dataset(file_path, sample_fraction=1.0)


def _create_split_dataset(tmp_path: Path) -> Path:
    """Create dataset with train/test split."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10),
            "log_volatility": np.random.randn(10),
            "log_volatility_t": np.random.randn(10),
            "feature1": np.random.randn(10),
            "ticker": ["AAA"] * 10,
            "ticker_id": [0] * 10,
            "split": ["train"] * 6 + ["test"] * 4,
        }
    )
    file_path = tmp_path / "with_split.csv"
    df.to_csv(file_path, index=False)
    return file_path


def _assert_train_split_lengths(X: pd.DataFrame, y: pd.Series) -> None:
    """Assert train split lengths."""
    assert len(X) == 6  # Only train split
    assert len(y) == 6


def _assert_train_split_columns(X: pd.DataFrame) -> None:
    """Assert train split columns."""
    assert "split" not in X.columns
    assert "log_volatility_t" in X.columns
    assert "ticker_id" in X.columns


def _assert_train_split_filtered(X: pd.DataFrame, y: pd.Series) -> None:
    """Assert train split filtering."""
    _assert_train_split_lengths(X, y)
    _assert_train_split_columns(X)


def test_load_dataset_filters_train_split(tmp_path: Path) -> None:
    """Test loading dataset filters to train split only."""
    file_path = _create_split_dataset(tmp_path)
    X, y = load_dataset(file_path, sample_fraction=1.0)
    _assert_train_split_filtered(X, y)


def test_load_dataset_requires_ticker_id(tmp_path: Path) -> None:
    """Datasets with ticker but without ticker_id should raise informative error."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=6),
            "log_volatility": np.random.randn(6),
            "feature1": np.random.randn(6),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "split": ["train"] * 6,
        }
    )
    file_path = tmp_path / "with_ticker.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="ticker_id"):
        load_dataset(file_path, sample_fraction=1.0)


def test_load_dataset_preserves_existing_ticker_id(tmp_path: Path) -> None:
    """Ensure ticker_id feature is retained and ticker removed from features."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-02-01", periods=6),
            "log_volatility": np.random.randn(6),
            "feature1": np.random.randn(6),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ticker_id": [0, 0, 1, 1, 2, 2],
            "split": ["train"] * 6,
        }
    )
    file_path = tmp_path / "with_ticker_id.csv"
    df.to_csv(file_path, index=False)

    X, _ = load_dataset(file_path, sample_fraction=1.0)

    assert "ticker" not in X.columns
    assert "ticker_id" in X.columns
    assert X["ticker_id"].nunique() == 3
    assert X["ticker_id"].dtype.kind in {"i", "u"}


# Removed test_walk_forward_cv_score and test_walk_forward_cv_score_deterministic
# These tests were specific to Random Forest and are no longer applicable with LightGBM


def _assert_results_types(results: dict[str, Any], best_params: dict[str, Any]) -> None:
    """Assert types of optimization results."""
    assert isinstance(results, dict)
    assert isinstance(best_params, dict)


def _assert_results_required_keys(results: dict[str, Any]) -> None:
    """Assert required keys in results."""
    required_keys = {
        "best_params",
        "best_rmse_cv",
        "n_trials",
        "study_name",
        "fold_rmses",
        "best_fold",
    }
    assert required_keys.issubset(set(results.keys()))


def _assert_results_keys(results: dict[str, Any]) -> None:
    """Assert required keys in results."""
    _assert_results_required_keys(results)


def _assert_results_structure(results: dict[str, Any], best_params: dict[str, Any]) -> None:
    """Assert basic structure of optimization results.

    Args:
        results: Results dictionary.
        best_params: Best parameters dictionary.
    """
    _assert_results_types(results, best_params)
    _assert_results_keys(results)


def _assert_results_trials_study(
    results: dict[str, Any], expected_trials: int, expected_study: str
) -> None:
    """Assert trials and study name."""
    assert results["n_trials"] == expected_trials
    assert results["study_name"] == expected_study


def _assert_results_rmse_folds(results: dict[str, Any]) -> None:
    """Assert RMSE and folds."""
    assert np.isfinite(results["best_rmse_cv"])
    assert isinstance(results["fold_rmses"], list)
    assert len(results["fold_rmses"]) == LIGHTGBM_OPTIMIZATION_N_SPLITS
    assert all(np.isfinite(rmse) for rmse in results["fold_rmses"])


def _assert_results_values(
    results: dict[str, Any], expected_trials: int, expected_study: str
) -> None:
    """Assert values in optimization results.

    Args:
        results: Results dictionary.
        expected_trials: Expected number of trials.
        expected_study: Expected study name.
    """
    _assert_results_trials_study(results, expected_trials, expected_study)
    _assert_results_rmse_folds(results)


def _assert_best_params_structure(best_params: dict[str, Any]) -> None:
    """Assert structure of best parameters.

    Args:
        best_params: Best parameters dictionary.
    """
    assert "num_leaves" in best_params or "learning_rate" in best_params
    assert len(best_params) > 0


def test_optimize_lightgbm(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test LightGBM optimization with Optuna."""
    X, y = mock_dataset

    results, best_params = optimize_lightgbm(X, y, study_name="test_study", n_trials=5)

    _assert_results_structure(results, best_params)
    _assert_results_values(results, expected_trials=5, expected_study="test_study")
    _assert_best_params_structure(results["best_params"])


def _create_test_results() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create test results dictionaries."""
    results_complete = {
        "best_params": {"n_estimators": 100},
        "best_rmse_cv": 0.5,
        "train_rmse": 0.55,
        "n_trials": 10,
        "study_name": "test_complete",
    }
    results_without = {
        "best_params": {"n_estimators": 150},
        "best_rmse_cv": 0.6,
        "train_rmse": 0.65,
        "n_trials": 10,
        "study_name": "test_without",
    }
    results_technical = {
        "best_params": {"n_estimators": 75},
        "best_rmse_cv": 0.55,
        "train_rmse": 0.6,
        "n_trials": 10,
        "study_name": "test_technical",
    }
    return results_complete, results_without, results_technical


def _load_saved_results(output_file: Path) -> dict[str, Any]:
    """Load saved results from JSON file."""
    assert output_file.exists()
    with open(output_file) as f:
        return json.load(f)


def _assert_saved_results_keys(saved_data: dict[str, Any]) -> None:
    """Assert saved results contain required keys."""
    required_keys = {
        "lightgbm_dataset_complete",
        "lightgbm_dataset_without_insights",
        "lightgbm_dataset_technical_indicators",
    }
    assert required_keys.issubset(set(saved_data.keys()))


def _assert_saved_results_values(
    saved_data: dict[str, Any],
    results_complete: dict[str, Any],
    results_without: dict[str, Any],
    results_technical: dict[str, Any],
) -> None:
    """Assert saved results values match expected."""
    assert saved_data["lightgbm_dataset_complete"] == results_complete
    assert saved_data["lightgbm_dataset_without_insights"] == results_without
    assert saved_data["lightgbm_dataset_technical_indicators"] == results_technical


def _assert_saved_results(
    output_file: Path,
    results_complete: dict[str, Any],
    results_without: dict[str, Any],
    results_technical: dict[str, Any],
) -> None:
    """Assert saved results match expected."""
    saved_data = _load_saved_results(output_file)
    _assert_saved_results_keys(saved_data)
    _assert_saved_results_values(saved_data, results_complete, results_without, results_technical)


def test_save_optimization_results(tmp_path: Path) -> None:
    """Test saving optimization results to JSON."""
    results_complete, results_without, results_technical = _create_test_results()
    output_file = tmp_path / "results" / "optimization_results.json"

    save_optimization_results(
        results_complete,
        results_without,
        output_file,
        results_technical=results_technical,
    )

    _assert_saved_results(output_file, results_complete, results_without, results_technical)


def _create_mock_datasets(
    tmp_path: Path, n_samples: int
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Create mock datasets for integration test."""
    np.random.seed(42)
    df_complete = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_samples),
            "log_volatility": np.random.randn(n_samples),
            "log_volatility_t": np.random.randn(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
            "ticker": ["AAA"] * (n_samples - 20) + ["BBB"] * 20,
            "ticker_id": [0] * (n_samples - 20) + [1] * 20,
            "split": ["train"] * (n_samples - 20) + ["test"] * 20,
        }
    )
    df_without = df_complete.copy()
    dataset_complete = tmp_path / "data" / "lightgbm_dataset_complete.csv"
    dataset_without = tmp_path / "data" / "lightgbm_dataset_without_insights.csv"
    dataset_complete.parent.mkdir(parents=True, exist_ok=True)
    dataset_without.parent.mkdir(parents=True, exist_ok=True)
    df_complete.to_csv(dataset_complete, index=False)
    df_without.to_csv(dataset_without, index=False)
    return df_complete, df_without, dataset_complete, dataset_without


def _patch_optimization_paths(
    monkeypatch: pytest.MonkeyPatch,
    dataset_complete: Path,
    dataset_without: Path,
) -> None:
    """Patch optimization paths for integration test."""
    monkeypatch.setattr("src.constants.LIGHTGBM_DATASET_COMPLETE_FILE", dataset_complete)
    monkeypatch.setattr("src.constants.LIGHTGBM_DATASET_WITHOUT_INSIGHTS_FILE", dataset_without)
    monkeypatch.setattr("src.constants.LIGHTGBM_DATASET_SIGMA_PLUS_BASE_FILE", dataset_complete)
    monkeypatch.setattr("src.constants.LIGHTGBM_DATASET_LOG_VOLATILITY_ONLY_FILE", dataset_without)
    # Mock _save_and_log_results to avoid writing to official paths during tests
    monkeypatch.setattr(
        "src.lightgbm.optimisation.execution._save_and_log_results", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.lightgbm.optimisation.execution._prepare_optimization_tasks",
        lambda n_trials: [
            (dataset_complete, "lightgbm_complete", n_trials),
            (dataset_without, "lightgbm_without_insights", n_trials),
            (dataset_complete, "lightgbm_sigma_plus_base", n_trials),
            (dataset_without, "lightgbm_log_volatility_only", n_trials),
        ],
    )


def _assert_integration_types(
    results_complete: dict[str, Any], results_without: dict[str, Any]
) -> None:
    """Assert integration results types."""
    assert isinstance(results_complete, dict)
    assert isinstance(results_without, dict)


def _assert_integration_keys(
    results_complete: dict[str, Any], results_without: dict[str, Any]
) -> None:
    """Assert integration results keys."""
    assert "best_rmse_cv" in results_complete
    assert "best_rmse_cv" in results_without


def _assert_integration_trials(
    results_complete: dict[str, Any], results_without: dict[str, Any], expected_trials: int
) -> None:
    """Assert integration results trials."""
    assert results_complete["n_trials"] == expected_trials
    assert results_without["n_trials"] == expected_trials


def _assert_integration_results(
    results_complete: dict[str, Any], results_without: dict[str, Any], expected_trials: int
) -> None:
    """Assert integration test results."""
    _assert_integration_types(results_complete, results_without)
    _assert_integration_keys(results_complete, results_without)
    _assert_integration_trials(results_complete, results_without, expected_trials)


def test_run_optimization_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full optimization run with mocked data."""
    _, _, dataset_complete, dataset_without = _create_mock_datasets(tmp_path, n_samples=100)
    _patch_optimization_paths(monkeypatch, dataset_complete, dataset_without)

    results_complete, results_without = run_optimization(n_trials=3)
    _assert_integration_results(results_complete, results_without, expected_trials=3)


def test_n_splits_constant() -> None:
    """Test that LIGHTGBM_OPTIMIZATION_N_SPLITS constant is reasonable for small dataset."""
    assert LIGHTGBM_OPTIMIZATION_N_SPLITS == 5
    assert isinstance(LIGHTGBM_OPTIMIZATION_N_SPLITS, int)
    assert LIGHTGBM_OPTIMIZATION_N_SPLITS > 1


# Removed test_walk_forward_cv_all_folds_used
# This test was specific to Random Forest and is no longer applicable with LightGBM


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
