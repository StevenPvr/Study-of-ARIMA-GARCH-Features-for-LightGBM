"""Unit tests for Random Forest optimization module."""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from unittest.mock import MagicMock, patch

import json
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.constants import RF_OPTIMIZATION_N_SPLITS
from src.random_forest.optimisation.optimisation import (
    load_dataset,
    objective,
    optimize_random_forest,
    run_optimization,
    save_optimization_results,
    walk_forward_cv_score,
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
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randn(n_samples), name="weighted_log_return")
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
            "weighted_log_return": np.random.randn(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )
    file_path = tmp_path / "test_dataset.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_dataset_success(mock_dataset_file: Path) -> None:
    """Test loading dataset from CSV file."""
    X, y = load_dataset(mock_dataset_file)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 100
    assert len(y) == 100
    assert y.name == "weighted_log_return"
    assert "date" not in X.columns
    assert "weighted_log_return" not in X.columns
    assert list(X.columns) == ["feature1", "feature2", "feature3"]


def test_load_dataset_file_not_found() -> None:
    """Test loading dataset raises error when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(Path("nonexistent.csv"))


def test_load_dataset_empty_file(tmp_path: Path) -> None:
    """Test loading empty dataset raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(empty_file, index=False)

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_dataset(empty_file)


def test_load_dataset_missing_target(tmp_path: Path) -> None:
    """Test loading dataset without target column raises ValueError."""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    file_path = tmp_path / "no_target.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="must contain 'weighted_log_return' column"):
        load_dataset(file_path)


def test_load_dataset_filters_train_split(tmp_path: Path) -> None:
    """Test loading dataset filters to train split only."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10),
            "weighted_log_return": np.random.randn(10),
            "feature1": np.random.randn(10),
            "split": ["train"] * 6 + ["test"] * 4,
        }
    )
    file_path = tmp_path / "with_split.csv"
    df.to_csv(file_path, index=False)

    X, y = load_dataset(file_path)

    assert len(X) == 6  # Only train split
    assert len(y) == 6
    assert "split" not in X.columns


def test_walk_forward_cv_score(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test walk-forward cross-validation scoring."""
    X, y = mock_dataset
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    }

    # Create mock trial for pruning
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    loss_mean = walk_forward_cv_score(trial, X, y, params)

    assert isinstance(loss_mean, float)
    assert np.isfinite(loss_mean)  # Log loss can be negative when MSE < 1
    assert trial.report.call_count == RF_OPTIMIZATION_N_SPLITS


def test_walk_forward_cv_score_deterministic(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test walk-forward CV produces same results with same random state."""
    X, y = mock_dataset
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    }

    # Create mock trials for pruning
    study1 = optuna.create_study(direction="minimize")
    trial1 = study1.ask()
    trial1.report = MagicMock()
    trial1.should_prune = MagicMock(return_value=False)

    study2 = optuna.create_study(direction="minimize")
    trial2 = study2.ask()
    trial2.report = MagicMock()
    trial2.should_prune = MagicMock(return_value=False)

    loss1 = walk_forward_cv_score(trial1, X, y, params)
    loss2 = walk_forward_cv_score(trial2, X, y, params)

    # Use approximate equality for floating-point comparisons
    assert loss1 == pytest.approx(loss2)


def test_objective(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test Optuna objective function."""
    X, y = mock_dataset

    # Create mock trial
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    # Mock suggest methods
    trial.suggest_int = MagicMock(side_effect=[10, 5, 5, 2])
    trial.suggest_categorical = MagicMock(side_effect=["sqrt", True])
    # Mock pruning methods for walk_forward_cv_score
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    loss = objective(trial, X, y)

    assert isinstance(loss, float)
    assert np.isfinite(loss)  # Log loss can be negative when MSE < 1
    assert trial.suggest_int.call_count == 4
    assert trial.suggest_categorical.call_count == 2


def test_optimize_random_forest(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test Random Forest optimization with Optuna."""
    X, y = mock_dataset

    results, study = optimize_random_forest(X, y, study_name="test_study", n_trials=5)

    assert isinstance(results, dict)
    assert isinstance(study, optuna.Study)
    assert "best_params" in results
    assert "best_loss" in results
    assert "n_trials" in results
    assert "study_name" in results

    assert results["n_trials"] == 5
    assert results["study_name"] == "test_study"
    assert np.isfinite(results["best_loss"])  # Log loss can be negative when MSE < 1

    # Check best params structure
    best_params = results["best_params"]
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params
    assert "min_samples_leaf" in best_params
    assert "max_features" in best_params
    assert "bootstrap" in best_params


def test_save_optimization_results(tmp_path: Path) -> None:
    """Test saving optimization results to JSON."""
    results_complete = {
        "best_params": {"n_estimators": 100},
        "best_loss": 0.5,
        "n_trials": 10,
        "study_name": "test_complete",
    }
    results_without = {
        "best_params": {"n_estimators": 150},
        "best_loss": 0.6,
        "n_trials": 10,
        "study_name": "test_without",
    }

    output_file = tmp_path / "results" / "optimization_results.json"
    save_optimization_results(results_complete, results_without, output_file)

    assert output_file.exists()

    with open(output_file) as f:
        saved_data = json.load(f)

    assert "rf_dataset_complete" in saved_data
    assert "rf_dataset_without_insights" in saved_data
    assert saved_data["rf_dataset_complete"] == results_complete
    assert saved_data["rf_dataset_without_insights"] == results_without


def test_run_optimization_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full optimization run with mocked data."""
    # Create mock datasets
    np.random.seed(42)
    n_samples = 100

    df_complete = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n_samples),
            "weighted_log_return": np.random.randn(n_samples),
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
            "feature3": np.random.randn(n_samples),
        }
    )

    df_without = df_complete.copy()

    dataset_complete = tmp_path / "data" / "rf_dataset_complete.csv"
    dataset_without = tmp_path / "data" / "rf_dataset_without_insights.csv"
    dataset_complete.parent.mkdir(parents=True, exist_ok=True)
    dataset_without.parent.mkdir(parents=True, exist_ok=True)

    df_complete.to_csv(dataset_complete, index=False)
    df_without.to_csv(dataset_without, index=False)

    # Monkeypatch paths
    monkeypatch.setattr(
        "src.constants.RF_DATASET_COMPLETE_FILE",
        dataset_complete,
    )
    monkeypatch.setattr(
        "src.constants.RF_DATASET_WITHOUT_INSIGHTS_FILE",
        dataset_without,
    )
    monkeypatch.setattr(
        "src.constants.RF_OPTIMIZATION_RESULTS_FILE",
        tmp_path / "results" / "optimization_results.json",
    )

    # Run optimization with fewer trials for speed
    results_complete, results_without = run_optimization(n_trials=3)

    assert isinstance(results_complete, dict)
    assert isinstance(results_without, dict)
    assert "best_loss" in results_complete
    assert "best_loss" in results_without
    assert results_complete["n_trials"] == 3
    assert results_without["n_trials"] == 3


def test_n_splits_constant() -> None:
    """Test that RF_OPTIMIZATION_N_SPLITS constant is reasonable for small dataset."""
    assert RF_OPTIMIZATION_N_SPLITS == 5
    assert isinstance(RF_OPTIMIZATION_N_SPLITS, int)
    assert RF_OPTIMIZATION_N_SPLITS > 1


def test_walk_forward_cv_all_folds_used(mock_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test that all CV folds are actually used."""
    X, y = mock_dataset
    params = {
        "n_estimators": 5,
        "max_depth": 3,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
    }

    # Create mock trial for pruning
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.report = MagicMock()
    trial.should_prune = MagicMock(return_value=False)

    # Mock to count folds
    original_fit = RandomForestRegressor.fit
    fit_count = {"count": 0}

    def counting_fit(self, X, y):
        fit_count["count"] += 1
        return original_fit(self, X, y)

    with patch.object(RandomForestRegressor, "fit", counting_fit):
        walk_forward_cv_score(trial, X, y, params)

    # Should fit RF_OPTIMIZATION_N_SPLITS times
    assert fit_count["count"] == RF_OPTIMIZATION_N_SPLITS


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
