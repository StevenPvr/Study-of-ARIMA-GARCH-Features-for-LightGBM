"""Unit tests for Spearman correlation calculation."""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any

import numpy as np
import pandas as pd
import pytest

# Don't import correlation module at top level - it imports matplotlib which is mocked
# Import inside test functions instead


def _remove_module_from_cache(module_name: str) -> None:
    """Remove a module from sys.modules if it exists."""
    if module_name in sys.modules:
        del sys.modules[module_name]


def _remove_main_modules() -> None:
    """Remove main matplotlib and seaborn modules from sys.modules."""
    modules_to_restore = [
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.backends",
        "matplotlib.backends.backend_agg",
        "seaborn",
    ]

    for mod_name in modules_to_restore:
        _remove_module_from_cache(mod_name)


def _remove_submodules() -> None:
    """Remove matplotlib and seaborn submodules from sys.modules."""
    modules_to_clean = [
        k
        for k in list(sys.modules.keys())
        if k.startswith("matplotlib.") or k.startswith("seaborn")
    ]
    for mod_name in modules_to_clean:
        _remove_module_from_cache(mod_name)


def _remove_mocked_modules() -> None:
    """Remove mocked matplotlib and seaborn modules from sys.modules.

    This is needed because conftest.py mocks matplotlib, but some tests need the real thing.
    """
    _remove_main_modules()
    _remove_submodules()


def _remove_correlation_module() -> None:
    """Remove correlation module from sys.modules if already imported."""
    if "src.random_forest.correlation.correlation" in sys.modules:
        del sys.modules["src.random_forest.correlation.correlation"]
    if "src.random_forest.correlation" in sys.modules:
        del sys.modules["src.random_forest.correlation"]


def _setup_matplotlib_backend() -> None:
    """Set up matplotlib with Agg backend for non-interactive use."""
    import matplotlib

    # Use force=True if available (matplotlib >= 3.1.0), otherwise just use()
    try:
        matplotlib.use("Agg", force=True)
    except TypeError:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401


def _import_correlation_module() -> Any:
    """Import correlation module directly from file to avoid triggering __init__.py.

    Returns:
        The imported correlation module.

    Raises:
        ImportError: If module spec creation or loading fails.
    """
    import importlib.util

    correlation_file = _script_dir / "correlation.py"
    spec = importlib.util.spec_from_file_location(
        "src.random_forest.correlation.correlation", correlation_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {correlation_file}")
    corr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corr_module)

    return corr_module


def _restore_matplotlib_and_import_correlation() -> Any:
    """Helper function to restore real matplotlib and import correlation module.

    This is needed because conftest.py mocks matplotlib, but some tests need the real thing.

    Returns:
        The imported correlation module.
    """
    _remove_mocked_modules()
    _remove_correlation_module()
    _setup_matplotlib_backend()
    return _import_correlation_module()


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create sample dataset for testing."""
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    np.random.seed(42)

    return pd.DataFrame(
        {
            "date": dates,
            "weighted_closing": np.random.randn(50) * 10 + 100,
            "weighted_open": np.random.randn(50) * 10 + 100,
            "rsi_14": np.random.uniform(0, 100, 50),
            "sma_20": np.random.randn(50) * 10 + 100,
            "ema_20": np.random.randn(50) * 10 + 100,
            "macd": np.random.randn(50),
            "sigma2_garch": np.random.uniform(0.0001, 0.001, 50),
        }
    )


def test_load_dataset(sample_dataset: pd.DataFrame, tmp_path: Path) -> None:
    """Test loading a dataset."""
    corr_module = _restore_matplotlib_and_import_correlation()

    dataset_path = tmp_path / "test_dataset.csv"
    sample_dataset.to_csv(dataset_path, index=False)

    df = corr_module.load_dataset(dataset_path)
    assert len(df) == len(sample_dataset)
    assert len(df.columns) == len(sample_dataset.columns)


def test_load_dataset_missing_file() -> None:
    """Test loading a non-existent dataset."""
    corr_module = _restore_matplotlib_and_import_correlation()

    fake_path = Path("/nonexistent/path/dataset.csv")
    with pytest.raises(FileNotFoundError):
        corr_module.load_dataset(fake_path)


def _assert_correlation_matrix_shape(corr_matrix: pd.DataFrame) -> None:
    """Assert that correlation matrix is square."""
    assert corr_matrix.shape[0] == corr_matrix.shape[1]


def _assert_correlation_matrix_properties(corr_matrix: pd.DataFrame) -> None:
    """Assert correlation matrix has expected properties."""
    # Check that diagonal values are 1.0 (perfect correlation with itself)
    assert (np.diag(corr_matrix) == 1.0).all()

    # Check that matrix is symmetric
    assert np.allclose(corr_matrix.values, corr_matrix.values.T)

    # Check that values are between -1 and 1
    assert (corr_matrix >= -1).all().all()
    assert (corr_matrix <= 1).all().all()


def test_calculate_spearman_correlation(sample_dataset: pd.DataFrame) -> None:
    """Test Spearman correlation calculation."""
    corr_module = _restore_matplotlib_and_import_correlation()

    corr_matrix = corr_module.calculate_spearman_correlation(sample_dataset)

    _assert_correlation_matrix_shape(corr_matrix)
    _assert_correlation_matrix_properties(corr_matrix)


def test_calculate_spearman_correlation_empty_dataframe() -> None:
    """Test correlation calculation with empty DataFrame."""
    corr_module = _restore_matplotlib_and_import_correlation()

    df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        corr_module.calculate_spearman_correlation(df)


def test_calculate_spearman_correlation_no_numeric() -> None:
    """Test correlation calculation with no numeric columns."""
    corr_module = _restore_matplotlib_and_import_correlation()

    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10, freq="D")})
    with pytest.raises(ValueError, match="No numeric columns found"):
        corr_module.calculate_spearman_correlation(df)


def test_plot_correlation_matrix(
    sample_dataset: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test plotting correlation matrix."""
    corr_module = _restore_matplotlib_and_import_correlation()

    corr_matrix = corr_module.calculate_spearman_correlation(sample_dataset)
    output_path = tmp_path / "test_correlation.png"

    corr_module.plot_correlation_matrix(corr_matrix, output_path, "test dataset")

    assert output_path.exists()
    # Check that file is not empty (basic check)
    assert output_path.stat().st_size > 0


def _create_test_datasets(sample_dataset: pd.DataFrame, tmp_path: Path) -> tuple[Path, Path]:
    """Create two test datasets for correlation testing."""
    complete_path = tmp_path / "rf_dataset_complete.csv"
    without_path = tmp_path / "rf_dataset_without_insights.csv"

    sample_dataset.to_csv(complete_path, index=False)
    # Remove some columns for the "without insights" dataset
    df_without = sample_dataset.drop(columns=["sigma2_garch"])
    df_without.to_csv(without_path, index=False)

    return complete_path, without_path


def _assert_correlation_results(corr_complete: pd.DataFrame, corr_without: pd.DataFrame) -> None:
    """Assert correlation matrices have expected properties."""
    # Check that correlation matrices were computed
    assert corr_complete.shape[0] > 0
    assert corr_without.shape[0] > 0

    # Complete dataset should have more columns (more features)
    assert corr_complete.shape[0] >= corr_without.shape[0]


def _assert_plot_files_exist(output_dir: Path) -> None:
    """Assert that correlation plot files were created and are not empty."""
    complete_plot = output_dir / "rf_correlation_complete.png"
    without_plot = output_dir / "rf_correlation_without_insights.png"

    assert complete_plot.exists()
    assert without_plot.exists()

    assert complete_plot.stat().st_size > 0
    assert without_plot.stat().st_size > 0


def test_compute_correlations(
    sample_dataset: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test computing correlations for both datasets."""
    corr_module = _restore_matplotlib_and_import_correlation()

    complete_path, without_path = _create_test_datasets(sample_dataset, tmp_path)

    corr_complete, corr_without = corr_module.compute_correlations(
        complete_dataset_path=complete_path,
        without_insights_dataset_path=without_path,
        output_dir=tmp_path,
    )

    _assert_correlation_results(corr_complete, corr_without)
    _assert_plot_files_exist(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
