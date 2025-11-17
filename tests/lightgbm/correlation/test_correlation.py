"""Unit tests for Spearman correlation calculation."""

from __future__ import annotations

import sys
from pathlib import Path

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
    if "src.lightgbm.correlation.correlation" in sys.modules:
        del sys.modules["src.lightgbm.correlation.correlation"]
    if "src.lightgbm.correlation" in sys.modules:
        del sys.modules["src.lightgbm.correlation"]


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

    correlation_file = _project_root.joinpath("src", "lightgbm", "correlation", "correlation.py")
    spec = importlib.util.spec_from_file_location(
        "src.lightgbm.correlation.correlation", correlation_file
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


def _create_test_datasets(
    sample_dataset: pd.DataFrame, tmp_path: Path
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Create seven test datasets for correlation testing."""
    complete_path = tmp_path / "lightgbm_dataset_complete.csv"
    without_path = tmp_path / "lightgbm_dataset_without_insights.csv"
    log_volatility_path = tmp_path / "lightgbm_dataset_log_volatility_only.csv"
    sigma_path = tmp_path / "lightgbm_dataset_sigma_plus_base.csv"
    insights_only_path = tmp_path / "lightgbm_dataset_insights_only.csv"
    technical_only_path = tmp_path / "lightgbm_dataset_technical_only_no_target_lags.csv"
    technical_plus_insights_path = (
        tmp_path / "lightgbm_dataset_technical_plus_insights_no_target_lags.csv"
    )

    # Add missing columns if they don't exist
    df_enhanced = sample_dataset.copy()
    if "split" not in df_enhanced.columns:
        df_enhanced["split"] = ["train"] * (len(df_enhanced) - 5) + ["test"] * 5
    if "weighted_log_return" not in df_enhanced.columns:
        # Use a derived column or create synthetic log return
        np.random.seed(42)
        df_enhanced["weighted_log_return"] = np.random.randn(len(df_enhanced)) * 0.01

    df_enhanced.to_csv(complete_path, index=False)
    # Remove some columns for the "without insights" dataset
    df_without = df_enhanced.drop(columns=["sigma2_garch"])
    df_without.to_csv(without_path, index=False)
    # Create log-volatility-only dataset
    df_log_volatility = df_enhanced[["date", "split", "rsi_14", "weighted_log_return"]]
    df_log_volatility.to_csv(log_volatility_path, index=False)
    # Create sigma plus base dataset (subset of complete)
    df_sigma = df_enhanced[["date", "split", "rsi_14", "sigma2_garch", "weighted_log_return"]]
    df_sigma.to_csv(sigma_path, index=False)
    # Create insights-only dataset (ARIMA-GARCH insights + ticker_id + log_volatility)
    if "arima_pred_return" in df_enhanced.columns:
        df_insights = df_enhanced[
            [
                "date",
                "split",
                "arima_pred_return",
                "arima_residual_return",
                "sigma2_garch",
                "sigma_garch",
                "std_resid_garch",
                "ticker_id",
                "weighted_log_return",
            ]
        ]
    else:
        # Fallback if ARIMA-GARCH columns don't exist
        df_insights = df_enhanced[["date", "split", "sigma2_garch", "weighted_log_return"]]
    df_insights.to_csv(insights_only_path, index=False)

    # Create technical-only dataset (technical indicators without target lags)
    df_technical_only = df_enhanced[["date", "split", "rsi_14", "sma_20", "ema_20", "macd"]]
    df_technical_only.to_csv(technical_only_path, index=False)

    # Create technical-plus-insights dataset (technical indicators with ARIMA-GARCH insights)
    if "arima_pred_return" in df_enhanced.columns:
        df_technical_plus_insights = df_enhanced[
            [
                "date",
                "split",
                "rsi_14",
                "sma_20",
                "ema_20",
                "macd",
                "arima_pred_return",
                "arima_residual_return",
                "sigma2_garch",
                "sigma_garch",
                "std_resid_garch",
            ]
        ]
    else:
        # Fallback if ARIMA-GARCH columns don't exist
        df_technical_plus_insights = df_enhanced[
            ["date", "split", "rsi_14", "sma_20", "ema_20", "macd", "sigma2_garch"]
        ]
    df_technical_plus_insights.to_csv(technical_plus_insights_path, index=False)

    return (
        complete_path,
        without_path,
        log_volatility_path,
        sigma_path,
        insights_only_path,
        technical_only_path,
        technical_plus_insights_path,
    )


def _assert_correlation_matrix_not_empty(corr_matrix: pd.DataFrame, name: str) -> None:
    """Assert that a correlation matrix is not empty.

    Args:
        corr_matrix: Correlation matrix to check.
        name: Name of the matrix for error messages.
    """
    assert corr_matrix.shape[0] > 0, f"{name} correlation matrix is empty"


def _assert_correlation_results(
    corr_complete: pd.DataFrame,
    corr_without: pd.DataFrame,
    corr_log_volatility: pd.DataFrame,
    corr_sigma: pd.DataFrame,
    corr_insights: pd.DataFrame,
    corr_technical_only: pd.DataFrame,
    corr_technical_plus_insights: pd.DataFrame,
) -> None:
    """Assert correlation matrices have expected properties."""
    _assert_correlation_matrix_not_empty(corr_complete, "Complete")
    _assert_correlation_matrix_not_empty(corr_without, "Without insights")
    _assert_correlation_matrix_not_empty(corr_log_volatility, "Log volatility only")
    _assert_correlation_matrix_not_empty(corr_sigma, "Sigma plus base")
    _assert_correlation_matrix_not_empty(corr_insights, "Insights only")
    _assert_correlation_matrix_not_empty(corr_technical_only, "Technical only no target lags")
    _assert_correlation_matrix_not_empty(
        corr_technical_plus_insights, "Technical plus insights no target lags"
    )

    # Complete dataset should have more columns (more features)
    assert corr_complete.shape[0] >= corr_without.shape[0]


def _assert_plot_file_exists(plot_path: Path, name: str) -> None:
    """Assert that a plot file exists and is not empty.

    Args:
        plot_path: Path to the plot file.
        name: Name of the plot for error messages.
    """
    assert plot_path.exists(), f"{name} plot file does not exist: {plot_path}"
    assert plot_path.stat().st_size > 0, f"{name} plot file is empty: {plot_path}"


def _assert_plot_files_exist(output_dir: Path) -> None:
    """Assert that correlation plot files were created and are not empty."""
    plot_files = [
        (output_dir / "lightgbm_correlation_complete.png", "Complete"),
        (output_dir / "lightgbm_correlation_without_insights.png", "Without insights"),
        (output_dir / "lightgbm_correlation_log_volatility_only.png", "Log volatility only"),
        (output_dir / "lightgbm_correlation_sigma_plus_base.png", "Sigma plus base"),
        (output_dir / "lightgbm_correlation_insights_only.png", "Insights only"),
        (
            output_dir / "lightgbm_correlation_technical_only_no_target_lags.png",
            "Technical only no target lags",
        ),
        (
            output_dir / "lightgbm_correlation_technical_plus_insights_no_target_lags.png",
            "Technical plus insights no target lags",
        ),
    ]

    for plot_path, name in plot_files:
        _assert_plot_file_exists(plot_path, name)


def test_compute_correlations(
    sample_dataset: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test computing correlations for all datasets."""
    corr_module = _restore_matplotlib_and_import_correlation()

    # Mock LIGHTGBM_CORRELATION_PLOTS_DIR to use tmp_path to avoid creating permanent files
    monkeypatch.setattr(corr_module, "LIGHTGBM_CORRELATION_PLOTS_DIR", tmp_path)

    (
        complete_path,
        without_path,
        log_volatility_path,
        sigma_path,
        insights_only_path,
        technical_only_path,
        technical_plus_insights_path,
    ) = _create_test_datasets(sample_dataset, tmp_path)

    (
        corr_complete,
        corr_without,
        corr_log_volatility,
        corr_sigma,
        corr_insights,
        corr_technical_only,
        corr_technical_plus_insights,
    ) = corr_module.compute_correlations(
        complete_dataset_path=complete_path,
        without_insights_dataset_path=without_path,
        log_volatility_only_dataset_path=log_volatility_path,
        sigma_plus_base_dataset_path=sigma_path,
        insights_only_dataset_path=insights_only_path,
        technical_only_no_target_lags_dataset_path=technical_only_path,
        technical_plus_insights_no_target_lags_dataset_path=technical_plus_insights_path,
        output_dir=tmp_path,
    )

    _assert_correlation_results(
        corr_complete,
        corr_without,
        corr_log_volatility,
        corr_sigma,
        corr_insights,
        corr_technical_only,
        corr_technical_plus_insights,
    )
    _assert_plot_files_exist(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
