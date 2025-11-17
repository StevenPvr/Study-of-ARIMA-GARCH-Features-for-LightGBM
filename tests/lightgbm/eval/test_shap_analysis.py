"""Tests for SHAP analysis utilities."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Set matplotlib backend before any imports that use it
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.lightgbm.eval.shap_analysis import compute_shap_values


@pytest.fixture
def mock_model(tmp_path: Path) -> RandomForestRegressor:
    """Create a mock LightGBM model."""
    np.random.seed(42)
    X_train = pd.DataFrame(
        {
            "log_volatility_t": np.random.randn(100),
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        }
    )
    y_train = pd.Series(np.random.randn(100), name="log_volatility")

    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def _assert_plot_path_exists(plot_path: Path) -> None:
    """Assert plot path exists and is a file."""
    assert plot_path.exists()
    assert plot_path.is_file()


def _assert_plot_path_format(plot_path: Path, model_name: str, tmp_path: Path) -> None:
    """Assert plot path has correct format."""
    assert plot_path.suffix == ".png"
    assert model_name in plot_path.name
    assert str(tmp_path) in str(plot_path), "Plot should be in temporary directory"


def _assert_shap_plot_path(plot_path: Path, model_name: str, tmp_path: Path) -> None:
    """Assert SHAP plot path is valid."""
    _assert_plot_path_exists(plot_path)
    _assert_plot_path_format(plot_path, model_name, tmp_path)


def _assert_shap_explanation(explanation, X: pd.DataFrame) -> None:
    """Assert SHAP explanation has correct shape."""
    assert isinstance(explanation.values, np.ndarray)
    assert explanation.values.shape[0] == len(X)
    assert explanation.values.shape[1] == X.shape[1]


def test_compute_shap_values(
    mock_model: RandomForestRegressor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test computing SHAP values and creating plots."""
    # Mock shap.plots.beeswarm, plt.gcf(), and plt.close() to avoid matplotlib compatibility issues
    # The actual code calls plt.gcf() after beeswarm and plt.close() at the end
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    from matplotlib.figure import Figure

    # Create figure directly without going through pyplot to avoid state issues
    test_fig = Figure(figsize=(12, 8))
    test_ax = test_fig.add_subplot(111)

    def mock_beeswarm(*args, **kwargs):
        """Mock beeswarm plot function."""
        return test_ax

    def mock_gcf():
        """Mock plt.gcf() to return our test figure."""
        return test_fig

    def mock_close(fig=None):
        """Mock plt.close() to be a no-op since our figure isn't registered with pyplot."""
        pass

    # Patch both the module-level import and the matplotlib.pyplot module
    from src.lightgbm.eval import shap_analysis

    monkeypatch.setattr("shap.plots.beeswarm", mock_beeswarm)
    monkeypatch.setattr(shap_analysis.plt, "gcf", mock_gcf)
    monkeypatch.setattr(shap_analysis.plt, "close", mock_close)
    monkeypatch.setattr("matplotlib.pyplot.gcf", mock_gcf)
    monkeypatch.setattr("matplotlib.pyplot.close", mock_close)

    np.random.seed(42)
    X = pd.DataFrame(
        {
            "log_volatility_t": np.random.randn(50),
            "feature_1": np.random.randn(50),
            "feature_2": np.random.randn(50),
            "feature_3": np.random.randn(50),
        }
    )

    output_dir = tmp_path / "shap"
    explanation, plot_path = compute_shap_values(
        mock_model, X, "test_model", output_dir=output_dir, max_display=10
    )

    _assert_shap_plot_path(plot_path, "test_model", tmp_path)
    _assert_shap_explanation(explanation, X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
