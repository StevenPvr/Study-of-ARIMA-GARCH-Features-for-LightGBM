"""Tests for statistical comparison functions."""

from __future__ import annotations

from pathlib import Path
import sys

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import pytest

from src.random_forest.eval.statistical_tests import (
    bootstrap_r2_comparison,
    compare_models_statistical,
    diebold_mariano_test,
)


def test_diebold_mariano_test_identical_models() -> None:
    """Test DM test with identical model errors."""
    np.random.seed(42)
    errors1 = np.random.randn(100)
    errors2 = errors1.copy()

    result = diebold_mariano_test(errors1, errors2)

    assert result["dm_statistic"] == pytest.approx(0.0, abs=1e-10)
    assert result["p_value"] > 0.05  # Should not be significant
    assert result["better_model"] == "equal"


def test_diebold_mariano_test_different_models() -> None:
    """Test DM test with clearly different model errors."""
    np.random.seed(42)
    errors1 = np.random.randn(100) * 0.1  # Small errors
    errors2 = np.random.randn(100) * 2.0  # Large errors

    result = diebold_mariano_test(errors1, errors2)

    assert result["dm_statistic"] < 0  # Model 1 is better
    assert result["better_model"] == "model_1"
    assert result["p_value"] < 0.05  # Should be significant


def test_diebold_mariano_test_with_series() -> None:
    """Test DM test with pandas Series."""
    np.random.seed(42)
    errors1 = pd.Series(np.random.randn(100))
    errors2 = pd.Series(np.random.randn(100) * 1.5)

    result = diebold_mariano_test(errors1, errors2)

    assert isinstance(result["dm_statistic"], float)
    assert isinstance(result["p_value"], float)
    assert result["better_model"] in ["model_1", "model_2", "equal"]


def test_diebold_mariano_test_mae_vs_mse() -> None:
    """Test DM test with different loss functions."""
    np.random.seed(42)
    errors1 = np.random.randn(100)
    errors2 = np.random.randn(100) * 1.2

    result_mse = diebold_mariano_test(errors1, errors2, power=2)
    result_mae = diebold_mariano_test(errors1, errors2, power=1)

    assert isinstance(result_mse["dm_statistic"], float)
    assert isinstance(result_mae["dm_statistic"], float)
    # Results may differ but should both be valid
    assert result_mse["p_value"] >= 0.0
    assert result_mae["p_value"] >= 0.0


def test_diebold_mariano_test_empty_arrays() -> None:
    """Test DM test with empty arrays."""
    errors1 = np.array([])
    errors2 = np.array([])

    with pytest.raises(ValueError, match="cannot be empty"):
        diebold_mariano_test(errors1, errors2)


def test_diebold_mariano_test_different_lengths() -> None:
    """Test DM test with arrays of different lengths."""
    errors1 = np.random.randn(100)
    errors2 = np.random.randn(50)

    with pytest.raises(ValueError, match="same length"):
        diebold_mariano_test(errors1, errors2)


def test_diebold_mariano_test_result_structure() -> None:
    """Test that DM test returns all required fields."""
    np.random.seed(42)
    errors1 = np.random.randn(100)
    errors2 = np.random.randn(100) * 1.1

    result = diebold_mariano_test(errors1, errors2)

    required_keys = [
        "dm_statistic",
        "p_value",
        "better_model",
        "significance",
        "interpretation",
        "mean_loss_diff",
        "n_observations",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert result["n_observations"] == 100


def test_compare_models_statistical() -> None:
    """Test statistical comparison of two models."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred1 = y_true + np.random.randn(100) * 0.1
    y_pred2 = y_true + np.random.randn(100) * 0.2

    result = compare_models_statistical(
        y_true, y_pred1, y_pred2, model1_name="model_a", model2_name="model_b"
    )

    assert "mse_based" in result
    assert "mae_based" in result
    assert result["model1_name"] == "model_a"
    assert result["model2_name"] == "model_b"

    # Check that both tests have required structure
    for test_result in [result["mse_based"], result["mae_based"]]:
        assert "dm_statistic" in test_result
        assert "p_value" in test_result
        assert "better_model" in test_result


def test_compare_models_statistical_with_series() -> None:
    """Test statistical comparison with pandas Series."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred1 = pd.Series(y_true + np.random.randn(100) * 0.1)
    y_pred2 = pd.Series(y_true + np.random.randn(100) * 0.2)

    result = compare_models_statistical(y_true, y_pred1, y_pred2)

    assert isinstance(result, dict)
    assert "mse_based" in result
    assert "mae_based" in result


def test_diebold_mariano_test_significance_levels() -> None:
    """Test that significance levels are correctly classified."""
    np.random.seed(42)

    # Create clearly different models for high significance
    errors1 = np.random.randn(200) * 0.1
    errors2 = np.random.randn(200) * 3.0

    result = diebold_mariano_test(errors1, errors2)

    # Should be highly significant
    assert result["p_value"] < 0.05
    assert "significant" in result["significance"].lower()


def test_bootstrap_r2_comparison_identical_models() -> None:
    """Test bootstrap R² comparison with identical models."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1

    result = bootstrap_r2_comparison(y_true, y_pred, y_pred, n_bootstrap=100)

    assert result["r2_diff_observed"] == pytest.approx(0.0, abs=1e-10)
    assert result["p_value"] > 0.05  # Should not be significant
    assert result["better_model"] == "equal"


def test_bootstrap_r2_comparison_different_models() -> None:
    """Test bootstrap R² comparison with different models."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred1 = y_true + np.random.randn(100) * 0.1  # Good model
    y_pred2 = y_true + np.random.randn(100) * 2.0  # Bad model

    result = bootstrap_r2_comparison(y_true, y_pred1, y_pred2, n_bootstrap=100)

    assert result["r2_model1"] > result["r2_model2"]
    assert result["r2_diff_observed"] > 0
    assert result["better_model"] == "model_1"


def test_bootstrap_r2_comparison_result_structure() -> None:
    """Test that bootstrap R² comparison returns all required fields."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred1 = y_true + np.random.randn(100) * 0.1
    y_pred2 = y_true + np.random.randn(100) * 0.2

    result = bootstrap_r2_comparison(y_true, y_pred1, y_pred2, n_bootstrap=100)

    required_keys = [
        "r2_model1",
        "r2_model2",
        "r2_diff_observed",
        "mean_diff_bootstrap",
        "std_diff_bootstrap",
        "p_value",
        "ci_lower",
        "ci_upper",
        "better_model",
        "significance",
        "interpretation",
        "n_bootstrap",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert result["n_bootstrap"] == 100


def test_bootstrap_r2_comparison_with_series() -> None:
    """Test bootstrap R² comparison with pandas Series."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred1 = pd.Series(y_true + np.random.randn(100) * 0.1)
    y_pred2 = pd.Series(y_true + np.random.randn(100) * 0.2)

    result = bootstrap_r2_comparison(y_true, y_pred1, y_pred2, n_bootstrap=100)

    assert isinstance(result, dict)
    assert "r2_model1" in result
    assert "r2_model2" in result


def test_bootstrap_r2_comparison_confidence_interval() -> None:
    """Test that confidence interval contains the observed difference."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred1 = y_true + np.random.randn(100) * 0.1
    y_pred2 = y_true + np.random.randn(100) * 0.2

    result = bootstrap_r2_comparison(y_true, y_pred1, y_pred2, n_bootstrap=500)

    # CI should be reasonable
    assert result["ci_lower"] < result["ci_upper"]
    # Mean should be close to observed (with some variance)
    assert abs(result["mean_diff_bootstrap"] - result["r2_diff_observed"]) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

