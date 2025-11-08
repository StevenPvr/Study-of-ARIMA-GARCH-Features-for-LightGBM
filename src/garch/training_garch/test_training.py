"""Tests for GARCH training utilities (numeric + IO with temp paths)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.garch.training_garch import training as trg
from src.garch.training_garch import utils as trg_utils
from src.garch.training_garch.training import (
    attach_outputs_to_dataframe,
    choose_best_fit,
    fit_egarch_candidates,
    train_egarch_from_dataset,
)
from src.garch.training_garch.utils import _build_variance_and_std_full


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


def test_build_variance_and_attach_outputs() -> None:
    train = _simulate_garch11(120, 0.02, 0.05, 0.9, seed=9)
    test = _simulate_garch11(80, 0.02, 0.05, 0.9, seed=10)
    params = {"omega": 0.02, "alpha": 0.05, "beta": 0.9}
    sigma2, z = _build_variance_and_std_full(np.concatenate([train, test]), params)
    assert sigma2.shape == z.shape == (train.size + test.size,)
    dates = pd.date_range("2020-01-01", periods=train.size + test.size, freq="D")
    split = ["train"] * train.size + ["test"] * test.size
    df = pd.DataFrame({"date": dates, "split": split})
    out = attach_outputs_to_dataframe(df, sigma2, z)
    assert {"sigma2_garch", "sigma_garch", "std_resid_garch"}.issubset(out.columns)


def _create_test_estimation_file(est_path: Path) -> None:
    """Create a test estimation file with normal and skewt distributions."""
    est = {
        "source": "dummy",
        "n_obs_train": 10,
        "n_obs_test": 5,
        "egarch_normal": {
            "omega": 1e-5,
            "alpha": 0.05,
            "beta": 0.9,
            "loglik": 1000.0,
            "converged": True,
        },
        "egarch_skewt": {
            "omega": 1.1e-5,
            "alpha": 0.05,
            "beta": 0.9,
            "loglik": 1005.0,  # Higher loglik -> better AIC (2*6 - 2*1005 = -1998 vs 2*4 - 2*1000 = -1992)
            "converged": True,
            "nu": 8.0,
            "lambda": -0.1,
        },
    }
    est_path.write_text(json.dumps(est))


def _patch_training_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path]:
    """Patch training module paths to use temporary directory.

    Returns:
        Tuple of (model_path, meta_path, var_path).
    """
    est_path = tmp_path / "garch_estimation.json"
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "model_meta.json"
    var_path = tmp_path / "variance.csv"

    monkeypatch.setattr(trg, "GARCH_ESTIMATION_FILE", est_path, raising=True)
    monkeypatch.setattr(trg_utils, "GARCH_ESTIMATION_FILE", est_path, raising=True)
    monkeypatch.setattr(trg_utils, "GARCH_MODEL_FILE", model_path, raising=True)
    monkeypatch.setattr(trg_utils, "GARCH_MODEL_METADATA_FILE", meta_path, raising=True)
    monkeypatch.setattr(trg_utils, "GARCH_VARIANCE_OUTPUTS_FILE", var_path, raising=True)

    return model_path, meta_path, var_path


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
            "arima_residual_return": resid,
        }
    )


def _validate_training_result(result: dict[str, Any], expected_dist: str = "skewt") -> None:
    """Validate training result structure and content."""
    assert result["dist"] == expected_dist
    assert "params" in result
    assert isinstance(result["params"], dict)

    diag = result.get("std_resid_diagnostics", {})
    assert {"n", "mean", "var", "std", "abs_gt_2", "abs_gt_3"}.issubset(diag.keys())


def _validate_output_files(model_path: Path, meta_path: Path, var_path: Path) -> None:
    """Validate that all output files were created."""
    assert model_path.exists()
    assert meta_path.exists()
    assert var_path.exists()


def test_train_uses_skewt_when_preestimated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that training uses skewt distribution when pre-estimated."""
    est_path = tmp_path / "garch_estimation.json"
    _create_test_estimation_file(est_path)

    model_path, meta_path, var_path = _patch_training_paths(monkeypatch, tmp_path)

    df = _create_test_dataframe()
    result = train_egarch_from_dataset(df)

    _validate_training_result(result, expected_dist="skewt")
    _validate_output_files(model_path, meta_path, var_path)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover


def test_choose_best_fit_tie_prefers_skewt() -> None:
    fits = {
        "egarch_normal": {
            "aic": 100.0,
            "converged": True,
            "omega": 1e-6,
            "alpha": 0.05,
            "beta": 0.9,
        },
        "egarch_skewt": {
            "aic": 100.0,
            "converged": True,
            "omega": 1.1e-6,
            "alpha": 0.05,
            "beta": 0.9,
            "nu": 8.0,
            "lambda": -0.1,
        },
    }
    key, _ = choose_best_fit(fits)  # type: ignore[arg-type]
    assert key == "egarch_skewt"


def test_fit_egarch_candidates_runs_on_synthetic() -> None:
    """Test that fit_egarch_candidates runs on synthetic data."""
    e = _simulate_garch11(200, 0.02, 0.05, 0.9, seed=42)
    fits = fit_egarch_candidates(e)
    assert isinstance(fits, dict)
    assert len(fits) >= 1
    # best selection returns a valid key
    key, best = choose_best_fit(fits)
    assert key in fits
    assert isinstance(best, dict)


def test_fit_egarch_candidates_empty_array() -> None:
    """Test that fit_egarch_candidates raises ValueError on empty array."""
    with pytest.raises(ValueError, match="empty"):
        fit_egarch_candidates(np.array([]))


def test_choose_best_fit_empty_dict() -> None:
    """Test that choose_best_fit raises ValueError on empty dict."""
    with pytest.raises(ValueError, match="No candidate fits"):
        choose_best_fit({})


def test_choose_best_fit_no_converged() -> None:
    """Test that choose_best_fit raises ValueError when no converged candidates."""
    fits = {
        "normal": {"aic": 100.0, "converged": False, "omega": 1e-6, "alpha": 0.05, "beta": 0.9},
    }
    with pytest.raises(ValueError, match="No converged"):
        choose_best_fit(fits)


def test_build_variance_empty_residuals() -> None:
    """Test that _build_variance_and_std_full raises ValueError on empty residuals."""
    params = {"omega": 0.02, "alpha": 0.05, "beta": 0.9}
    with pytest.raises(ValueError, match="empty"):
        _build_variance_and_std_full(np.array([]), params)


def test_build_variance_missing_params() -> None:
    """Test that _build_variance_and_std_full raises ValueError on missing params."""
    resid = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="Missing required"):
        _build_variance_and_std_full(resid, {"omega": 0.02})


def test_attach_outputs_length_mismatch() -> None:
    """Test that attach_outputs_to_dataframe raises ValueError on length mismatch."""
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5, freq="D")})
    sigma2 = np.array([1.0, 2.0])
    z = np.array([0.5, 1.0])
    with pytest.raises(ValueError, match="Length mismatch"):
        attach_outputs_to_dataframe(df, sigma2, z)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover