"""Tests for training orchestration functions."""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


from src.lightgbm.training.training import run_training
from tests.lightgbm.training.conftest import _assert_results_file, _assert_training_results


def test_run_training(tmp_path: Path, mock_optimization_results: Path, mock_dataset: Path) -> None:
    """Test running parallel training for both models."""
    models_dir = tmp_path / "models"
    results_file = tmp_path / "training_results.json"

    results = run_training(
        optimization_results_path=mock_optimization_results,
        dataset_complete=mock_dataset,
        dataset_without_insights=mock_dataset,
        models_dir=models_dir,
        results_file=results_file,
    )

    assert "lightgbm_complete" in results
    assert "lightgbm_without_insights" in results

    for name in ["lightgbm_complete", "lightgbm_without_insights"]:
        _assert_training_results(results[name], name)

    _assert_results_file(results_file)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])  # pragma: no cover
