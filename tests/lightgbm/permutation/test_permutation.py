"""Tests for block permutation importance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.lightgbm.permutation.permutation import compute_block_permutation_importance


def test_compute_block_permutation_importance_structure() -> None:
    """Basic structural test for permutation importance output."""
    rng = np.random.RandomState(0)
    n = 120
    X = pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        }
    )
    # Create a weak linear relation on x1, no relation on x2
    x1_array = X["x1"].to_numpy()
    y = pd.Series(0.1 * x1_array.astype(float) + 0.01 * rng.randn(n))

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X.iloc[:80], y.iloc[:80])

    out = compute_block_permutation_importance(
        model,
        X.iloc[80:],
        y.iloc[80:],
        block_size=10,
        n_repeats=5,
        random_state=42,
    )

    assert set(out.keys()) == {"x1", "x2"}
    for stats in out.values():
        assert "delta_r2_mean" in stats
        assert "delta_r2_std" in stats
        assert "delta_rmse_mean" in stats
        assert "delta_rmse_std" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
