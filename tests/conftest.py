"""Pytest configuration for src tests.

This file runs BEFORE any test imports, allowing us to mock
dependencies before they are imported by the modules under test.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock dependencies BEFORE any module imports them
# This must happen at the very top, before pytest collects tests
sys.modules["yfinance"] = MagicMock()
# Use real scientific stack (pandas, scipy, statsmodels) for numeric tests.
# Set matplotlib to non-interactive backend for tests (no GUI required)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

# Ensure lightgbm is imported early to avoid import order issues
# Note: tests/lightgbm/__init__.py should not exist to avoid shadowing the package
try:
    import lightgbm as lgb  # noqa: F401

    # Verify critical attributes exist
    if not hasattr(lgb, "Dataset"):
        raise AttributeError("lightgbm.Dataset not found - lightgbm may not be properly installed")
    if not hasattr(lgb, "LGBMRegressor"):
        raise AttributeError(
            "lightgbm.LGBMRegressor not found - lightgbm may not be properly installed"
        )
except ImportError as e:
    # If lightgbm is not available, tests will fail with a clear error
    raise ImportError(
        "lightgbm is required for tests but not installed. " "Install it with: pip install lightgbm"
    ) from e
except AttributeError as e:
    # If lightgbm is imported but missing attributes, it might be shadowed
    raise ImportError(
        f"lightgbm import issue: {e}. "
        "This may be caused by a local 'lightgbm' module shadowing the package. "
        "Ensure tests/lightgbm/__init__.py is removed if not needed."
    ) from e
