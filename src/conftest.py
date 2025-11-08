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
