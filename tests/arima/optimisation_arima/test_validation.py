"""Tests for ARIMA optimization validation functions."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from src.arima.optimisation_arima.validation import compute_test_size_from_ratio


class TestComputeTestSizeFromRatio:
    """Test suite for compute_test_size_from_ratio function."""

    def test_compute_test_size_basic(self) -> None:
        """Test basic computation with standard inputs."""
        # 1000 observations, 20% test ratio, 5 splits
        # Expected: 1000 * 0.2 / 5 = 40
        result = compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=5)
        assert result == 40

    def test_compute_test_size_10_percent(self) -> None:
        """Test computation with 10% ratio."""
        # 1000 observations, 10% test ratio, 5 splits
        # Expected: 1000 * 0.1 / 5 = 20
        result = compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.1, n_splits=5)
        assert result == 20

    def test_compute_test_size_30_percent(self) -> None:
        """Test computation with 30% ratio."""
        # 500 observations, 30% test ratio, 3 splits
        # Expected: 500 * 0.3 / 3 = 50
        result = compute_test_size_from_ratio(train_len=500, test_size_ratio=0.3, n_splits=3)
        assert result == 50

    def test_compute_test_size_integer_division(self) -> None:
        """Test that integer division works correctly."""
        # 1000 observations, 20% test ratio, 4 splits
        # Expected: 1000 * 0.2 / 4 = 50
        result = compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=4)
        assert result == 50

    def test_compute_test_size_truncation(self) -> None:
        """Test that fractional results are truncated."""
        # 1000 observations, 20% test ratio, 7 splits
        # Expected: floor(1000 * 0.2 / 7) = floor(28.57) = 28
        result = compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=7)
        assert result == 28

    def test_compute_test_size_small_dataset(self) -> None:
        """Test with a small dataset."""
        # 100 observations, 20% test ratio, 2 splits
        # Expected: 100 * 0.2 / 2 = 10
        result = compute_test_size_from_ratio(train_len=100, test_size_ratio=0.2, n_splits=2)
        assert result == 10

    def test_compute_test_size_invalid_ratio_zero(self) -> None:
        """Test that ratio=0 raises ValueError."""
        with pytest.raises(ValueError, match="test_size_ratio must be in \\(0, 1\\)"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.0, n_splits=5)

    def test_compute_test_size_invalid_ratio_negative(self) -> None:
        """Test that negative ratio raises ValueError."""
        with pytest.raises(ValueError, match="test_size_ratio must be in \\(0, 1\\)"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=-0.1, n_splits=5)

    def test_compute_test_size_invalid_ratio_one(self) -> None:
        """Test that ratio=1 raises ValueError."""
        with pytest.raises(ValueError, match="test_size_ratio must be in \\(0, 1\\)"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=1.0, n_splits=5)

    def test_compute_test_size_invalid_ratio_greater_than_one(self) -> None:
        """Test that ratio>1 raises ValueError."""
        with pytest.raises(ValueError, match="test_size_ratio must be in \\(0, 1\\)"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=1.5, n_splits=5)

    def test_compute_test_size_invalid_n_splits_zero(self) -> None:
        """Test that n_splits=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=0)

    def test_compute_test_size_invalid_n_splits_negative(self) -> None:
        """Test that negative n_splits raises ValueError."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=-1)

    def test_compute_test_size_too_small_result(self) -> None:
        """Test that too small computed test_size raises ValueError."""
        # Very small dataset with large n_splits
        # 10 observations, 20% ratio, 10 splits -> test_size = 0
        with pytest.raises(ValueError, match="Computed test_size=0 is too small"):
            compute_test_size_from_ratio(train_len=10, test_size_ratio=0.2, n_splits=10)

    def test_compute_test_size_edge_case_minimum_valid(self) -> None:
        """Test edge case where result is exactly 1."""
        # 10 observations, 50% ratio, 5 splits
        # Expected: 10 * 0.5 / 5 = 1
        result = compute_test_size_from_ratio(train_len=10, test_size_ratio=0.5, n_splits=5)
        assert result == 1

    def test_compute_test_size_large_dataset(self) -> None:
        """Test with a large dataset."""
        # 10000 observations, 20% ratio, 5 splits
        # Expected: 10000 * 0.2 / 5 = 400
        result = compute_test_size_from_ratio(train_len=10000, test_size_ratio=0.2, n_splits=5)
        assert result == 400

    def test_compute_test_size_single_split(self) -> None:
        """Test with single split."""
        # 1000 observations, 20% ratio, 1 split
        # Expected: 1000 * 0.2 / 1 = 200
        result = compute_test_size_from_ratio(train_len=1000, test_size_ratio=0.2, n_splits=1)
        assert result == 200

    def test_compute_test_size_realistic_scenario(self) -> None:
        """Test realistic scenario matching typical S&P 500 data."""
        # ~5 years of daily data (252*5 = 1260), 20% ratio, 5 splits
        # Expected: 1260 * 0.2 / 5 = 50.4 -> 50
        result = compute_test_size_from_ratio(train_len=1260, test_size_ratio=0.2, n_splits=5)
        assert result == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
