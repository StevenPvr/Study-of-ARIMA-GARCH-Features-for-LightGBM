"""Tests for three-phase validation split.

Tests train/validation/test splitting for hyperparameter optimization.
"""

from __future__ import annotations

import sys
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pytest

from src.garch.garch_params.optimization.validation import ThreePhaseDataset, combine_train_validation, split_three_phase


class TestThreePhaseDataset:
    """Test suite for ThreePhaseDataset dataclass."""

    def test_dataset_initialization_valid(self) -> None:
        """Test dataset initialization with valid splits."""
        train = np.random.randn(600)
        validation = np.random.randn(200)
        test = np.random.randn(200)

        dataset = ThreePhaseDataset(
            train=train,
            validation=validation,
            test=test,
            train_indices=(0, 600),
            val_indices=(600, 800),
            test_indices=(800, 1000),
        )

        assert dataset.train.size == 600
        assert dataset.validation.size == 200
        assert dataset.test.size == 200

    def test_dataset_initialization_empty_train(self) -> None:
        """Test dataset initialization with empty train split raises ValueError."""
        train = np.array([])
        validation = np.random.randn(200)
        test = np.random.randn(200)

        with pytest.raises(ValueError, match="Training split is empty"):
            ThreePhaseDataset(
                train=train,
                validation=validation,
                test=test,
                train_indices=(0, 0),
                val_indices=(0, 200),
                test_indices=(200, 400),
            )

    def test_dataset_initialization_empty_validation(self) -> None:
        """Test dataset initialization with empty validation split raises ValueError."""
        train = np.random.randn(600)
        validation = np.array([])
        test = np.random.randn(200)

        with pytest.raises(ValueError, match="Validation split is empty"):
            ThreePhaseDataset(
                train=train,
                validation=validation,
                test=test,
                train_indices=(0, 600),
                val_indices=(600, 600),
                test_indices=(600, 800),
            )

    def test_dataset_initialization_empty_test(self) -> None:
        """Test dataset initialization with empty test split raises ValueError."""
        train = np.random.randn(600)
        validation = np.random.randn(200)
        test = np.array([])

        with pytest.raises(ValueError, match="Test split is empty"):
            ThreePhaseDataset(
                train=train,
                validation=validation,
                test=test,
                train_indices=(0, 600),
                val_indices=(600, 800),
                test_indices=(800, 800),
            )


class TestSplitThreePhase:
    """Test suite for three-phase splitting function."""

    def test_split_three_phase_basic(self) -> None:
        """Test basic three-phase splitting."""
        residuals = np.random.randn(1000)

        dataset = split_three_phase(residuals)

        # Check sizes (60/30/10 split)
        assert dataset.train.size == 600
        assert dataset.validation.size == 300
        assert dataset.test.size == 100

        # Check indices
        assert dataset.train_indices == (0, 600)
        assert dataset.val_indices == (600, 900)
        assert dataset.test_indices == (900, 1000)

        # Check temporal ordering (no overlap)
        assert np.array_equal(dataset.train, residuals[:600])
        assert np.array_equal(dataset.validation, residuals[600:900])
        assert np.array_equal(dataset.test, residuals[900:])

    def test_split_three_phase_exact_ratios(self) -> None:
        """Test that split ratios are respected."""
        residuals = np.random.randn(1000)
        dataset = split_three_phase(residuals)

        total_size = residuals.size
        train_size = dataset.train.size
        val_size = dataset.validation.size
        test_size = dataset.test.size

        # Check that all data is used
        assert train_size + val_size + test_size == total_size

        # Check approximate ratios (60/30/10)
        assert abs(train_size / total_size - 0.6) < 0.01
        assert abs(val_size / total_size - 0.3) < 0.05  # More tolerance for small splits
        assert abs(test_size / total_size - 0.1) < 0.05

    def test_split_three_phase_insufficient_data(self) -> None:
        """Test splitting with insufficient data raises ValueError."""
        # Too few observations for meaningful split (will result in empty validation)
        residuals = np.random.randn(2)

        with pytest.raises(ValueError, match="Insufficient residuals|Validation split is empty"):
            split_three_phase(residuals)

    def test_split_three_phase_temporal_order(self) -> None:
        """Test that temporal ordering is preserved."""
        # Create residuals with increasing values for easy verification
        residuals = np.arange(1000, dtype=float)
        dataset = split_three_phase(residuals)

        # Train should be first
        assert dataset.train[0] == 0
        assert dataset.train[-1] < dataset.validation[0]

        # Validation should be middle
        assert dataset.validation[0] == dataset.train[-1] + 1
        assert dataset.validation[-1] < dataset.test[0]

        # Test should be last
        assert dataset.test[0] == dataset.validation[-1] + 1
        assert dataset.test[-1] == 999

    def test_split_three_phase_no_data_leakage(self) -> None:
        """Test that there is no data leakage between splits."""
        residuals = np.arange(1000, dtype=float)
        dataset = split_three_phase(residuals)

        # Convert to sets for intersection check
        train_set = set(dataset.train.tolist())
        val_set = set(dataset.validation.tolist())
        test_set = set(dataset.test.tolist())

        # Check no overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        # Check all data is used
        assert len(train_set | val_set | test_set) == 1000


class TestCombineTrainValidation:
    """Test suite for combining train and validation sets."""

    def test_combine_train_validation_basic(self) -> None:
        """Test basic combination of train and validation."""
        train = np.array([1.0, 2.0, 3.0])
        validation = np.array([4.0, 5.0])
        test = np.array([6.0, 7.0])

        dataset = ThreePhaseDataset(
            train=train,
            validation=validation,
            test=test,
            train_indices=(0, 3),
            val_indices=(3, 5),
            test_indices=(5, 7),
        )

        combined = combine_train_validation(dataset)

        # Check size
        assert combined.size == 5

        # Check content (train + validation)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(combined, expected)

        # Check that test is not included
        assert 6.0 not in combined
        assert 7.0 not in combined

    def test_combine_train_validation_preserves_order(self) -> None:
        """Test that combination preserves temporal ordering."""
        residuals = np.arange(1000, dtype=float)
        dataset = split_three_phase(residuals)

        combined = combine_train_validation(dataset)

        # Check that combined is in order
        assert np.all(combined[:-1] <= combined[1:])

        # Check size
        expected_size = dataset.train.size + dataset.validation.size
        assert combined.size == expected_size

        # Check that last value of combined is last value of validation
        assert combined[-1] == dataset.validation[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])  # pragma: no cover
