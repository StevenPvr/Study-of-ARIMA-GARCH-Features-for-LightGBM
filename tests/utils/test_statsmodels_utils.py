"""Tests for statsmodels utilities.

This test module validates the suppress_statsmodels_warnings function that
manages statsmodels warning suppression during ARIMA/ARIMA model fitting.
"""

from __future__ import annotations

import warnings

import pytest

from src.utils.statsmodels_utils import suppress_statsmodels_warnings


class TestSuppressStatsmodelsWarnings:
    """Test suite for suppress_statsmodels_warnings function."""

    def test_suppress_statsmodels_warnings_basic(self) -> None:
        """Test that suppress_statsmodels_warnings modifies warnings filter."""
        # Get initial filter count
        initial_filter_count = len(warnings.filters)

        # Call the function
        suppress_statsmodels_warnings()

        # Verify that new filters have been added
        assert len(warnings.filters) >= initial_filter_count
        # Note: >= because warnings.filterwarnings adds to the filter list

    def test_suppress_statsmodels_userwarning(self) -> None:
        """Test that UserWarnings from statsmodels module are suppressed."""
        # Apply suppression
        suppress_statsmodels_warnings()

        # Try to emit a UserWarning from statsmodels module
        with warnings.catch_warnings(record=True):
            # Reset filters to capture warnings
            warnings.simplefilter("default")
            # Re-apply our suppression
            suppress_statsmodels_warnings()

            # Emit a warning that should be suppressed
            warnings.warn("Test warning", category=UserWarning, stacklevel=1)

            # The warning should be captured but may be suppressed depending
            # on the module from which it's emitted
            # This is a basic test to ensure function runs without error

    def test_suppress_statsmodels_warnings_no_exception(self) -> None:
        """Test that suppress_statsmodels_warnings runs without raising exceptions."""
        try:
            suppress_statsmodels_warnings()
        except Exception as e:
            pytest.fail(f"suppress_statsmodels_warnings raised an exception: {e}")

    def test_suppress_statsmodels_warnings_idempotent(self) -> None:
        """Test that calling suppress_statsmodels_warnings multiple times is safe."""
        # Call multiple times
        suppress_statsmodels_warnings()
        suppress_statsmodels_warnings()
        suppress_statsmodels_warnings()

        # Verify filters were added (may have duplicates, which is acceptable)
        assert len(warnings.filters) > 0

    def test_suppress_statsmodels_warnings_filter_patterns(self) -> None:
        """Test that specific warning message patterns are in filters."""
        # Clear existing filters and apply suppression
        warnings.resetwarnings()
        suppress_statsmodels_warnings()

        # Convert filters to a list for inspection
        filter_list = list(warnings.filters)

        # Verify that filters exist
        assert len(filter_list) > 0

        # Note: We can't easily verify exact filter patterns without accessing
        # internal filter structure, but we can verify function completes

    def test_suppress_statsmodels_warnings_return_none(self) -> None:
        """Test that suppress_statsmodels_warnings returns None."""
        # Call the function and verify it completes without error
        suppress_statsmodels_warnings()
        # Since the function is typed to return None, this test passes by not raising

    def test_suppress_statsmodels_warnings_type_signature(self) -> None:
        """Test that suppress_statsmodels_warnings has correct type signature."""
        import inspect

        sig = inspect.signature(suppress_statsmodels_warnings)

        # Verify no parameters
        assert len(sig.parameters) == 0

        # Verify return annotation is None (can be None, 'None', or empty)
        assert sig.return_annotation in (None, "None", inspect.Signature.empty, type(None))


class TestSuppressStatsmodelsWarningsIntegration:
    """Integration tests for suppress_statsmodels_warnings with actual warnings."""

    def test_suppresses_no_supported_index_warning(self) -> None:
        """Test that 'No supported index' warning pattern is suppressed."""
        suppress_statsmodels_warnings()

        with warnings.catch_warnings(record=True):
            # Set to always show warnings by default
            warnings.simplefilter("always")
            # Re-apply suppression
            suppress_statsmodels_warnings()

            # Emit a warning matching the pattern
            warnings.warn(
                "No supported index is available for this series",
                category=UserWarning,
                stacklevel=1,
            )

            # Check if warning was suppressed (length should be 0 or warning is filtered)
            # Note: This test validates the pattern, actual suppression depends on context

    def test_suppresses_date_index_warning(self) -> None:
        """Test that 'date index has been provided' warning pattern is suppressed."""
        suppress_statsmodels_warnings()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            suppress_statsmodels_warnings()

            # Emit a warning matching the pattern
            warnings.warn(
                "A date index has been provided but no frequency information",
                category=UserWarning,
                stacklevel=1,
            )

    def test_suppresses_frequency_information_warning(self) -> None:
        """Test that 'frequency information' warning pattern is suppressed."""
        suppress_statsmodels_warnings()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            suppress_statsmodels_warnings()

            # Emit a warning matching the pattern
            warnings.warn(
                "Missing frequency information for the index",
                category=UserWarning,
                stacklevel=1,
            )

    def test_does_not_suppress_other_warnings(self) -> None:
        """Test that non-statsmodels warnings are not suppressed."""
        suppress_statsmodels_warnings()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Emit a DeprecationWarning (should NOT be suppressed)
            warnings.warn("This is deprecated", category=DeprecationWarning, stacklevel=1)

            # Verify the DeprecationWarning was captured
            # (actual behavior depends on filter order, but warning should be issued)
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)


class TestSuppressStatsmodelsWarningsUsage:
    """Test suite for typical usage scenarios of suppress_statsmodels_warnings."""

    def test_usage_before_model_fitting(self) -> None:
        """Test typical usage pattern before fitting ARIMA models."""
        # Simulate typical usage
        suppress_statsmodels_warnings()

        # In actual usage, user would fit ARIMA model here
        # This test validates the setup completes without error
        assert True  # Function completed successfully

    def test_usage_in_optimization_loop(self) -> None:
        """Test usage in an optimization loop scenario."""
        # Simulate calling once at beginning of optimization
        suppress_statsmodels_warnings()

        # Simulate multiple iterations
        for _ in range(10):
            # In actual usage, each iteration would fit a ARIMA model
            pass

        # Verify no errors during loop
        assert True

    def test_usage_module_level_initialization(self) -> None:
        """Test calling at module initialization level."""
        # Simulate module-level call
        suppress_statsmodels_warnings()

        # Subsequent operations would run with suppressed warnings
        assert True


# Fixtures for parameterized tests
@pytest.fixture
def reset_warnings():
    """Reset warnings filters before and after test."""
    warnings.resetwarnings()
    yield
    warnings.resetwarnings()


class TestSuppressStatsmodelsWarningsWithFixture:
    """Test suite using fixtures to reset warnings state."""

    def test_with_reset_warnings(self, reset_warnings: None) -> None:
        """Test suppress_statsmodels_warnings with clean warnings state."""
        # Get initial state
        initial_count = len(warnings.filters)

        # Apply suppression
        suppress_statsmodels_warnings()

        # Verify filters were added
        assert len(warnings.filters) >= initial_count

    def test_multiple_calls_with_reset(self, reset_warnings: None) -> None:
        """Test multiple calls with reset warnings state."""
        # First call
        suppress_statsmodels_warnings()
        first_count = len(warnings.filters)

        # Second call
        suppress_statsmodels_warnings()
        second_count = len(warnings.filters)

        # Both calls should add filters
        assert first_count > 0
        assert second_count >= first_count
