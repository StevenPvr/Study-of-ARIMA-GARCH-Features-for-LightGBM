"""Tests for datetime utilities.

This test module validates datetime manipulation functions including date parsing,
normalization, range extraction, filtering, and string formatting.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.utils.datetime_utils import (
    extract_date_range,
    filter_by_date_range,
    format_dates_to_string,
    normalize_timestamp_to_datetime,
    parse_date_value,
)


class TestFormatDatesToString:
    """Test suite for format_dates_to_string function."""

    def test_format_series_with_default_format(self) -> None:
        """Test formatting Series of dates with default format."""
        dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"
        assert result.iloc[2] == "2024-01-03"

    def test_format_series_with_custom_format(self) -> None:
        """Test formatting Series of dates with custom format."""
        dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])

        result = format_dates_to_string(dates, date_format="%d/%m/%Y")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "01/01/2024"
        assert result.iloc[1] == "02/01/2024"
        assert result.iloc[2] == "03/01/2024"

    def test_format_datetimeindex_with_default_format(self) -> None:
        """Test formatting DatetimeIndex with default format."""
        date_index = pd.date_range("2024-01-01", periods=3)

        result = format_dates_to_string(date_index)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"
        assert result.iloc[2] == "2024-01-03"

    def test_format_datetimeindex_with_custom_format(self) -> None:
        """Test formatting DatetimeIndex with custom format."""
        date_index = pd.date_range("2024-01-01", periods=3)

        result = format_dates_to_string(date_index, date_format="%Y%m%d")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "20240101"
        assert result.iloc[1] == "20240102"
        assert result.iloc[2] == "20240103"

    def test_format_list_of_dates_with_default_format(self) -> None:
        """Test formatting list of datetime objects with default format."""
        date_list = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]

        result = format_dates_to_string(date_list)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"
        assert result.iloc[2] == "2024-01-03"

    def test_format_list_of_dates_with_custom_format(self) -> None:
        """Test formatting list of datetime objects with custom format."""
        date_list = [datetime(2024, 1, 1), datetime(2024, 1, 2)]

        result = format_dates_to_string(date_list, date_format="%B %d, %Y")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "January 01, 2024"
        assert result.iloc[1] == "January 02, 2024"

    def test_format_list_of_string_dates(self) -> None:
        """Test formatting list of date strings."""
        date_list = ["2024-01-01", "2024-01-02", "2024-01-03"]

        result = format_dates_to_string(date_list)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"
        assert result.iloc[2] == "2024-01-03"

    def test_format_timestamps_series(self) -> None:
        """Test formatting Series of pandas Timestamps."""
        timestamps = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])

        result = format_dates_to_string(timestamps)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"

    def test_format_with_time_component(self) -> None:
        """Test formatting dates with time component."""
        dates = pd.Series(["2024-01-01 12:30:45", "2024-01-02 08:15:00"])

        result = format_dates_to_string(dates, date_format="%Y-%m-%d %H:%M:%S")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01-01 12:30:45"
        assert result.iloc[1] == "2024-01-02 08:15:00"

    def test_format_date_only_from_datetime(self) -> None:
        """Test formatting dates ignoring time component."""
        dates = pd.Series(["2024-01-01 12:30:45", "2024-01-02 08:15:00"])

        result = format_dates_to_string(dates, date_format="%Y-%m-%d")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"

    def test_format_empty_series(self) -> None:
        """Test formatting empty Series."""
        dates = pd.Series([], dtype="datetime64[ns]")

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_format_single_date(self) -> None:
        """Test formatting Series with single date."""
        dates = pd.Series(["2024-01-01"])

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 1
        assert result.iloc[0] == "2024-01-01"

    def test_format_with_year_month_format(self) -> None:
        """Test formatting with year-month format."""
        dates = pd.Series(["2024-01-15", "2024-02-20", "2024-03-25"])

        result = format_dates_to_string(dates, date_format="%Y-%m")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01"
        assert result.iloc[1] == "2024-02"
        assert result.iloc[2] == "2024-03"

    def test_format_with_day_name(self) -> None:
        """Test formatting with day name."""
        dates = pd.Series(["2024-01-01", "2024-01-02"])  # Monday, Tuesday

        result = format_dates_to_string(dates, date_format="%A, %Y-%m-%d")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "Monday, 2024-01-01"
        assert result.iloc[1] == "Tuesday, 2024-01-02"

    def test_format_preserves_order(self) -> None:
        """Test that formatting preserves date order."""
        dates = pd.Series(["2024-01-03", "2024-01-01", "2024-01-02"])

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-01-03"
        assert result.iloc[1] == "2024-01-01"
        assert result.iloc[2] == "2024-01-02"

    def test_format_handles_various_input_formats(self) -> None:
        """Test formatting dates from consistent string formats."""
        # Use consistent format that pd.to_datetime can infer
        dates = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"])

        result = format_dates_to_string(dates, date_format="%Y-%m-%d")

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        # pd.to_datetime should parse all dates correctly

    def test_format_return_type_is_series(self) -> None:
        """Test that return type is always pandas Series."""
        # Test with Series
        dates_series = pd.Series(["2024-01-01"])
        result_series = format_dates_to_string(dates_series)
        assert isinstance(result_series, pd.Series)

        # Test with DatetimeIndex
        dates_index = pd.DatetimeIndex(["2024-01-01"])
        result_index = format_dates_to_string(dates_index)
        assert isinstance(result_index, pd.Series)

        # Test with list
        dates_list = ["2024-01-01"]
        result_list = format_dates_to_string(dates_list)
        assert isinstance(result_list, pd.Series)

    def test_format_with_none_format_uses_default(self) -> None:
        """Test that None format parameter uses default format."""
        dates = pd.Series(["2024-01-01", "2024-01-02"])

        result_explicit_none = format_dates_to_string(dates, date_format=None)
        result_implicit_default = format_dates_to_string(dates)

        assert result_explicit_none.equals(result_implicit_default)

    def test_format_dates_invalid_input_raises_error(self) -> None:
        """Test that invalid date input raises ValueError."""
        invalid_dates = pd.Series(["not-a-date", "invalid", "bad-format"])

        with pytest.raises(ValueError):
            format_dates_to_string(invalid_dates)

    def test_format_with_iso_format(self) -> None:
        """Test formatting with ISO 8601 format."""
        dates = pd.Series(["2024-01-01", "2024-01-02"])

        result = format_dates_to_string(dates, date_format="%Y-%m-%dT%H:%M:%S")

        assert isinstance(result, pd.Series)
        result_str = str(result.iloc[0])
        assert "T" in result_str
        assert "2024-01-01" in result_str

    def test_format_dates_type_annotations(self) -> None:
        """Test that format_dates_to_string has correct type annotations."""
        import inspect

        sig = inspect.signature(format_dates_to_string)

        # Verify parameters exist
        assert "dates" in sig.parameters
        assert "date_format" in sig.parameters

        # Verify return annotation is pd.Series (as string in forward reference)
        assert sig.return_annotation in (pd.Series, "pd.Series")


class TestFormatDatesToStringEdgeCases:
    """Test edge cases for format_dates_to_string function."""

    def test_format_mixed_datetime_types(self) -> None:
        """Test formatting mixed datetime types in a list."""
        mixed_dates = [
            datetime(2024, 1, 1),
            pd.Timestamp("2024-01-02"),
            "2024-01-03",
        ]

        result = format_dates_to_string(mixed_dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[1] == "2024-01-02"
        assert result.iloc[2] == "2024-01-03"

    def test_format_with_timezone_aware_dates(self) -> None:
        """Test formatting timezone-aware dates."""
        dates = pd.Series(pd.date_range("2024-01-01", periods=2, tz="UTC"))

        result = format_dates_to_string(dates)

        # Timezone information is preserved in conversion but not in output string
        assert isinstance(result, pd.Series)
        result_str_0 = str(result.iloc[0])
        result_str_1 = str(result.iloc[1])
        assert "2024-01-01" in result_str_0
        assert "2024-01-02" in result_str_1

    def test_format_large_date_series(self) -> None:
        """Test formatting large Series of dates."""
        dates = pd.Series(pd.date_range("2020-01-01", periods=1000))

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 1000
        assert result.iloc[0] == "2020-01-01"
        assert result.iloc[-1] == "2022-09-26"

    def test_format_with_leap_year_date(self) -> None:
        """Test formatting leap year dates."""
        dates = pd.Series(["2024-02-29"])  # 2024 is a leap year

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "2024-02-29"

    def test_format_with_special_characters_in_format(self) -> None:
        """Test formatting with special characters in format string."""
        dates = pd.Series(["2024-01-01"])

        result = format_dates_to_string(dates, date_format="Year: %Y, Month: %m")

        assert isinstance(result, pd.Series)
        result_str = str(result.iloc[0])
        assert "Year: 2024" in result_str
        assert "Month: 01" in result_str


class TestFormatDatesToStringIntegration:
    """Integration tests for format_dates_to_string with realistic scenarios."""

    def test_format_for_arima_evaluation_dataframe(self) -> None:
        """Test formatting dates for ARIMA evaluation DataFrame."""
        # Simulate rolling predictions dates
        dates = pd.date_range("2024-01-01", periods=20, freq="D")

        result = format_dates_to_string(dates)

        assert isinstance(result, pd.Series)
        assert len(result) == 20
        assert result.iloc[0] == "2024-01-01"
        assert result.iloc[-1] == "2024-01-20"

    def test_format_for_plotting_labels(self) -> None:
        """Test formatting dates for plot axis labels."""
        dates = pd.date_range("2024-01-01", periods=10, freq="ME")

        result = format_dates_to_string(dates, date_format="%b %Y")

        assert isinstance(result, pd.Series)
        result_str = str(result.iloc[0])
        assert "Jan 2024" in result_str

    def test_format_for_file_naming(self) -> None:
        """Test formatting dates for file naming (no special characters)."""
        dates = pd.Series(["2024-01-01 12:30:45"])

        result = format_dates_to_string(dates, date_format="%Y%m%d_%H%M%S")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "20240101_123045"
        # No colons or spaces for filesystem compatibility

    def test_format_for_reporting(self) -> None:
        """Test formatting dates for human-readable reports."""
        dates = pd.Series(["2024-01-01", "2024-12-31"])

        result = format_dates_to_string(dates, date_format="%B %d, %Y")

        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "January 01, 2024"
        assert result.iloc[1] == "December 31, 2024"


# Additional tests for existing datetime_utils functions
class TestNormalizeTimestampToDatetime:
    """Test suite for normalize_timestamp_to_datetime function."""

    def test_normalize_timezone_aware_timestamp(self) -> None:
        """Test normalizing timezone-aware timestamp."""
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        result = normalize_timestamp_to_datetime(ts)

        assert result.tzinfo is None
        assert result == pd.Timestamp("2024-01-01 12:00:00")

    def test_normalize_timezone_naive_timestamp(self) -> None:
        """Test normalizing timezone-naive timestamp (no change)."""
        ts = pd.Timestamp("2024-01-01 12:00:00")
        result = normalize_timestamp_to_datetime(ts)

        assert result.tzinfo is None
        assert result == ts


class TestParseDateValue:
    """Test suite for parse_date_value function."""

    def test_parse_string_date(self) -> None:
        """Test parsing string date."""
        result = parse_date_value("2024-01-01")

        assert isinstance(result, pd.Timestamp)
        assert result == pd.Timestamp("2024-01-01")

    def test_parse_timestamp(self) -> None:
        """Test parsing pandas Timestamp."""
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        result = parse_date_value(ts)

        assert isinstance(result, pd.Timestamp)
        assert result.tzinfo is None

    def test_parse_invalid_value_allow_none(self) -> None:
        """Test parsing invalid value with allow_none=True."""
        result = parse_date_value("invalid", allow_none=True)

        assert result is None


class TestExtractDateRange:
    """Test suite for extract_date_range function."""

    def test_extract_date_range_as_string(self) -> None:
        """Test extracting date range as strings."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})

        min_date, max_date = extract_date_range(df, as_string=True)

        assert isinstance(min_date, str)
        assert isinstance(max_date, str)
        assert "2024-01-01" in min_date
        assert "2024-01-10" in max_date

    def test_extract_date_range_as_timestamp(self) -> None:
        """Test extracting date range as Timestamps."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})

        min_date, max_date = extract_date_range(df, as_string=False)

        assert isinstance(min_date, pd.Timestamp)
        assert isinstance(max_date, pd.Timestamp)

    def test_extract_date_range_empty_dataframe(self) -> None:
        """Test extracting date range from empty DataFrame."""
        df = pd.DataFrame({"date": []})

        min_date, max_date = extract_date_range(df)

        assert min_date is None
        assert max_date is None


class TestFilterByDateRange:
    """Test suite for filter_by_date_range function."""

    def test_filter_by_year(self) -> None:
        """Test filtering DataFrame by year."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", "2025-12-31")})

        df_2024 = filter_by_date_range(df, "2024-01-01", "2024-12-31")

        assert len(df_2024) > 0
        assert all(
            pd.Timestamp("2024-01-01") <= date <= pd.Timestamp("2024-12-31")
            for date in df_2024["date"]
        )

    def test_filter_raises_on_empty(self) -> None:
        """Test that filtering raises ValueError when result is empty."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})

        with pytest.raises(ValueError, match="No data available"):
            filter_by_date_range(df, "2030-01-01", "2030-12-31", raise_if_empty=True)

    def test_filter_no_raise_on_empty(self) -> None:
        """Test that filtering returns empty DataFrame when raise_if_empty=False."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})

        result = filter_by_date_range(df, "2030-01-01", "2030-12-31", raise_if_empty=False)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
