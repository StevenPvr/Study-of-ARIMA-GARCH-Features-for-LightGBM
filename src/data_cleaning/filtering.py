"""Dataset filtering functions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_cleaning.validation import assert_not_empty
from src.path import DATASET_FILTERED_FILE, DATASET_FILTERED_PARQUET_FILE
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_observations_per_ticker(raw_df: pd.DataFrame) -> pd.Series:
    """Calculate number of observations per ticker.

    **Important**: This function calculates statistics on the entire dataset.
    It must ONLY be used BEFORE train/test split to avoid data leakage.
    Never call this function on data that has already been split into train/test sets.

    Args:
        raw_df: Raw dataset DataFrame (must be unsplit, before train/test separation).

    Returns:
        Series indexed by ticker with observation counts.

    Raises:
        KeyError: If the ``tickers`` column is missing.
        ValueError: If the DataFrame is empty.
    """
    assert_not_empty(raw_df, name="raw dataset")

    if "tickers" not in raw_df.columns:
        msg = "Missing required column: 'tickers'"
        raise KeyError(msg)

    # keep Series for convenient indexing in the callers
    return raw_df.groupby("tickers").size()  # type: ignore[return-value]


def validate_filtering_inputs(raw_df: pd.DataFrame, valid_tickers: Iterable[str]) -> None:
    """Validate inputs for dataset filtering.

    Args:
        raw_df: Raw dataset DataFrame.
        valid_tickers: Iterable of valid ticker symbols.

    Raises:
        KeyError: If ``tickers`` column is missing.
        ValueError: If ``valid_tickers`` is empty or ``raw_df`` is empty.
    """
    assert_not_empty(raw_df, name="raw dataset")

    if "tickers" not in raw_df.columns:
        msg = "Missing required column: 'tickers'"
        raise KeyError(msg)

    valid_tickers = list(valid_tickers)
    if not valid_tickers:
        msg = "No valid tickers to filter. Dataset would be empty."
        raise ValueError(msg)


def write_filtered_dataset(filtered_df: pd.DataFrame, output_file: Path | None = None) -> Path:
    """Persist a filtered dataset to both CSV and Parquet formats.

    Args:
        filtered_df: Filtered dataset DataFrame.
        output_file: Optional explicit output file path. If not provided,
            ``DATASET_FILTERED_FILE`` is used. The parquet file will be saved
            with the same name but .parquet extension.

    Returns:
        Path to the written CSV file.

    Raises:
        OSError: If file cannot be written.
    """
    if output_file is None:
        output_file = DATASET_FILTERED_FILE

    from src.utils.io import ensure_output_dir

    ensure_output_dir(output_file)

    # Determine parquet file path
    parquet_file = output_file.with_suffix(".parquet")
    if output_file == DATASET_FILTERED_FILE:
        parquet_file = DATASET_FILTERED_PARQUET_FILE

    try:
        # Save CSV with explicit parameters to ensure proper formatting
        # Use QUOTE_MINIMAL to escape values containing commas or quotes
        # Use lineterminator='\n' for Unix-style line endings
        filtered_df.to_csv(
            output_file,
            index=False,
            sep=",",
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        logger.info("Saved filtered dataset to CSV: %s", output_file)

        # Save Parquet format
        filtered_df.to_parquet(parquet_file, index=False)
        logger.info("Saved filtered dataset to Parquet: %s", parquet_file)
    except OSError as e:
        msg = f"Failed to save filtered dataset to {output_file}: {e}"
        raise OSError(msg) from e

    return output_file


def save_filtered_dataset(raw_df: pd.DataFrame, valid_tickers: list[str]) -> pd.DataFrame:
    """Filter dataset and save to CSV and Parquet formats.

    This is a high-level convenience wrapper that validates inputs, performs
    ticker filtering, persists the result to disk (both CSV and Parquet), and
    logs a short summary.

    Args:
        raw_df: Raw dataset DataFrame.
        valid_tickers: List of valid ticker symbols.

    Returns:
        Filtered DataFrame.

    Raises:
        KeyError: If ``tickers`` column is missing.
        ValueError: If ``valid_tickers`` is empty or ``raw_df`` is empty.
        OSError: If file cannot be written.
    """
    validate_filtering_inputs(raw_df, valid_tickers)

    # Use bracket-based boolean indexing (not .loc) to ease testing with mocks
    filtered_df: pd.DataFrame = raw_df[raw_df["tickers"].isin(valid_tickers)].reset_index(drop=True)  # type: ignore[assignment]

    if filtered_df.empty:
        logger.warning("Filtered dataset is empty")

    write_filtered_dataset(filtered_df)

    logger.info("\nAfter filtering:")
    num_tickers = int(filtered_df["tickers"].nunique())  # type: ignore[attr-defined]
    logger.info("  Number of tickers: %d", num_tickers)
    logger.info("  Total observations: %s", f"{len(filtered_df):,}")

    return filtered_df
