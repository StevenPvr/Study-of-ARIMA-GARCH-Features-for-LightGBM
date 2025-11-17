"""Unit tests for the GARCH evaluation CLI helpers."""

from __future__ import annotations

import typing as t

import pytest

from src.constants import (
    GARCH_EVAL_FORECAST_MODE_DEFAULT,
    GARCH_EVAL_FORECAST_MODE_HYBRID,
    GARCH_EVAL_FORECAST_MODE_NO_REFIT,
)
from src.garch.garch_eval import main as garch_eval_main
from src.garch.training_garch import orchestration


def test_parse_args_uses_no_refit_by_default() -> None:
    """The CLI defaults to the conservative no-refit mode."""

    args = garch_eval_main.parse_args([])

    assert args.forecast_mode == GARCH_EVAL_FORECAST_MODE_DEFAULT


def test_parse_args_accepts_hybrid_mode() -> None:
    """Users can explicitly select the hybrid mode via the flag."""

    args = garch_eval_main.parse_args([
        "--forecast-mode",
        GARCH_EVAL_FORECAST_MODE_HYBRID,
    ])

    assert args.forecast_mode == GARCH_EVAL_FORECAST_MODE_HYBRID


def test_resolve_forecast_generator_maps_modes_correctly() -> None:
    """Each forecast mode maps to the expected orchestration helper."""

    generator = garch_eval_main._resolve_forecast_generator(  # type: ignore[attr-defined]
        GARCH_EVAL_FORECAST_MODE_NO_REFIT
    )

    assert generator is orchestration.generate_full_sample_forecasts_from_trained_model


def test_resolve_forecast_generator_rejects_invalid_modes() -> None:
    """The resolver refuses unknown modes to avoid silent fallbacks."""

    with pytest.raises(ValueError):
        garch_eval_main._resolve_forecast_generator(  # type: ignore[attr-defined]
            t.cast(garch_eval_main.ForecastMode, "invalid")
        )
