"""Random Forest correlation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checkers see the real imports
    from src.random_forest.correlation.correlation import (
        calculate_spearman_correlation,
        compute_correlations,
        load_dataset,
        plot_correlation_matrix,
    )
else:
    # Lazy imports to avoid importing matplotlib during test collection
    def __getattr__(name: str):
        if name in (
            "calculate_spearman_correlation",
            "compute_correlations",
            "load_dataset",
            "plot_correlation_matrix",
        ):
            from src.random_forest.correlation.correlation import (  # noqa: F401
                calculate_spearman_correlation,
                compute_correlations,
                load_dataset,
                plot_correlation_matrix,
            )

            return locals()[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "calculate_spearman_correlation",
    "compute_correlations",
    "load_dataset",
    "plot_correlation_matrix",
]
