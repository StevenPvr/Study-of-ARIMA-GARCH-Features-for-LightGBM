"""Global tests for the S&P 500 Forecasting project.

This script runs all unit test files in the project in the same logical order
as the main pipeline execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
# This must be done before importing src modules
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import pytest


def find_all_test_files() -> list[Path]:
    """Find all test_*.py files in the tests directory, sorted by pipeline order.

    Returns:
        List of Path objects pointing to test files, sorted by execution order:
        1. Data pipeline tests
        2. ARIMA/ARIMA pipeline tests
        3. GARCH pipeline tests
        4. LightGBM pipeline tests
    """
    tests_dir = _script_dir
    test_files = sorted(tests_dir.rglob("test_*.py"))
    # Exclude this file itself to avoid recursion
    test_files = [f for f in test_files if f != Path(__file__)]
    test_files = [f for f in test_files if "test_e2e" not in str(f)]
    test_files = [f for f in test_files if "test_integration" not in str(f)]

    # Define execution order priority (same as main_global.py):
    # 1. Data pipeline
    # 2. ARIMA/ARIMA pipeline
    # 3. GARCH pipeline
    # 4. LightGBM pipeline
    def get_priority(path: Path) -> tuple[int, str]:
        """Get priority for sorting test files by pipeline order."""
        path_parts = path.parts

        # Data pipeline first (priority 1)
        if "data_fetching" in path_parts:
            return (1, "01_data_fetching")
        if "data_cleaning" in path_parts:
            return (1, "02_data_cleaning")
        if "data_conversion" in path_parts:
            return (1, "03_data_conversion")
        if "data_preparation" in path_parts and "lightgbm" not in path_parts:
            return (1, "04_data_preparation")
        # ARIMA pipeline second (priority 2)
        if "arima" in path_parts and "data_visualisation" in path_parts:
            return (2, "01_data_visualisation_arima")
        if "stationnarity_check" in path_parts:
            return (2, "02_stationnarity_check")
        if "optimisation_arima" in path_parts:
            return (2, "03_optimisation_arima")
        if "training_arima" in path_parts:
            return (2, "04_training_arima")
        if "evaluation_arima" in path_parts:
            return (2, "05_evaluation_arima")
        # GARCH pipeline third (priority 3)
        if "garch_data_visualisation" in path_parts:
            return (3, "01_garch_data_visualisation")
        if "garch_numerical_test" in path_parts:
            return (3, "02_garch_numerical_test")
        if "structure_garch" in path_parts:
            return (3, "03_structure_garch")
        if "garch_params" in path_parts:
            return (3, "04_garch_params")
        if "garch_diagnostic" in path_parts:
            return (3, "05_garch_diagnostic")
        if "training_garch" in path_parts:
            return (3, "06_training_garch")
        if "garch_eval" in path_parts:
            return (3, "07_garch_eval")
        if "rolling_garch" in path_parts:
            return (3, "08_rolling_garch")
        # LightGBM pipeline fourth (priority 4)
        if "lightgbm" in path_parts and "data_preparation" in path_parts:
            return (4, "01_lgbm_data_preparation")
        if "lightgbm" in path_parts and "correlation" in path_parts:
            return (4, "02_lgbm_correlation")
        if "lightgbm" in path_parts and "optimisation" in path_parts:
            return (4, "03_lgbm_optimisation")
        if "lightgbm" in path_parts and "training" in path_parts:
            return (4, "04_lgbm_training")
        if "lightgbm" in path_parts and "eval" in path_parts:
            return (4, "05_lgbm_eval")
        if "lightgbm" in path_parts and "permutation" in path_parts:
            return (4, "06_lgbm_permutation")
        if "lightgbm" in path_parts and "data_leakage_checkup" in path_parts:
            return (4, "07_lgbm_data_leakage_checkup")
        if "lightgbm" in path_parts and "baseline" in path_parts:
            return (4, "08_lgbm_baseline")
        # Unknown tests last (priority 99)
        return (99, str(path))

    # Sort by priority
    test_files.sort(key=get_priority)
    return test_files


if __name__ == "__main__":  # pragma: no cover - convenience runner
    pytest.main([__file__, "-q", "-x"])
    test_files = find_all_test_files()
    if not test_files:
        print("No test files found in src/")
        sys.exit(1)

    print(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file.relative_to(_project_root)}")

    # Convert Path objects to strings for pytest
    test_paths = [str(f) for f in test_files]
    # Add verbose flag and run all tests
    exit_code = pytest.main(["-v"] + test_paths)
    sys.exit(exit_code)
