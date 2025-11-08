"""Global tests for the S&P 500 Forecasting project.

This script runs all unit test files in the project.
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
    """Find all test_*.py files in the src directory.

    Returns:
        List of Path objects pointing to test files.
    """
    src_dir = _script_dir
    test_files = sorted(src_dir.rglob("test_*.py"))
    # Exclude this file itself to avoid recursion
    test_files = [f for f in test_files if f != Path(__file__)]
    test_files = [f for f in test_files if "test_e2e" not in str(f)]
    test_files = [f for f in test_files if "test_integration" not in str(f)]

    return test_files


if __name__ == "__main__":
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
