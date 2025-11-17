"""Global script to run tests and main pipeline in sequence.

This script executes:
1. test_global.py (all unit tests)
2. test_utils_validation.py (utils validation tests)
3. main_global.py (complete pipeline execution)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.utils import get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def run_script(script_path: Path, script_name: str) -> bool:
    """Run a Python script and return True if successful, False otherwise.

    Args:
        script_path: Path to the script to run.
        script_name: Human-readable name for logging.

    Returns:
        True if script executed successfully, False otherwise.
    """
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False

    logger.info("=" * 80)
    logger.info("Running: %s", script_name)
    logger.info("=" * 80)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [str(script_path)]
        runpy.run_path(str(script_path), run_name="__main__")
        logger.info("✓ %s completed successfully", script_name)
        return True
    except SystemExit as ex:
        if ex.code == 0 or ex.code is None:
            logger.info("✓ %s completed successfully", script_name)
            return True
        logger.error("✗ %s failed with exit code %s", script_name, ex.code)
        return False
    except KeyboardInterrupt:
        logger.error("\n" + "=" * 80)
        logger.error("✗✗✗ INTERRUPTED BY USER ✗✗✗")
        logger.error("=" * 80)
        sys.exit(130)
    except Exception as ex:
        logger.error("\n" + "=" * 80)
        logger.error("✗✗✗ %s FAILED ✗✗✗", script_name)
        logger.error("Error: %s", ex)
        logger.error("=" * 80)
        return False
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("S&P 500 FORECASTING - COMPLETE EXECUTION")
    logger.info("=" * 80)

    # Define scripts to run in order
    scripts = [
        (_project_root / "tests" / "test_global.py", "test_global"),
        (_project_root / "tests" / "test_utils_validation.py", "test_utils_validation"),
        (_project_root / "src" / "main_global.py", "main_global"),
    ]

    failed_scripts = []
    for script_path, script_name in scripts:
        success = run_script(script_path, script_name)
        if not success:
            failed_scripts.append(script_name)
            # Stop execution if a script fails
            logger.error("\n" + "=" * 80)
            logger.error("✗✗✗ EXECUTION STOPPED ✗✗✗")
            logger.error("Failed at: %s", script_name)
            logger.error("=" * 80)
            sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("✓✓✓ ALL SCRIPTS COMPLETED SUCCESSFULLY ✓✓✓")
    logger.info("=" * 80)
    sys.exit(0)

