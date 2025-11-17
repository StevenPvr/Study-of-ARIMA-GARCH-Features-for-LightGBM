"""Global main entry point to run all pipeline modules in sequence.

This module automatically discovers and executes all main.py files in the src/
directory in the correct order:
1. Data pipeline (fetching → cleaning → conversion → preparation)
2. ARIMA pipeline (
   data visualization → stationarity check → optimization → training → evaluation
)
3. GARCH pipeline (
   visualization → numerical tests → structure → params → diagnostic
   → training → eval
)
4. LightGBM pipeline (
   data preparation → correlation → optimization → training → evaluation
)

Note: The rolling GARCH forecasts are generated within training_garch module.
      The ablation study is performed within lightgbm/eval module.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

# Add project root to Python path for direct execution
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config_logging import setup_logging
from src.utils import get_logger

# Setup logging first
setup_logging()
logger = get_logger(__name__)


def find_all_main_files() -> list[Path]:
    """Find all main.py files and main scripts in the src directory.

    Returns:
        List of Path objects pointing to main.py files and main scripts,
        sorted by execution order.
    """
    src_dir = _script_dir
    main_files = sorted(src_dir.rglob("main.py"))
    # Exclude this file itself and any main.py in virtual environments or site-packages
    main_files = [
        f
        for f in main_files
        if f != Path(__file__)
        and "SP_Forecasting" not in str(f)
        and "site-packages" not in str(f)
        and "__pycache__" not in str(f)
    ]

    # Define execution order priority:
    # 1. Data pipeline
    # 2. ARIMA pipeline
    # 3. GARCH pipeline
    # 4. LightGBM pipeline
    def get_priority(path: Path) -> tuple[int, str]:
        """Get priority for sorting main files by execution order."""
        path_str = str(path)
        # Use more specific path matching to avoid ambiguities
        path_parts = path.parts

        # Data pipeline first (priority 1)
        # Check exact directory names in path parts
        if "data_fetching" in path_parts:
            return (1, "01_data_fetching")
        if "data_cleaning" in path_parts:
            return (1, "02_data_cleaning")
        if "data_conversion" in path_parts:
            return (1, "03_data_conversion")
        if "data_preparation" in path_parts and "lightgbm" not in path_parts:
            return (1, "04_data_preparation")
        # ARIMA pipeline second (priority 2)
        # Data visualization and stationarity check are part of ARIMA pipeline
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
        if "lightgbm" in path_parts and "data_leakage_checkup" in path_parts:
            return (4, "06_lgbm_data_leakage_checkup")
        if "lightgbm" in path_parts and "baseline" in path_parts:
            return (4, "07_lgbm_baseline")
        if "lightgbm" in path_parts and "permutation" in path_parts:
            return (4, "08_lgbm_permutation")
        # Unknown modules last (priority 99)
        return (99, path_str)

    # Sort by priority
    main_files.sort(key=get_priority)
    return main_files


if __name__ == "__main__":
    main_files = find_all_main_files()
    if not main_files:
        logger.error("No main.py files found in src/")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("S&P 500 FORECASTING - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info("Found %d main file(s) to execute:", len(main_files))
    for i, main_file in enumerate(main_files, 1):
        rel_path = main_file.relative_to(_project_root)
        logger.info("  [%d] %s", i, rel_path)

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTING PIPELINE")
    logger.info("=" * 80)

    failed_files = []
    for i, main_file in enumerate(main_files, 1):
        rel_path = main_file.relative_to(_project_root)
        logger.info("\n[%d/%d] Running %s...", i, len(main_files), rel_path)

        # Save original sys.argv to restore it after execution
        original_argv = sys.argv.copy()
        try:
            # Temporarily set sys.argv to only contain the module name
            # This allows argparse-based modules to use their default values
            sys.argv = [str(main_file)]
            runpy.run_path(str(main_file), run_name="__main__")
            logger.info("✓ %s completed", rel_path)
        except SystemExit as ex:
            # Some modules use argparse which calls sys.exit()
            # Check if it's a successful exit (code 0) or error
            if ex.code == 0 or ex.code is None:
                logger.info("✓ %s completed", rel_path)
            else:
                logger.error("✗ %s failed with exit code %s", rel_path, ex.code)
                failed_files.append(main_file)
        except KeyboardInterrupt:
            logger.error("\n" + "=" * 80)
            logger.error("✗✗✗ PIPELINE INTERRUPTED BY USER ✗✗✗")
            logger.error("=" * 80)
            sys.exit(130)  # Standard exit code for SIGINT
        except Exception as ex:
            # Check if this is a non-critical module (visualization or correlation)
            is_non_critical = (
                "visualisation" in str(main_file)
                or "visualization" in str(main_file)
                or "correlation" in str(main_file)
            )
            if is_non_critical:
                logger.warning("⚠ %s failed (non-critical): %s", rel_path, ex)
            else:
                logger.error("\n" + "=" * 80)
                logger.error("✗✗✗ PIPELINE FAILED AT %s ✗✗✗", rel_path)
                logger.error("Error: %s", ex)
                logger.error("=" * 80)
                raise
        finally:
            # Always restore original sys.argv
            sys.argv = original_argv

    logger.info("\n" + "=" * 80)
    if failed_files:
        logger.warning("⚠ PIPELINE COMPLETED WITH WARNINGS")
        logger.warning("Failed files (non-critical):")
        for f in failed_files:
            logger.warning("  - %s", f.relative_to(_project_root))
    else:
        logger.info("✓✓✓ ALL PIPELINES COMPLETED SUCCESSFULLY ✓✓✓")
    logger.info("=" * 80)
