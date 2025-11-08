"""Generate ML-ready dataset with rolling EGARCH forecasts on train+test."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.constants import PROJECT_ROOT
from src.garch.rolling_garch.rolling import build_ml_dataset
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Generate ML-ready dataset with rolling EGARCH volatility forecasts.
    
    Generates rolling EGARCH(1,1) forecasts on train+test with refit every 20 days.
    Output: results/garch_ml_dataset.csv
    """
    logger.info("=" * 60)
    logger.info("ROLLING GARCH - ML DATASET GENERATION (TRAIN + TEST)")
    logger.info("=" * 60)

    # Refit every 20 days for better stability
    refit_every = 20

    try:
        df_ml = build_ml_dataset(refit_every=refit_every)
    except (FileNotFoundError, ValueError) as ex:
        logger.error("Failed to build ML dataset: %s", ex)
        raise

    # Save dataset
    output_file = PROJECT_ROOT / "results" / "garch_ml_dataset.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_ml.to_csv(output_file, index=False)

    logger.info("Saved ML dataset: %s", output_file)
    logger.info("Dataset shape: %s", df_ml.shape)
    logger.info("Columns: %s", list(df_ml.columns))
    logger.info("Train samples: %d", (df_ml["split"] == "train").sum())
    logger.info("Test samples: %d", (df_ml["split"] == "test").sum())
    logger.info("Missing sigma2_forecast: %d", df_ml["sigma2_forecast"].isna().sum())
    logger.info("Refit frequency: %d days", refit_every)


if __name__ == "__main__":
    main()
