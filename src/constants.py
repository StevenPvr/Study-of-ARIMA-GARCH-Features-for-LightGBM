"""Constants for the S&P 500 Forecasting project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_EVAL_DIR = RESULTS_DIR / "eval"

# Data file paths
SP500_TICKERS_FILE = DATA_DIR / "sp500_tickers.csv"
DATASET_FILE = DATA_DIR / "dataset.csv"
DATASET_FILTERED_FILE = DATA_DIR / "dataset_filtered.csv"
LIQUIDITY_WEIGHTS_FILE = DATA_DIR / "liquidity_weights.csv"
WEIGHTED_LOG_RETURNS_FILE = DATA_DIR / "weighted_log_returns.csv"
WEIGHTED_LOG_RETURNS_SPLIT_FILE = DATA_DIR / "weighted_log_returns_split.csv"
GARCH_VARIANCE_FILE = DATA_DIR / "garch_variance.csv"
FETCH_REPORT_FILE = RESULTS_DIR / "fetch_report.json"
DATA_QUALITY_REPORT_FILE = RESULTS_EVAL_DIR / "data_quality_report.json"

# Results file paths
SARIMA_BEST_MODELS_FILE = RESULTS_DIR / "sarima_best_models.json"
SARIMA_OPTIMIZATION_RESULTS_FILE = RESULTS_DIR / "sarima_optimization_results.csv"
ROLLING_PREDICTIONS_SARIMA_FILE = RESULTS_DIR / "rolling_predictions_sarima_000.csv"
ROLLING_VALIDATION_METRICS_SARIMA_FILE = RESULTS_DIR / "rolling_validation_metrics_sarima_000.json"
LJUNGBOX_RESIDUALS_SARIMA_FILE = RESULTS_DIR / "ljungbox_residuals_sarima_000.json"
SARIMA_LJUNGBOX_LAGS_DEFAULT: int = 20
SARIMA_RESIDUALS_LJUNGBOX_PLOT = PLOTS_DIR / "ljungbox_residuals_sarima_000.png"
SARIMA_DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (0, 0, 0, 12)
SARIMA_DEFAULT_SEASONAL_PERIOD: int = 12
# Rolling ARIMA refit defaults
# Refit every 20 days balances model adaptation with computational efficiency
# Forecasts remain at 1-day horizon (steps=1) for daily predictions
SARIMA_REFIT_EVERY_DEFAULT: int = 20
LJUNGBOX_SIGNIFICANCE_LEVEL: float = 0.05
DATE_FORMAT_DEFAULT: str = "%Y-%m-%d"
PLOT_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)
PLACEHOLDER_DATE_PREFIX: str = "date_"
SARIMA_TRAINED_MODEL_FILE = RESULTS_DIR / "models" / "sarima_trained_model.pkl"
SARIMA_TRAINED_MODEL_METADATA_FILE = RESULTS_DIR / "models" / "sarima_trained_model_metadata.json"
GARCH_DATASET_FILE = RESULTS_DIR / "dataset_garch.csv"
NAIVE_LAST_VALUE_METRICS_FILE = RESULTS_DIR / "naive_last_value_metrics.json"
ROLLING_PREDICTIONS_NAIVE_FILE = RESULTS_DIR / "rolling_predictions_naive_last_value.csv"
BACKTEST_RESULTS_FILE = RESULTS_DIR / "backtest_results.json"

# GARCH structure detection directory
GARCH_STRUCTURE_DIR = RESULTS_DIR / "garch" / "structure"
GARCH_DIAGNOSTICS_DIR = RESULTS_DIR / "garch" / "diagnostic"
GARCH_EVAL_DIR = RESULTS_DIR / "garch" / "eval"

# GARCH results (fixed filenames, no versioning)
GARCH_DIAGNOSTICS_FILE = GARCH_STRUCTURE_DIR / "garch_diagnostics.json"
GARCH_NUMERICAL_TESTS_FILE = GARCH_STRUCTURE_DIR / "garch_numerical_tests.json"
GARCH_ESTIMATION_FILE = GARCH_EVAL_DIR / "garch_estimation.json"
GARCH_LJUNGBOX_FILE = GARCH_DIAGNOSTICS_DIR / "garch_postcheck_ljungbox.json"
GARCH_DISTRIBUTION_DIAGNOSTICS_FILE = GARCH_DIAGNOSTICS_DIR / "garch_distribution_diagnostics.json"
GARCH_SUMMARY_FILE = RESULTS_DIR / "garch_summary.json"

# Plot directories
SARIMA_DATA_VISU_PLOTS_DIR = PLOTS_DIR / "data_visu_sarima"
GARCH_DATA_VISU_PLOTS_DIR = PLOTS_DIR / "data_visu_garch"
GARCH_STRUCTURE_PLOTS_DIR = PLOTS_DIR / "garch" / "structure"
GARCH_DIAGNOSTICS_PLOTS_DIR = PLOTS_DIR / "garch" / "diagnostics"

# GARCH plots (fixed filenames)
GARCH_STRUCTURE_PLOT = GARCH_STRUCTURE_PLOTS_DIR / "garch_structure.png"
GARCH_RESIDUALS_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "garch_residuals.png"
GARCH_STD_SQUARED_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "garch_std_squared_acf_pacf.png"
GARCH_STD_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "garch_std_acf_pacf.png"
GARCH_STD_QQ_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "garch_std_residuals_qq.png"
# GARCH data visualization plots (in data_visu_garch subdirectory)
GARCH_RETURNS_CLUSTERING_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "garch_returns_clustering.png"
GARCH_ACF_SQUARED_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "garch_acf_squared.png"
GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT = (
    GARCH_DATA_VISU_PLOTS_DIR / "garch_squared_residuals_acf_ljungbox.png"
)
GARCH_SQUARED_RESIDUALS_ANALYSIS_PLOT = (
    GARCH_DATA_VISU_PLOTS_DIR / "garch_squared_residuals_analysis.png"
)

# GARCH model artifacts
GARCH_MODEL_FILE = RESULTS_DIR / "models" / "garch_model.joblib"
GARCH_MODEL_METADATA_FILE = RESULTS_DIR / "models" / "garch_model.json"
GARCH_VARIANCE_OUTPUTS_FILE = RESULTS_DIR / "garch_variance.csv"
GARCH_FORECASTS_FILE = RESULTS_DIR / "garch_forecasts.csv"
GARCH_EVAL_METRICS_FILE = RESULTS_DIR / "garch_eval_metrics.json"
GARCH_ROLLING_FORECASTS_FILE = RESULTS_DIR / "garch_rolling_forecasts.csv"
GARCH_ROLLING_EVAL_FILE = RESULTS_DIR / "garch_rolling_eval.json"

# GARCH evaluation plots
GARCH_EVAL_VAR_TIMESERIES_PLOT = PLOTS_DIR / "garch_eval_var_timeseries.png"
GARCH_EVAL_VAR_SCATTER_PLOT = PLOTS_DIR / "garch_eval_var_scatter.png"
GARCH_EVAL_VAR_RESIDUALS_PLOT = PLOTS_DIR / "garch_eval_var_residuals.png"
# Template file name for VaR violations plots; will be formatted with alpha
GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE = "garch_eval_var_violations_alpha_{alpha}.png"

# Data fetching constants
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YFINANCE_HISTORY_YEARS = 12
# Fixed end date for reproducible dataset (12 years: 2013-01-01 to 2024-12-31)
DATA_FETCH_END_DATE = datetime(2024, 12, 31)
DATA_FETCH_START_DATE = datetime(2013, 1, 1)

# Data cleaning constants
MIN_VOLUME_THRESHOLD = 0  # Minimum volume threshold (exclusive)
TOP_N_TICKERS_REPORT = 10  # Number of top tickers to report in quality analysis
MONOTONICITY_CHECK_DEFAULT_DIFF = 1  # Default diff value for monotonicity check (seconds)

# Data conversion constants
LIQUIDITY_WEIGHTS_WINDOW_DEFAULT: int = 20  # Default trailing window (days) for liquidity weights

# Data preparation constants
TRAIN_RATIO_DEFAULT: float = 0.8  # Default train/test split ratio
TIMESERIES_SPLIT_N_SPLITS: int = 2  # Number of splits for TimeSeriesSplit (minimum required)
# Random Forest feature engineering defaults
RF_LAG_WINDOWS: tuple[int, ...] = (1, 5, 10, 20)
RF_LAG_FEATURE_COLUMNS: tuple[str, ...] = (
    "weighted_log_return",
    "weighted_closing",
    "weighted_open",
    "sigma2_garch",
    "sigma_garch",
    "std_resid_garch",
    "rsi_14",
    "sma_20",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_histogram",
)
# ARIMA-GARCH insight columns to exclude from baseline dataset
RF_ARIMA_GARCH_INSIGHT_COLUMNS: tuple[str, ...] = (
    "arima_pred_return",
    "arima_residual_return",
    "sigma2_garch",
    "sigma_garch",
    "std_resid_garch",
)

# Default random state for reproducibility
DEFAULT_RANDOM_STATE = 42

# Benchmark defaults
DEFAULT_INITIAL_CAPITAL: float = 10_000.0
DEFAULT_TRANSACTION_COST: float = 0.001
DEFAULT_SIGNAL_THRESHOLD: float = 0.001

# GARCH defaults and constraints
GARCH_ALPHA_BETA_SUM_MAX: float = 0.999
GARCH_MIN_OMEGA: float = 1e-12
GARCH_MIN_INIT_VAR: float = 1e-10
GARCH_DEFAULT_ALPHA: float = 0.05
GARCH_DEFAULT_BETA: float = 0.90
GARCH_STUDENT_NU_MIN: float = 2.01
GARCH_STUDENT_NU_MAX: float = 200.0
GARCH_STUDENT_NU_INIT: float = 8.0
GARCH_SKEWT_LAMBDA_MIN: float = -0.99
GARCH_SKEWT_LAMBDA_MAX: float = 0.99
GARCH_SKEWT_LAMBDA_INIT: float = -0.1

# GARCH diagnostics defaults
GARCH_LM_LAGS_DEFAULT: int = 12
GARCH_ACF_LAGS_DEFAULT: int = 20
GARCH_PLOT_Z_CONF: float = 1.96
GARCH_LJUNG_BOX_LAGS_DEFAULT: int = 20
GARCH_STD_EPSILON: float = 1e-12  # Small epsilon for numerical stability in standardization

# Rolling GARCH defaults
# Refit every 20 observations to balance adaptation and stability
# Keeps rolling backtests consistent with SARIMA refit cadence
GARCH_REFIT_EVERY_DEFAULT: int = 20
GARCH_REFIT_WINDOW_DEFAULT: str = "expanding"  # or "rolling"
GARCH_REFIT_WINDOW_SIZE_DEFAULT: int = 1000
# Maximum log-variance for numerical stability (exp(700) ~ 8.2e307)
GARCH_LOG_VAR_MAX: float = 700.0
GARCH_FIT_MIN_SIZE_KURTOSIS: int = 20  # Minimum size for kurtosis-based distribution selection
GARCH_FIT_MIN_SIZE: int = 100  # Minimum train size before starting rolling forecasts
GARCH_FIT_DEFAULT_PERSISTENCE: float = 0.95  # Default persistence (alpha + beta) for initial params
GARCH_FIT_FALLBACK_VAR: float = 1e-6  # Fallback variance when training data is empty

# GARCH calibration defaults
GARCH_CALIBRATION_EPS: float = 1e-12

# GARCH evaluation defaults
GARCH_EVAL_DEFAULT_LEVEL: float = 0.95  # Default prediction interval level
GARCH_EVAL_DEFAULT_HORIZON: int = 5  # Default forecast horizon
GARCH_EVAL_DEFAULT_ALPHAS: tuple[float, float] = (0.01, 0.05)  # Default VaR alpha levels
GARCH_EVAL_EPSILON: float = 1e-12  # Epsilon for numerical stability
GARCH_EVAL_MIN_ALPHA: float = 1e-6  # Minimum alpha value for quantiles
GARCH_EVAL_AIC_MULTIPLIER: float = 2.0  # AIC = 2k - 2*loglik
GARCH_EVAL_DEFAULT_SLOPE: float = 1.0  # Default MZ calibration slope
GARCH_EVAL_PLOT_LIMIT_MULTIPLIER: float = 1.05  # Plot limit multiplier
GARCH_EVAL_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)  # Default figure size
GARCH_EVAL_FIGURE_SIZE_SCATTER: tuple[int, int] = (5, 5)  # Scatter plot figure size
GARCH_EVAL_FIGURE_SIZE_RESIDUALS: tuple[int, int] = (10, 3)  # Residuals plot figure size
GARCH_EVAL_HALF: float = 0.5  # Half value for threshold comparisons

# Volatility backtest outputs
VOL_BACKTEST_FORECASTS_FILE = RESULTS_DIR / "vol_backtest_forecasts.csv"
VOL_BACKTEST_METRICS_FILE = RESULTS_DIR / "vol_backtest_metrics.json"
VOL_BACKTEST_PLOTS_DIR = PLOTS_DIR / "vol_backtest"
VOL_BACKTEST_VOLATILITY_PLOT = VOL_BACKTEST_PLOTS_DIR / "volatility_forecasts_comparison.png"

# Volatility baselines defaults
VOL_EWMA_LAMBDA_DEFAULT: float = 0.94
VOL_ROLLING_WINDOW_DEFAULT: int = 20
VOL_HAR_WEEK_WINDOW: int = 5
VOL_HAR_MONTH_WINDOW: int = 22
VOL_VAR_ALPHAS_DEFAULT = (0.01, 0.05)

# Stationarity check outputs
STATIONARITY_REPORT_FILE = RESULTS_EVAL_DIR / "stationarity_report.json"
# Stationarity analysis configuration
# Weekly resampling ensures tests are run on a smoother, calendar-aligned series
# For log-returns, weekly aggregation uses sum (log-additivity across time)
STATIONARITY_RESAMPLE_FREQ: str = "W"

# Random Forest dataset file paths
RF_DATASET_COMPLETE_FILE = DATA_DIR / "rf_dataset_complete.csv"
RF_DATASET_WITHOUT_INSIGHTS_FILE = DATA_DIR / "rf_dataset_without_insights.csv"
RF_DATASET_WITHOUT_SIGMA2_FILE = DATA_DIR / "rf_dataset_without_sigma2.csv"

# Random Forest plot directories
RF_CORRELATION_PLOTS_DIR = PLOTS_DIR / "correlation"

# Random Forest evaluation constants
RF_RESULTS_DIR = RESULTS_DIR / "random_forest"
RF_MODELS_DIR = RF_RESULTS_DIR / "models"
RF_EVAL_RESULTS_FILE = RF_RESULTS_DIR / "eval_results.json"
RF_SHAP_PLOTS_DIR = PLOTS_DIR / "random_forest" / "shap"
RF_SHAP_MAX_DISPLAY_DEFAULT: int = 20

# Random Forest optimization constants
RF_OPTIMIZATION_RESULTS_FILE = RF_RESULTS_DIR / "optimization_results.json"
RF_OPTIMIZATION_N_SPLITS: int = 5  # 5 folds for walk-forward validation with ~3000 rows
RF_OPTIMIZATION_N_TRIALS: int = 50  # Number of Optuna trials
