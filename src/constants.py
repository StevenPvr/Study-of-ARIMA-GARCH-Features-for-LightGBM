"""Constants for the S&P 500 Forecasting project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# ============================================================================
# DATA PIPELINE - File paths
# ============================================================================

# Raw data files
SP500_TICKERS_FILE = DATA_DIR / "sp500_tickers.csv"
DATASET_FILE = DATA_DIR / "dataset.csv"
DATASET_FILTERED_FILE = DATA_DIR / "dataset_filtered.csv"
LIQUIDITY_WEIGHTS_FILE = DATA_DIR / "liquidity_weights.csv"
WEIGHTED_LOG_RETURNS_FILE = DATA_DIR / "weighted_log_returns.csv"
WEIGHTED_LOG_RETURNS_SPLIT_FILE = DATA_DIR / "weighted_log_returns_split.csv"

# Random Forest datasets
RF_DATASET_COMPLETE_FILE = DATA_DIR / "rf_dataset_complete.csv"
RF_DATASET_WITHOUT_INSIGHTS_FILE = DATA_DIR / "rf_dataset_without_insights.csv"
RF_DATASET_WITHOUT_SIGMA2_FILE = DATA_DIR / "rf_dataset_without_sigma2.csv"
# Deprecated alias retained for backward compatibility; prefer RF_DATASET_SIGMA_PLUS_BASE_FILE.
RF_DATASET_SIGMA2_ONLY_FILE = DATA_DIR / "rf_dataset_sigma2_only.csv"
RF_DATASET_SIGMA_PLUS_BASE_FILE = DATA_DIR / "rf_dataset_sigma_plus_base.csv"
RF_DATASET_RSI14_ONLY_FILE = DATA_DIR / "rf_dataset_rsi14_only.csv"
RF_DATASET_TECHNICAL_INDICATORS_FILE = DATA_DIR / "rf_dataset_technical_indicators.csv"

# ============================================================================
# RESULTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline results
DATA_RESULTS_DIR = RESULTS_DIR / "data"
FETCH_REPORT_FILE = DATA_RESULTS_DIR / "fetch_report.json"
DATA_QUALITY_REPORT_FILE = DATA_RESULTS_DIR / "data_quality_report.json"
STATIONARITY_REPORT_FILE = DATA_RESULTS_DIR / "stationarity_report.json"

# ARIMA/SARIMA results
ARIMA_RESULTS_DIR = RESULTS_DIR / "arima"
ARIMA_OPTIMIZATION_DIR = ARIMA_RESULTS_DIR / "optimization"
ARIMA_TRAINING_DIR = ARIMA_RESULTS_DIR / "training"
ARIMA_EVALUATION_DIR = ARIMA_RESULTS_DIR / "evaluation"
ARIMA_OUTPUTS_DIR = ARIMA_RESULTS_DIR / "outputs"

# ARIMA optimization files
SARIMA_BEST_MODELS_FILE = ARIMA_OPTIMIZATION_DIR / "sarima_best_models.json"
SARIMA_OPTIMIZATION_RESULTS_FILE = ARIMA_OPTIMIZATION_DIR / "optimization_results.csv"

# ARIMA training files
SARIMA_TRAINED_MODEL_FILE = ARIMA_TRAINING_DIR / "model.pkl"
SARIMA_TRAINED_MODEL_METADATA_FILE = ARIMA_TRAINING_DIR / "metadata.json"

# ARIMA evaluation files
ROLLING_PREDICTIONS_SARIMA_FILE = ARIMA_EVALUATION_DIR / "rolling_predictions.csv"
ROLLING_VALIDATION_METRICS_SARIMA_FILE = ARIMA_EVALUATION_DIR / "rolling_metrics.json"
LJUNGBOX_RESIDUALS_SARIMA_FILE = ARIMA_EVALUATION_DIR / "ljungbox_residuals.json"

# ARIMA outputs (for next pipeline step)
GARCH_DATASET_FILE = ARIMA_OUTPUTS_DIR / "dataset_garch.csv"

# GARCH results
GARCH_RESULTS_DIR = RESULTS_DIR / "garch"
GARCH_STRUCTURE_DIR = GARCH_RESULTS_DIR / "structure"
GARCH_ESTIMATION_DIR = GARCH_RESULTS_DIR / "estimation"
GARCH_TRAINING_DIR = GARCH_RESULTS_DIR / "training"
GARCH_DIAGNOSTIC_DIR = GARCH_RESULTS_DIR / "diagnostic"
GARCH_EVALUATION_DIR = GARCH_RESULTS_DIR / "evaluation"
GARCH_ROLLING_DIR = GARCH_RESULTS_DIR / "rolling"

# GARCH structure detection files
GARCH_DIAGNOSTICS_FILE = GARCH_STRUCTURE_DIR / "diagnostics.json"
GARCH_NUMERICAL_TESTS_FILE = GARCH_STRUCTURE_DIR / "numerical_tests.json"

# GARCH estimation files
GARCH_ESTIMATION_FILE = GARCH_ESTIMATION_DIR / "estimation.json"

# GARCH training files
GARCH_MODEL_FILE = GARCH_TRAINING_DIR / "model.joblib"
GARCH_MODEL_METADATA_FILE = GARCH_TRAINING_DIR / "model_metadata.json"
GARCH_VARIANCE_OUTPUTS_FILE = GARCH_TRAINING_DIR / "variance_outputs.csv"

# GARCH diagnostic files
GARCH_LJUNGBOX_FILE = GARCH_DIAGNOSTIC_DIR / "ljungbox.json"
GARCH_DISTRIBUTION_DIAGNOSTICS_FILE = GARCH_DIAGNOSTIC_DIR / "distribution_diagnostics.json"

# GARCH evaluation files
GARCH_FORECASTS_FILE = GARCH_EVALUATION_DIR / "forecasts.csv"
GARCH_EVAL_METRICS_FILE = GARCH_EVALUATION_DIR / "metrics.json"

# GARCH rolling files
GARCH_ROLLING_FORECASTS_FILE = GARCH_ROLLING_DIR / "forecasts.csv"
GARCH_ROLLING_EVAL_FILE = GARCH_ROLLING_DIR / "metrics.json"
GARCH_ROLLING_VARIANCE_FILE = GARCH_ROLLING_DIR / "variance.csv"
GARCH_ML_DATASET_FILE = GARCH_ROLLING_DIR / "ml_dataset.csv"

# Random Forest results
RF_RESULTS_DIR = RESULTS_DIR / "random_forest"
RF_OPTIMIZATION_DIR = RF_RESULTS_DIR / "optimization"
RF_TRAINING_DIR = RF_RESULTS_DIR / "training"
RF_MODELS_DIR = RF_RESULTS_DIR / "models"
RF_EVALUATION_DIR = RF_RESULTS_DIR / "evaluation"
RF_ABLATION_DIR = RF_RESULTS_DIR / "ablation"

# Random Forest optimization files
RF_OPTIMIZATION_RESULTS_FILE = RF_OPTIMIZATION_DIR / "results.json"

# Random Forest training files
RF_TRAINING_RESULTS_FILE = RF_TRAINING_DIR / "results.json"

# Random Forest evaluation files
RF_EVAL_RESULTS_FILE = RF_EVALUATION_DIR / "results.json"
RF_PERMUTATION_RESULTS_FILE = RF_EVALUATION_DIR / "permutation_importance.json"

# Random Forest ablation files
RF_ABLATION_SIGMA2_RESULTS_FILE = RF_ABLATION_DIR / "sigma2_results.json"

# Benchmark results
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmark"
VOL_BACKTEST_FORECASTS_FILE = BENCHMARK_RESULTS_DIR / "forecasts.csv"
VOL_BACKTEST_METRICS_FILE = BENCHMARK_RESULTS_DIR / "metrics.json"

# ============================================================================
# PLOTS DIRECTORIES - Organized by pipeline step/model
# ============================================================================

# Data pipeline plots
DATA_PLOTS_DIR = PLOTS_DIR / "data"
DATA_VISU_PLOTS_DIR = DATA_PLOTS_DIR / "visualization"

# ARIMA/SARIMA plots
ARIMA_PLOTS_DIR = PLOTS_DIR / "arima"
ARIMA_DATA_VISU_PLOTS_DIR = ARIMA_PLOTS_DIR / "data_visualization"
ARIMA_EVALUATION_PLOTS_DIR = ARIMA_PLOTS_DIR / "evaluation"

# ARIMA data visualization plots
SARIMA_DATA_VISU_PLOTS_DIR = ARIMA_DATA_VISU_PLOTS_DIR
SARIMA_RESIDUALS_LJUNGBOX_PLOT = ARIMA_EVALUATION_PLOTS_DIR / "ljungbox_residuals.png"

# GARCH plots
GARCH_PLOTS_DIR = PLOTS_DIR / "garch"
GARCH_DATA_VISU_PLOTS_DIR = GARCH_PLOTS_DIR / "data_visualization"
GARCH_STRUCTURE_PLOTS_DIR = GARCH_PLOTS_DIR / "structure"
GARCH_DIAGNOSTICS_PLOTS_DIR = GARCH_PLOTS_DIR / "diagnostics"
GARCH_EVALUATION_PLOTS_DIR = GARCH_PLOTS_DIR / "evaluation"

# GARCH structure plots
GARCH_STRUCTURE_PLOT = GARCH_STRUCTURE_PLOTS_DIR / "structure.png"

# GARCH diagnostic plots
GARCH_RESIDUALS_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "residuals.png"
GARCH_STD_SQUARED_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_squared_acf_pacf.png"
GARCH_STD_ACF_PACF_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_acf_pacf.png"
GARCH_STD_QQ_PLOT = GARCH_DIAGNOSTICS_PLOTS_DIR / "std_residuals_qq.png"

# GARCH data visualization plots
GARCH_RETURNS_CLUSTERING_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "returns_clustering.png"
GARCH_ACF_SQUARED_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "acf_squared.png"
GARCH_SQUARED_RESIDUALS_ACF_LB_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "squared_residuals_acf_ljungbox.png"
GARCH_SQUARED_RESIDUALS_ANALYSIS_PLOT = GARCH_DATA_VISU_PLOTS_DIR / "squared_residuals_analysis.png"

# GARCH evaluation plots
GARCH_EVAL_VAR_TIMESERIES_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_timeseries.png"
GARCH_EVAL_VAR_SCATTER_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_scatter.png"
GARCH_EVAL_VAR_RESIDUALS_PLOT = GARCH_EVALUATION_PLOTS_DIR / "var_residuals.png"
# Template path for VaR violations plots; will be formatted with alpha
GARCH_EVAL_VAR_VIOLATIONS_TEMPLATE = str(GARCH_EVALUATION_PLOTS_DIR / "var_violations_alpha_{alpha}.png")

# Random Forest plots
RF_PLOTS_DIR = PLOTS_DIR / "random_forest"
RF_CORRELATION_PLOTS_DIR = RF_PLOTS_DIR / "correlation"
RF_SHAP_PLOTS_DIR = RF_PLOTS_DIR / "shap"
RF_PERMUTATION_PLOTS_DIR = RF_PLOTS_DIR / "permutation"

# Benchmark plots
BENCHMARK_PLOTS_DIR = PLOTS_DIR / "benchmark"
VOL_BACKTEST_PLOTS_DIR = BENCHMARK_PLOTS_DIR / "volatility"
VOL_BACKTEST_VOLATILITY_PLOT = VOL_BACKTEST_PLOTS_DIR / "forecasts_comparison.png"

# ============================================================================
# MODEL PARAMETERS & DEFAULTS
# ============================================================================

# ARIMA/SARIMA defaults
SARIMA_DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (0, 0, 0, 12)
SARIMA_DEFAULT_SEASONAL_PERIOD: int = 12
SARIMA_REFIT_EVERY_DEFAULT: int = 20
SARIMA_BACKTEST_N_SPLITS_DEFAULT: int = 3
SARIMA_BACKTEST_TEST_SIZE_DEFAULT: int = 20
SARIMA_LJUNGBOX_LAGS_DEFAULT: int = 20
LJUNGBOX_SIGNIFICANCE_LEVEL: float = 0.05

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
GARCH_REFIT_EVERY_DEFAULT: int = 20
GARCH_REFIT_WINDOW_DEFAULT: str = "expanding"  # or "rolling"
GARCH_REFIT_WINDOW_SIZE_DEFAULT: int = 1000
GARCH_LOG_VAR_MAX: float = 700.0  # Maximum log-variance for numerical stability
GARCH_FIT_MIN_SIZE_KURTOSIS: int = 20
GARCH_FIT_MIN_SIZE: int = 100
GARCH_FIT_DEFAULT_PERSISTENCE: float = 0.95
GARCH_FIT_FALLBACK_VAR: float = 1e-6

# GARCH calibration defaults
GARCH_CALIBRATION_EPS: float = 1e-12

# GARCH evaluation defaults
GARCH_EVAL_DEFAULT_LEVEL: float = 0.95
GARCH_EVAL_DEFAULT_HORIZON: int = 5
GARCH_EVAL_DEFAULT_ALPHAS: tuple[float, float] = (0.01, 0.05)
GARCH_EVAL_EPSILON: float = 1e-12
GARCH_EVAL_MIN_ALPHA: float = 1e-6
GARCH_EVAL_AIC_MULTIPLIER: float = 2.0
GARCH_EVAL_DEFAULT_SLOPE: float = 1.0
GARCH_EVAL_PLOT_LIMIT_MULTIPLIER: float = 1.05
GARCH_EVAL_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)
GARCH_EVAL_FIGURE_SIZE_SCATTER: tuple[int, int] = (5, 5)
GARCH_EVAL_FIGURE_SIZE_RESIDUALS: tuple[int, int] = (10, 3)
GARCH_EVAL_HALF: float = 0.5

# Random Forest defaults
RF_LAG_WINDOWS: tuple[int, ...] = (1, 5, 10, 20)
RF_BASE_FEATURE_COLUMNS: tuple[str, ...] = ("rsi_14",)
RF_LAG_FEATURE_COLUMNS: tuple[str, ...] = (
    "weighted_log_return",
    "sigma2_garch",
    "sigma_garch",
    "std_resid_garch",
)
RF_ARIMA_GARCH_INSIGHT_COLUMNS: tuple[str, ...] = (
    "arima_pred_return",
    "arima_residual_return",
    "sigma2_garch",
    "sigma_garch",
    "std_resid_garch",
)
RF_SHAP_MAX_DISPLAY_DEFAULT: int = 20
RF_OPTIMIZATION_N_SPLITS: int = 5
RF_OPTIMIZATION_N_TRIALS: int = 50
RF_EVAL_RESAMPLE_FRACTION: float = 0.5
RF_EVAL_RESAMPLE_MIN_SIZE: int = 120
RF_EVAL_RESAMPLE_N_WINDOWS: int = 5
RF_TECHNICAL_SMA_WINDOW: int = 20
RF_TECHNICAL_EMA_SPAN: int = 20
RF_TECHNICAL_MACD_FAST: int = 12
RF_TECHNICAL_MACD_SLOW: int = 26
RF_TECHNICAL_MACD_SIGNAL: int = 9
RF_TECHNICAL_BB_WINDOW: int = 20
RF_TECHNICAL_BB_STD: float = 2.0
RF_TECHNICAL_ROC_PERIOD: int = 10
RF_TECHNICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    "sma_20",
    "ema_20",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "roc_10",
)

# Benchmark defaults
DEFAULT_INITIAL_CAPITAL: float = 10_000.0
DEFAULT_TRANSACTION_COST: float = 0.001
DEFAULT_SIGNAL_THRESHOLD: float = 0.001
VOL_EWMA_LAMBDA_DEFAULT: float = 0.94
VOL_ROLLING_WINDOW_DEFAULT: int = 20
VOL_HAR_WEEK_WINDOW: int = 5
VOL_HAR_MONTH_WINDOW: int = 22
VOL_VAR_ALPHAS_DEFAULT = (0.01, 0.05)

# ============================================================================
# DATA PIPELINE CONSTANTS
# ============================================================================

# Data fetching constants
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YFINANCE_HISTORY_YEARS = 12
DATA_FETCH_END_DATE = datetime(2024, 12, 31)
DATA_FETCH_START_DATE = datetime(2013, 1, 1)

# Data cleaning constants
MIN_VOLUME_THRESHOLD = 0
TOP_N_TICKERS_REPORT = 10
MONOTONICITY_CHECK_DEFAULT_DIFF = 1

# Data conversion constants
LIQUIDITY_WEIGHTS_WINDOW_DEFAULT: int = 20

# Data preparation constants
TRAIN_RATIO_DEFAULT: float = 0.8
TIMESERIES_SPLIT_N_SPLITS: int = 2
STATIONARITY_RESAMPLE_FREQ: str = "W"

# ============================================================================
# GENERAL CONSTANTS
# ============================================================================

DEFAULT_RANDOM_STATE = 42
DATE_FORMAT_DEFAULT: str = "%Y-%m-%d"
PLOT_FIGURE_SIZE_DEFAULT: tuple[int, int] = (10, 4)
PLACEHOLDER_DATE_PREFIX: str = "date_"
