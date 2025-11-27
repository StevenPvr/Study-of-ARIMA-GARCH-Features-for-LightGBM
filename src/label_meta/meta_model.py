"""Meta-labeling pipeline built on triple-barrier primary labels.

The module performs the following steps:
- Load best triple-barrier parameters from the primary labeling stage
- Optionally optimize the meta-model hyperparameters (user prompt-friendly)
- Train the meta-model on events filtered with the primary triple-barrier label
- Evaluate the meta-model on the held-out test split

The implementation follows Lopez de Prado's meta-labeling approach by
training only on events where the primary triple-barrier label is non-zero
and learning a binary meta-label indicating whether the primary signal
is expected to be profitable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config_logging import get_logger
from src.constants import (
    DEFAULT_RANDOM_STATE,
    LABEL_META_EVALUATION_RESULTS_FILE,
    LABEL_META_MAX_ITER,
    LABEL_META_MODEL_FILE,
    LABEL_META_OPTIMIZATION_N_SPLITS,
    LABEL_META_OPTIMIZATION_N_TRIALS,
    LABEL_META_OPTIMIZATION_RESULTS_FILE,
    LABEL_META_RESULTS_DIR,
    LABEL_META_SPLIT_COLUMN,
    LABEL_META_TARGET_COLUMN,
    LABEL_META_TRAINING_RESULTS_FILE,
    LABEL_PRIMAIRE_BEST_PARAMS_FILE,
    LABEL_PRIMAIRE_LABELED_DATA_FILE,
    LABEL_PRIMAIRE_LABEL_COLUMN,
)
from src.utils.io import ensure_output_dir, load_json_data, read_dataset_file, save_json_pretty
from src.utils.validation import (
    has_both_splits,
    validate_dataframe_not_empty,
    validate_file_exists,
    validate_required_columns,
)

logger = get_logger(__name__)


def _prompt_optimize(meta_results_path: Path, mode: str) -> bool:
    """Determine whether to run hyperparameter optimization.

    Args:
        meta_results_path: Path to stored optimization results.
        mode: One of {"ask", "yes", "no"} controlling the decision flow.

    Returns:
        True if optimization should run.
    """

    if mode == "yes":
        return True
    if mode == "no":
        if not meta_results_path.exists():
            logger.warning(
                "Optimization results are missing but optimization was explicitly disabled."
            )
        return False

    # Interactive prompt mode
    if meta_results_path.exists():
        prompt = "Optimization results found. Re-run optimization? [y/N]: "
        default_response = "n"
    else:
        prompt = "No optimization results detected. Run optimization now? [Y/n]: "
        default_response = "y"

    response = input(prompt).strip().lower()
    if not response:
        response = default_response
    return response in {"y", "yes"}


def _load_primary_triple_barrier_params() -> dict[str, Any]:
    """Load best triple-barrier parameters from the primary labeling stage."""

    validate_file_exists(LABEL_PRIMAIRE_BEST_PARAMS_FILE, "Primary triple-barrier params")
    params = load_json_data(LABEL_PRIMAIRE_BEST_PARAMS_FILE)
    if not isinstance(params, dict) or not params:
        raise ValueError(
            "Primary triple-barrier parameters file is empty or not a valid mapping."
        )
    logger.info("Loaded primary triple-barrier parameters.")
    return params


def _load_labeled_events(label_col: str, split_col: str) -> pd.DataFrame:
    """Load the dataset enriched with primary triple-barrier labels."""

    validate_file_exists(LABEL_PRIMAIRE_LABELED_DATA_FILE, "Primary labeled dataset")
    df = read_dataset_file(LABEL_PRIMAIRE_LABELED_DATA_FILE)
    validate_dataframe_not_empty(df, "Primary labeled dataset")
    validate_required_columns(df, {label_col, split_col}, "Primary labeled dataset")

    if not has_both_splits(df, split_col):
        raise ValueError(
            f"Dataset must contain both train and test splits in column '{split_col}'."
        )
    return df


def _filter_meta_events(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Filter events using Lopez de Prado's non-zero label rule and create meta labels."""

    filtered = df[df[label_col] != 0].copy()
    validate_dataframe_not_empty(filtered, "Filtered meta events")

    filtered[LABEL_META_TARGET_COLUMN] = (
        filtered[label_col] > 0
    ).astype(int)
    return filtered


def _select_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """Select numeric feature columns excluding label-related metadata."""

    excluded = {label_col, LABEL_META_TARGET_COLUMN, LABEL_META_SPLIT_COLUMN}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feature_cols = [col for col in numeric_cols if col not in excluded]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for meta-model training.")
    return feature_cols


def _split_train_test(df: pd.DataFrame, split_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test partitions using the split column."""

    train_df = df[df[split_col] == "train"].copy()
    test_df = df[df[split_col] == "test"].copy()

    validate_dataframe_not_empty(train_df, "Train split")
    validate_dataframe_not_empty(test_df, "Test split")
    return train_df, test_df


def _build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """Build a logistic regression pipeline with scaling."""

    if params is None:
        params = {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}

    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")
    l1_ratio = params.get("l1_ratio")

    clf = LogisticRegression(
        penalty=penalty,
        C=params.get("C", 1.0),
        solver=solver,
        l1_ratio=l1_ratio,
        max_iter=LABEL_META_MAX_ITER,
        random_state=DEFAULT_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def _optimize_meta_model(X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict[str, Any]:
    """Run Optuna optimization for the meta-model."""

    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver = "saga" if penalty in {"l1", "elasticnet"} else "lbfgs"
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        params = {"penalty": penalty, "solver": solver, "C": C}
        if l1_ratio is not None:
            params["l1_ratio"] = l1_ratio

        pipeline = _build_pipeline(params)
        cv = StratifiedKFold(
            n_splits=LABEL_META_OPTIMIZATION_N_SPLITS,
            shuffle=True,
            random_state=DEFAULT_RANDOM_STATE,
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            error_score="raise",
            n_jobs=-1,
        )
        mean_score = scores.mean()
        if pd.isna(mean_score):
            raise TrialPruned("ROC-AUC evaluation returned NaN")
        return mean_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    logger.info("Meta-model optimization completed.")
    logger.info(f"Best ROC-AUC: {best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    return {"best_params": best_params, "best_score": best_value}


def _train_meta_model(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any] | None
) -> tuple[Pipeline, dict[str, float]]:
    """Train the meta-model and return training metrics."""

    pipeline = _build_pipeline(params)
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    train_proba = pipeline.predict_proba(X_train)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_train, train_pred)),
        "f1": float(f1_score(y_train, train_pred)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_train, train_proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    precision, recall, _, _ = precision_recall_fscore_support(
        y_train, train_pred, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)

    return pipeline, metrics


def _evaluate_meta_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate the trained meta-model on the test split."""

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probas))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)

    return metrics


def run_meta_labeling_pipeline(
    *, optimize_mode: str = "ask", n_trials: int = LABEL_META_OPTIMIZATION_N_TRIALS
) -> dict[str, Any]:
    """Run the full meta-labeling workflow."""

    primary_params = _load_primary_triple_barrier_params()
    events_df = _load_labeled_events(LABEL_PRIMAIRE_LABEL_COLUMN, LABEL_META_SPLIT_COLUMN)
    events_df = _filter_meta_events(events_df, LABEL_PRIMAIRE_LABEL_COLUMN)

    train_df, test_df = _split_train_test(events_df, LABEL_META_SPLIT_COLUMN)
    feature_columns = _select_feature_columns(train_df, LABEL_PRIMAIRE_LABEL_COLUMN)

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_META_TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_META_TARGET_COLUMN]

    ensure_output_dir(LABEL_META_RESULTS_DIR)

    should_optimize = _prompt_optimize(LABEL_META_OPTIMIZATION_RESULTS_FILE, optimize_mode)
    optimization_results: dict[str, Any] | None = None
    if should_optimize:
        optimization_results = _optimize_meta_model(X_train, y_train, n_trials)
        save_json_pretty(optimization_results, LABEL_META_OPTIMIZATION_RESULTS_FILE)
    elif LABEL_META_OPTIMIZATION_RESULTS_FILE.exists():
        optimization_results = load_json_data(LABEL_META_OPTIMIZATION_RESULTS_FILE)
        logger.info("Loaded existing meta-model optimization results.")

    best_params = optimization_results.get("best_params") if optimization_results else None

    model, train_metrics = _train_meta_model(X_train, y_train, best_params)
    ensure_output_dir(LABEL_META_MODEL_FILE)
    joblib.dump(model, LABEL_META_MODEL_FILE)
    save_json_pretty(train_metrics, LABEL_META_TRAINING_RESULTS_FILE)

    eval_metrics = _evaluate_meta_model(model, X_test, y_test)
    save_json_pretty(eval_metrics, LABEL_META_EVALUATION_RESULTS_FILE)

    logger.info("Meta-labeling pipeline completed successfully.")

    return {
        "primary_params": primary_params,
        "feature_columns": feature_columns,
        "train_metrics": train_metrics,
        "evaluation_metrics": eval_metrics,
        "optimization_results": optimization_results,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run meta-labeling pipeline using triple-barrier primary labels and optional Optuna tuning."
        )
    )
    parser.add_argument(
        "--optimize-meta",
        choices=["ask", "yes", "no"],
        default="ask",
        help="Control hyperparameter optimization (ask interactively by default).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=LABEL_META_OPTIMIZATION_N_TRIALS,
        help="Number of Optuna trials when optimization is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for meta-labeling."""

    args = parse_args()
    run_meta_labeling_pipeline(
        optimize_mode=args.optimize_meta,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
