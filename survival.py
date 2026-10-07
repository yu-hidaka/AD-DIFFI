"""
Survival-analysis utilities for the AD-DIFFI benchmark.

The module supports:
- multivariable Cox proportional-hazards models;
- variance filtering;
- correlation filtering;
- top-k or full-feature analyses;
- univariate fallback when multivariable Cox fitting fails;
- held-out evaluation;
- repeated paired comparisons between Original DIFFI and AD-DIFFI.

This module does not download datasets and does not generate synthetic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================


@dataclass
class SurvivalConfig:
    """
    Configuration for repeated Cox survival analysis.
    """

    n_repeats: int = 20
    test_size: float = 0.30

    top_k: int = 6

    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    max_features: Optional[int] = 10

    penalizer: float = 0.1
    l1_ratio: float = 0.0

    time_col: str = "Survival Months"
    event_col: str = "Status_num"

    base_seed: int = 42


# ============================================================
# Data validation and preparation
# ============================================================


def validate_survival_columns(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_names: Sequence[str],
) -> None:
    """
    Validate survival and feature columns.
    """
    required = list(feature_names) + [
        time_col,
        event_col,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[time_col].isna().all():
        raise ValueError(
            f"All values are missing in time column: {time_col}"
        )

    if df[event_col].isna().all():
        raise ValueError(
            f"All values are missing in event column: {event_col}"
        )


def prepare_survival_dataframe(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """
    Convert survival and feature columns to numeric values.

    Missing rows are not removed here. Row filtering is performed within each
    repetition after the train/test split to avoid using evaluation data for
    training-time preprocessing.
    """
    validate_survival_columns(
        df=df,
        time_col=time_col,
        event_col=event_col,
        feature_names=feature_names,
    )

    data = df[
        list(feature_names)
        + [time_col, event_col]
    ].copy()

    for column in feature_names:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[time_col] = pd.to_numeric(
        data[time_col],
        errors="coerce",
    )

    data[event_col] = pd.to_numeric(
        data[event_col],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            time_col,
            event_col,
        ]
    ).reset_index(drop=True)

    data[event_col] = (
        data[event_col]
        .astype(int)
        .clip(0, 1)
    )

    data = data.loc[
        data[time_col] > 0
    ].reset_index(drop=True)

    if data[event_col].nunique() < 2:
        raise ValueError(
            "The survival event column must contain both event states."
        )

    return data


# ============================================================
# Training-only preprocessing
# ============================================================


def fit_variance_filter(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    threshold: float,
):
    """
    Fit a variance filter on training data only.
    """
    selector = VarianceThreshold(
        threshold=threshold,
    )

    selector.fit(
        X_train[list(feature_names)]
    )

    selected_indices = selector.get_support(
        indices=True
    )

    selected_features = [
        feature_names[index]
        for index in selected_indices
    ]

    return selector, selected_features


def fit_correlation_filter(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    threshold: float,
    max_features: Optional[int] = None,
):
    """
    Fit a correlation-based filter on training data only.

    For every pair with absolute correlation above threshold, the later column
    in the current feature order is removed. This deterministic rule is used
    only for preprocessing and does not use survival outcomes.
    """
    if len(feature_names) == 0:
        return []

    X = X_train[
        list(feature_names)
    ].copy()

    correlation = X.corr().abs()

    upper = correlation.where(
        np.triu(
            np.ones(
                correlation.shape,
                dtype=bool,
            ),
            k=1,
        )
    )

    remove = []

    for column in upper.columns:
        if any(
            upper[column] > threshold
        ):
            remove.append(column)

    selected = [
        feature
        for feature in feature_names
        if feature not in remove
    ]

    if max_features is not None:
        selected = selected[
            :max_features
        ]

    return selected


def fit_survival_preprocessor(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    variance_threshold: float,
    correlation_threshold: float,
    max_features: Optional[int],
):
    """
    Fit variance and correlation filters using training data only.
    """
    variance_selector, variance_features = (
        fit_variance_filter(
            X_train=X_train,
            feature_names=feature_names,
            threshold=variance_threshold,
        )
    )

    correlation_features = (
        fit_correlation_filter(
            X_train=X_train,
            feature_names=variance_features,
            threshold=correlation_threshold,
            max_features=max_features,
        )
    )

    if len(correlation_features) == 0:
        raise ValueError(
            "No features remained after variance and correlation filtering."
        )

    scaler = StandardScaler()

    scaler.fit(
        X_train[
            correlation_features
        ]
    )

    return {
        "variance_selector": variance_selector,
        "variance_features": variance_features,
        "selected_features": correlation_features,
        "scaler": scaler,
    }


def transform_survival_features(
    X: pd.DataFrame,
    preprocessing: Dict[str, Any],
) -> pd.DataFrame:
    """
    Transform features using a training-fitted survival preprocessor.
    """
    selected_features = preprocessing[
        "selected_features"
    ]

    scaler = preprocessing[
        "scaler"
    ]

    X_selected = X[
        selected_features
    ].copy()

    X_scaled = scaler.transform(
        X_selected
    )

    return pd.DataFrame(
        X_scaled,
        columns=selected_features,
        index=X.index,
    )


# ============================================================
# Cox model fitting
# ============================================================


def fit_multivariable_cox(
    X_train: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    feature_names: Sequence[str],
    penalizer: float,
    l1_ratio: float,
):
    """
    Fit a multivariable Cox model.
    """
    if len(feature_names) == 0:
        raise ValueError(
            "No features available for Cox model."
        )

    model_data = X_train[
        list(feature_names)
    ].copy()

    model_data["__duration__"] = (
        durations.to_numpy()
    )

    model_data["__event__"] = (
        events.to_numpy()
        .astype(int)
    )

    model = CoxPHFitter(
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )

    model.fit(
        model_data,
        duration_col="__duration__",
        event_col="__event__",
    )

    return model


def fit_univariate_cox(
    X_train: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    feature_name: str,
    penalizer: float,
    l1_ratio: float,
):
    """
    Fit a univariate Cox fallback model.
    """
    model_data = X_train[
        [feature_name]
    ].copy()

    model_data["__duration__"] = (
        durations.to_numpy()
    )

    model_data["__event__"] = (
        events.to_numpy()
        .astype(int)
    )

    model = CoxPHFitter(
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )

    model.fit(
        model_data,
        duration_col="__duration__",
        event_col="__event__",
    )

    return model


def calculate_c_index(
    model: CoxPHFitter,
    X_eval: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
) -> float:
    """
    Calculate C-index using Cox partial hazards.

    lifelines' concordance_index assumes that larger predicted scores represent
    longer survival. Cox partial hazard is a risk score, so its negative is
    passed to concordance_index.
    """
    risk = (
        model
        .predict_partial_hazard(
            X_eval
        )
        .to_numpy()
        .reshape(-1)
    )

    return float(
        concordance_index(
            durations.to_numpy(),
            -risk,
            events.to_numpy().astype(int),
        )
    )


def fit_cox_with_fallback(
    X_train: pd.DataFrame,
    durations_train: pd.Series,
    events_train: pd.Series,
    feature_names: Sequence[str],
    penalizer: float,
    l1_ratio: float,
):
    """
    Fit a multivariable Cox model and use the highest-ranked feature as a
    univariate fallback if multivariable fitting fails.
    """
    multivariable_error = None

    try:
        model = fit_multivariable_cox(
            X_train=X_train,
            durations=durations_train,
            events=events_train,
            feature_names=feature_names,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
        )

        return {
            "model": model,
            "model_type": "multivariable",
            "selected_features": list(
                feature_names
            ),
            "error": None,
        }

    except Exception as exc:
        multivariable_error = exc

    if len(feature_names) == 0:
        return {
            "model": None,
            "model_type": "failed",
            "selected_features": [],
            "error": str(
                multivariable_error
            ),
        }

    fallback_feature = feature_names[0]
    fallback_error = None

    try:
        model = fit_univariate_cox(
            X_train=X_train,
            durations=durations_train,
            events=events_train,
            feature_name=fallback_feature,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
        )

        return {
            "model": model,
            "model_type": "univariate_fallback",
            "selected_features": [
                fallback_feature
            ],
            "error": str(
                multivariable_error
            ),
        }

    except Exception as exc:
        fallback_error = exc

    return {
        "model": None,
        "model_type": "failed",
        "selected_features": [],
        "error": (
            f"Multivariable error: "
            f"{multivariable_error}; "
            f"Fallback error: "
            f"{fallback_error}"
        ),
    }


# ============================================================
# One survival-analysis evaluation
# ============================================================


def evaluate_survival_feature_set(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_names: Sequence[str],
    time_col: str,
    event_col: str,
    variance_threshold: float,
    correlation_threshold: float,
    max_features: Optional[int],
    penalizer: float,
    l1_ratio: float,
):
    """
    Fit a training-only preprocessing pipeline and evaluate a Cox model on
    held-out survival data.
    """
    train_data = train_data.dropna(
        subset=[
            time_col,
            event_col,
        ]
    ).copy()

    test_data = test_data.dropna(
        subset=[
            time_col,
            event_col,
        ]
    ).copy()

    if len(train_data) < 20:
        return {
            "c_index": 0.5,
            "model_type": "unavailable",
            "selected_features": [],
            "n_train": len(train_data),
            "n_test": len(test_data),
            "error": "Fewer than 20 training observations.",
        }

    if train_data[event_col].nunique() < 2:
        return {
            "c_index": 0.5,
            "model_type": "unavailable",
            "selected_features": [],
            "n_train": len(train_data),
            "n_test": len(test_data),
            "error": "Training data contain one event state.",
        }

    X_train = train_data[
        list(feature_names)
    ].astype(float)

    X_test = test_data[
        list(feature_names)
    ].astype(float)

    preprocessing = fit_survival_preprocessor(
        X_train=X_train,
        feature_names=feature_names,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        max_features=max_features,
    )

    X_train_processed = (
        transform_survival_features(
            X=X_train,
            preprocessing=preprocessing,
        )
    )

    X_test_processed = (
        transform_survival_features(
            X=X_test,
            preprocessing=preprocessing,
        )
    )

    model_result = fit_cox_with_fallback(
        X_train=X_train_processed,
        durations=train_data[time_col],
        events=train_data[event_col],
        feature_names=preprocessing[
            "selected_features"
        ],
        penalizer=penalizer,
        l1_ratio=l1_ratio,
    )

    if model_result["model"] is None:
        return {
            "c_index": 0.5,
            "model_type": model_result[
                "model_type"
            ],
            "selected_features": model_result[
                "selected_features"
            ],
            "n_train": len(train_data),
            "n_test": len(test_data),
            "error": model_result["error"],
        }

    eval_features = model_result[
        "selected_features"
    ]

    c_index = calculate_c_index(
        model=model_result["model"],
        X_eval=X_test_processed[
            eval_features
        ],
        durations=test_data[time_col],
        events=test_data[event_col],
    )

    return {
        "c_index": c_index,
        "model_type": model_result[
            "model_type"
        ],
        "selected_features": eval_features,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "error": model_result["error"],
        "variance_features": preprocessing[
            "variance_features"
        ],
        "preprocessed_features": preprocessing[
            "selected_features"
        ],
    }


# ============================================================
# Repeated Cox analysis
# ============================================================


def run_survival_repetition(
    df: pd.DataFrame,
    y_event: np.ndarray,
    feature_names: List[str],
    original_scores_func: Any,
    ad_scores_func: Any,
    config: SurvivalConfig,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    original_top_k: Optional[int] = None,
    ad_top_k: Optional[int] = None,
):
    """
    Run one paired survival-analysis repetition.

    Feature rankings must be supplied by functions that calculate importance
    using training data only.
    """
    time_col = time_col or config.time_col
    event_col = event_col or config.event_col

    seed = config.base_seed + 1000 + y_event.size

    train_indices, test_indices = (
        train_test_split(
            np.arange(len(df)),
            test_size=config.test_size,
            random_state=seed,
            stratify=y_event,
        )
    )

    train_data = df.iloc[
        train_indices
    ].copy()

    test_data = df.iloc[
        test_indices
    ].copy()

    original_scores = np.asarray(
        original_scores_func(
            train_data,
            feature_names,
            seed,
        ),
        dtype=float,
    )

    ad_scores = np.asarray(
        ad_scores_func(
            train_data,
            feature_names,
            seed,
        ),
        dtype=float,
    )

    original_order = np.argsort(
        -original_scores,
        kind="stable",
    )

    ad_order = np.argsort(
        -ad_scores,
        kind="stable",
    )

    k_original = (
        original_top_k
        or config.top_k
    )

    k_ad = (
        ad_top_k
        or config.top_k
    )

    original_features = [
        feature_names[index]
        for index in original_order[:k_original]
    ]

    ad_features = [
        feature_names[index]
        for index in ad_order[:k_ad]
    ]

    original_result = evaluate_survival_feature_set(
        train_data=train_data,
        test_data=test_data,
        feature_names=original_features,
        time_col=time_col,
        event_col=event_col,
        variance_threshold=config.variance_threshold,
        correlation_threshold=config.correlation_threshold,
        max_features=config.max_features,
        penalizer=config.penalizer,
        l1_ratio=config.l1_ratio,
    )

    ad_result = evaluate_survival_feature_set(
        train_data=train_data,
        test_data=test_data,
        feature_names=ad_features,
        time_col=time_col,
        event_col=event_col,
        variance_threshold=config.variance_threshold,
        correlation_threshold=config.correlation_threshold,
        max_features=config.max_features,
        penalizer=config.penalizer,
        l1_ratio=config.l1_ratio,
    )

    return {
        "repeat": seed,
        "original_features": original_features,
        "ad_features": ad_features,
        "original": original_result,
        "ad_diffi": ad_result,
        "paired_difference": (
            ad_result["c_index"]
            - original_result["c_index"]
        ),
    }


# ============================================================
# Full-feature and top-k evaluation
# ============================================================


def evaluate_full_feature_survival(
    df: pd.DataFrame,
    feature_names: List[str],
    config: SurvivalConfig,
):
    """
    Evaluate a full-feature Cox model using repeated held-out splits.

    This function does not use DIFFI or AD-DIFFI rankings. It is a descriptive
    full-feature reference analysis.
    """
    records = []

    event = pd.to_numeric(
        df[config.event_col],
        errors="coerce",
    ).fillna(0).astype(int).to_numpy()

    for repeat in range(config.n_repeats):
        seed = config.base_seed + repeat

        train_indices, test_indices = (
            train_test_split(
                np.arange(len(df)),
                test_size=config.test_size,
                random_state=seed,
                stratify=event,
            )
        )

        train_data = df.iloc[
            train_indices
        ].copy()

        test_data = df.iloc[
            test_indices
        ].copy()

        result = evaluate_survival_feature_set(
            train_data=train_data,
            test_data=test_data,
            feature_names=feature_names,
            time_col=config.time_col,
            event_col=config.event_col,
            variance_threshold=config.variance_threshold,
            correlation_threshold=config.correlation_threshold,
            max_features=config.max_features,
            penalizer=config.penalizer,
            l1_ratio=config.l1_ratio,
        )

        records.append(
            {
                "repeat": repeat,
                "c_index": result["c_index"],
                "model_type": result["model_type"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "n_features": len(
                    result["selected_features"]
                ),
                "error": result["error"],
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Summary functions
# ============================================================


def summarize_survival_results(
    repeat_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create mean, SD, MCSE, convergence, and fallback summaries.
    """
    records = []

    method_specs = [
        (
            "Original DIFFI",
            "original",
        ),
        (
            "AD-DIFFI",
            "ad_diffi",
        ),
    ]

    n_repeats = len(repeat_results)

    for method_name, key in method_specs:
        values = np.array(
            [
                item[key]["c_index"]
                for _, item in repeat_results.iterrows()
            ],
            dtype=float,
        )

        model_types = [
            item[key]["model_type"]
            for _, item in repeat_results.iterrows()
        ]

        records.append(
            {
                "method": method_name,
                "c_index_mean": float(
                    np.mean(values)
                ),
                "c_index_sd": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "c_index_mcse": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    / np.sqrt(n_repeats)
                ),
                "multivariable_rate": float(
                    np.mean(
                        np.array(
                            model_types
                        )
                        == "multivariable"
                    )
                ),
                "fallback_rate": float(
                    np.mean(
                        np.array(
                            model_types
                        )
                        == "univariate_fallback"
                    )
                ),
                "failed_rate": float(
                    np.mean(
                        np.array(
                            model_types
                        )
                        == "failed"
                    )
                ),
            }
        )

    paired_values = np.array(
        [
            item["paired_difference"]
            for _, item in repeat_results.iterrows()
        ],
        dtype=float,
    )

    records.append(
        {
            "method": "AD-DIFFI minus Original DIFFI",
            "c_index_mean": float(
                paired_values.mean()
            ),
            "c_index_sd": float(
                paired_values.std(ddof=1)
            ),
            "c_index_mcse": float(
                paired_values.std(ddof=1)
                / np.sqrt(n_repeats)
            ),
            "multivariable_rate": np.nan,
            "fallback_rate": np.nan,
            "failed_rate": np.nan,
        }
    )

    return pd.DataFrame(records)


# ============================================================
# Export helpers
# ============================================================


def serialize_survival_results(
    repeat_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert nested repeat-level results to a flat dataframe.
    """
    records = []

    for result in repeat_results:
        original = result["original"]
        ad = result["ad_diffi"]

        records.append(
            {
                "repeat": result["repeat"],
                "original_c_index": original[
                    "c_index"
                ],
                "ad_diffi_c_index": ad[
                    "c_index"
                ],
                "paired_difference": result[
                    "paired_difference"
                ],
                "original_model_type": original[
                    "model_type"
                ],
                "ad_diffi_model_type": ad[
                    "model_type"
                ],
                "original_n_features": len(
                    original[
                        "selected_features"
                    ]
                ),
                "ad_diffi_n_features": len(
                    ad[
                        "selected_features"
                    ]
                ),
                "original_error": original[
                    "error"
                ],
                "ad_diffi_error": ad[
                    "error"
                ],
            }
        )

    return pd.DataFrame(records)


def save_survival_results(
    repeat_results: List[Dict[str, Any]],
    summary: pd.DataFrame,
    output_dir: str | Path,
    dataset_name: str = "breast_cancer",
):
    """
    Save survival-analysis outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    repeat_df = serialize_survival_results(
        repeat_results
    )

    repeat_df.to_csv(
        output_dir
        / f"{dataset_name}_survival_repeats.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / f"{dataset_name}_survival_summary.csv",
        index=False,
    )

    metadata = {
        "dataset": dataset_name,
        "n_repeats": len(repeat_results),
    }

    with open(
        output_dir
        / f"{dataset_name}_survival_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )
