from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .core import compute_enrichment_ad_diffi
from .utils import (
    train_test_split_mixed_type,
    validate_feature_metadata,
)


ORIGINAL_METHOD = (
    "Original DIFFI"
)

AD_DIFFI_METHOD = (
    "AD-DIFFI"
)

METHOD_ORDER = [
    ORIGINAL_METHOD,
    AD_DIFFI_METHOD,
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for repeated real-world benchmarking.
    """

    n_repeats: int = 20
    test_size: float = 0.30

    n_estimators: int = 100
    max_samples: int = 256
    max_features: float = 1.0
    bootstrap: bool = False

    model_contamination: Optional[
        float
    ] = None

    top_k: int = 6
    downstream_max_iter: int = 1000
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.n_repeats < 1:
            raise ValueError(
                "n_repeats must be at least 1."
            )

        if not (
            0.0 < self.test_size < 1.0
        ):
            raise ValueError(
                "test_size must be in (0, 1)."
            )

        if self.n_estimators < 1:
            raise ValueError(
                "n_estimators must be at least 1."
            )

        if self.max_samples < 2:
            raise ValueError(
                "max_samples must be at least 2."
            )

        if (
            self.model_contamination
            is not None
            and not (
                0.0
                < self.model_contamination
                < 0.5
            )
        ):
            raise ValueError(
                "model_contamination must be "
                "in (0, 0.5) or None."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if self.max_features <= 0:
            raise ValueError(
                "max_features must be positive."
            )


def resolve_contamination(
    y: np.ndarray,
    config: BenchmarkConfig,
) -> float:
    """
    Resolve the Isolation Forest contamination parameter.

    If no value is supplied, use the observed dataset-level anomaly rate.
    This value is computed once before repeated splitting and is fixed across
    repetitions.
    """
    if (
        config.model_contamination
        is not None
    ):
        return float(
            config.model_contamination
        )

    y = np.asarray(
        y,
        dtype=int,
    )

    contamination = float(
        y.mean()
    )

    if not (
        0.0
        < contamination
        < 0.5
    ):
        raise ValueError(
            "Observed contamination must be "
            "in (0, 0.5)."
        )

    return contamination


def make_if_params(
    config: BenchmarkConfig,
    n_train: int,
    contamination: float,
) -> Dict[str, object]:
    """
    Create Isolation Forest parameters.
    """
    if n_train < 2:
        raise ValueError(
            "n_train must be at least 2."
        )

    return {
        "n_estimators": config.n_estimators,
        "max_samples": min(
            config.max_samples,
            n_train,
        ),
        "contamination": contamination,
        "max_features": config.max_features,
        "bootstrap": config.bootstrap,
        "n_jobs": -1,
    }


def fit_isolation_forest(
    X_train: np.ndarray,
    config: BenchmarkConfig,
    contamination: float,
    random_state: int,
) -> IsolationForest:
    """
    Fit Isolation Forest on training data only.
    """
    X_train = np.asarray(
        X_train,
        dtype=float,
    )

    model = IsolationForest(
        **make_if_params(
            config=config,
            n_train=X_train.shape[0],
            contamination=contamination,
        ),
        random_state=random_state,
    )

    model.fit(
        X_train
    )

    return model


def get_model_anomaly_mask(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate model-defined outlier labels.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    return (
        model.predict(X) == -1
    )


def get_model_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return larger-is-more-anomalous scores.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    return -model.decision_function(
        X
    )


def _get_iic(
    estimator,
    predictions: np.ndarray,
    is_leaves: np.ndarray,
    adjust_iic: bool = True,
) -> np.ndarray:
    """
    Compute Original DIFFI node-level coefficients.
    """
    from math import ceil

    tree = estimator.tree_

    lambda_values = np.zeros(
        tree.node_count,
        dtype=float,
    )

    children_left = (
        tree.children_left
    )

    children_right = (
        tree.children_right
    )

    if predictions.shape[0] == 0:
        return lambda_values

    node_indicator = (
        estimator.decision_path(
            predictions
        ).toarray()
    )

    n_samples_node = np.sum(
        node_indicator,
        axis=0,
    )

    for node in range(
        tree.node_count
    ):
        n_current = (
            n_samples_node[node]
        )

        if (
            children_left[node]
            == children_right[node]
            or is_leaves[node]
        ):
            lambda_values[node] = -1.0
            continue

        if n_current <= 1:
            lambda_values[node] = -1.0
            continue

        n_left = n_samples_node[
            children_left[node]
        ]

        n_right = n_samples_node[
            children_right[node]
        ]

        if n_left == 0 or n_right == 0:
            lambda_values[node] = 0.0
            continue

        current_min = (
            0.5
            if n_current == 2
            else ceil(n_current / 2)
            / n_current
        )

        current_max = (
            n_current - 1
        ) / n_current

        split_imbalance = max(
            n_left,
            n_right,
        ) / n_current

        if (
            adjust_iic
            and current_min != current_max
        ):
            lambda_values[node] = (
                (
                    split_imbalance
                    - current_min
                )
                / (
                    current_max
                    - current_min
                )
                * 0.5
                + 0.5
            )
        else:
            lambda_values[node] = (
                split_imbalance
            )

    return lambda_values


def original_diffi_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    adjust_iic: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute Original DIFFI feature scores.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if not hasattr(
        iforest,
        "estimators_",
    ):
        raise ValueError(
            "iforest must be fitted."
        )

    n_features = X.shape[1]

    cfi_outliers = np.zeros(
        n_features,
        dtype=float,
    )

    cfi_inliers = np.zeros(
        n_features,
        dtype=float,
    )

    counter_outliers = np.zeros(
        n_features,
        dtype=int,
    )

    counter_inliers = np.zeros(
        n_features,
        dtype=int,
    )

    anomaly_scores = (
        iforest.decision_function(X)
    )

    for tree_index, estimator in enumerate(
        iforest.estimators_
    ):
        inbag_indices = list(
            iforest.estimators_samples_[
                tree_index
            ]
        )

        X_inbag = X[
            inbag_indices
        ]

        score_inbag = anomaly_scores[
            inbag_indices
        ]

        X_outliers = X_inbag[
            score_inbag < 0
        ]

        X_inliers = X_inbag[
            score_inbag >= 0
        ]

        if (
            len(X_outliers) == 0
            or len(X_inliers) == 0
        ):
            continue

        tree = estimator.tree_

        children_left = (
            tree.children_left
        )

        children_right = (
            tree.children_right
        )

        split_feature = tree.feature

        node_depth = np.zeros(
            tree.node_count,
            dtype=np.int64,
        )

        is_leaves = np.zeros(
            tree.node_count,
            dtype=bool,
        )

        stack = [(0, -1)]

        while stack:
            node_id, parent_depth = (
                stack.pop()
            )

            node_depth[node_id] = (
                parent_depth + 1
            )

            if (
                children_left[node_id]
                != children_right[node_id]
            ):
                stack.append(
                    (
                        children_left[node_id],
                        parent_depth + 1,
                    )
                )

                stack.append(
                    (
                        children_right[node_id],
                        parent_depth + 1,
                    )
                )
            else:
                is_leaves[node_id] = True

        def accumulate_cfi(
            X_subset: np.ndarray,
            cfi_array: np.ndarray,
            counter_array: np.ndarray,
        ) -> None:
            lambda_values = _get_iic(
                estimator=estimator,
                predictions=X_subset,
                is_leaves=is_leaves.copy(),
                adjust_iic=adjust_iic,
            )

            node_indicator = (
                estimator.decision_path(
                    X_subset
                ).toarray()
            )

            for row in range(
                len(X_subset)
            ):
                path = np.where(
                    node_indicator[row] == 1
                )[0]

                if len(path) == 0:
                    continue

                leaf_depth = node_depth[
                    path[-1]
                ]

                if leaf_depth == 0:
                    continue

                for node in path:
                    feature_index = (
                        split_feature[node]
                    )

                    lambda_value = (
                        lambda_values[node]
                    )

                    if (
                        feature_index >= 0
                        and lambda_value != -1
                    ):
                        cfi_array[
                            feature_index
                        ] += (
                            lambda_value
                            / leaf_depth
                        )

                        counter_array[
                            feature_index
                        ] += 1

        accumulate_cfi(
            X_subset=X_outliers,
            cfi_array=cfi_outliers,
            counter_array=counter_outliers,
        )

        accumulate_cfi(
            X_subset=X_inliers,
            cfi_array=cfi_inliers,
            counter_array=counter_inliers,
        )

    fi_out = np.divide(
        cfi_outliers,
        counter_outliers,
        out=np.zeros_like(
            cfi_outliers
        ),
        where=counter_outliers > 0,
    )

    fi_in = np.divide(
        cfi_inliers,
        counter_inliers,
        out=np.zeros_like(
            cfi_inliers
        ),
        where=counter_inliers > 0,
    )

    scores = np.divide(
        fi_out,
        fi_in,
        out=np.zeros_like(
            fi_out
        ),
        where=fi_in > 0,
    )

    return (
        scores,
        cfi_outliers,
        cfi_inliers,
    )


def compute_ad_diffi_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    feature_types: Dict[int, str],
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute raw enrichment-based AD-DIFFI scores using core.py.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    anomaly_mask = np.asarray(
        anomaly_mask,
        dtype=bool,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )

    if anomaly_mask.shape != (
        X.shape[0],
    ):
        raise ValueError(
            "anomaly_mask must have shape "
            "(n_samples,)."
        )

    if anomaly_mask.all() or (
        ~anomaly_mask
    ).all():
        raise ValueError(
            "anomaly_mask must contain both "
            "outliers and inliers."
        )

    scores = compute_enrichment_ad_diffi(
        iforest=iforest,
        X_data=X,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if scores.shape != (
        X.shape[1],
    ):
        raise ValueError(
            "Unexpected AD-DIFFI score shape: "
            f"{scores.shape}"
        )

    return scores


def safe_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            scores,
        )
    )


def safe_average_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if y_true.sum() == 0:
        return float("nan")

    return float(
        average_precision_score(
            y_true,
            scores,
        )
    )


def select_top_features(
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """
    Select top-k features in descending score order.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional."
        )

    top_k = min(
        top_k,
        len(scores),
    )

    ranking = np.argsort(
        -np.nan_to_num(
            scores,
            nan=-np.inf,
        ),
        kind="stable",
    )

    return ranking[
        :top_k
    ]


def get_feature_ranks(
    scores: np.ndarray,
) -> np.ndarray:
    """
    Return one-based ranks.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )

    order = np.argsort(
        -np.nan_to_num(
            scores,
            nan=-np.inf,
        ),
        kind="stable",
    )

    ranks = np.empty(
        len(scores),
        dtype=int,
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1,
    )

    return ranks


def fit_downstream_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_indices: np.ndarray,
    config: BenchmarkConfig,
    contamination: float,
    random_state: int,
) -> Dict[str, float]:
    """
    Fit downstream Isolation Forest and logistic regression.

    Feature selection has already been performed using training data only.
    """
    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if selected_indices.size == 0:
        raise ValueError(
            "selected_indices must not be empty."
        )

    X_train_selected = X_train[
        :,
        selected_indices,
    ]

    X_test_selected = X_test[
        :,
        selected_indices,
    ]

    downstream_if = IsolationForest(
        **make_if_params(
            config=config,
            n_train=X_train_selected.shape[0],
            contamination=contamination,
        ),
        random_state=random_state,
    )

    downstream_if.fit(
        X_train_selected
    )

    if_scores = (
        -downstream_if.decision_function(
            X_test_selected
        )
    )

    if_predictions = (
        downstream_if.predict(
            X_test_selected
        ) == -1
    ).astype(int)

    result = {
        "if_auc": safe_roc_auc(
            y_test,
            if_scores,
        ),
        "if_average_precision": (
            safe_average_precision(
                y_test,
                if_scores,
            )
        ),
        "if_precision": float(
            precision_score(
                y_test,
                if_predictions,
                zero_division=0,
            )
        ),
        "if_recall": float(
            recall_score(
                y_test,
                if_predictions,
                zero_division=0,
            )
        ),
        "if_f1": float(
            f1_score(
                y_test,
                if_predictions,
                zero_division=0,
            )
        ),
        "if_balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                if_predictions,
            )
        ),
    }

    logistic = LogisticRegression(
        max_iter=config.downstream_max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )

    logistic.fit(
        X_train_selected,
        y_train,
    )

    logistic_scores = (
        logistic.predict_proba(
            X_test_selected
        )[:, 1]
    )

    logistic_predictions = (
        logistic_scores >= 0.5
    ).astype(int)

    result.update(
        {
            "lr_auc": safe_roc_auc(
                y_test,
                logistic_scores,
            ),
            "lr_average_precision": (
                safe_average_precision(
                    y_test,
                    logistic_scores,
                )
            ),
            "lr_precision": float(
                precision_score(
                    y_test,
                    logistic_predictions,
                    zero_division=0,
                )
            ),
            "lr_recall": float(
                recall_score(
                    y_test,
                    logistic_predictions,
                    zero_division=0,
                )
            ),
            "lr_f1": float(
                f1_score(
                    y_test,
                    logistic_predictions,
                    zero_division=0,
                )
            ),
            "lr_balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test,
                    logistic_predictions,
                )
            ),
        }
    )

    return result


def run_one_repeat(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: BenchmarkConfig,
    repeat_index: int,
    signal_mask: Optional[np.ndarray] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run one real-world benchmark repetition.

    signal_mask is optional and should be supplied only for controlled
    simulation data. For real-world data, it must remain None.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            X,
            columns=feature_names,
        )

    if list(X.columns) != list(
        feature_names
    ):
        raise ValueError(
            "X columns must match feature_names."
        )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same "
            "number of rows."
        )

    random_state = (
        config.base_seed
        + repeat_index
    )

    contamination = (
        resolve_contamination(
            y=y,
            config=config,
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split_mixed_type(
        X=X,
        y=y,
        feature_types=feature_types,
        test_size=config.test_size,
        random_state=random_state,
    )

    model = fit_isolation_forest(
        X_train=X_train,
        config=config,
        contamination=contamination,
        random_state=random_state,
    )

    train_anomaly_mask = (
        get_model_anomaly_mask(
            model=model,
            X=X_train,
        )
    )

    original_scores, _, _ = (
        original_diffi_importance(
            iforest=model,
            X=X_train,
            adjust_iic=True,
        )
    )

    ad_diffi_scores = (
        compute_ad_diffi_importance(
            iforest=model,
            X=X_train,
            feature_types=feature_types,
            anomaly_mask=train_anomaly_mask,
        )
    )

    methods = {
        ORIGINAL_METHOD: original_scores,
        AD_DIFFI_METHOD: ad_diffi_scores,
    }

    score_records = []
    downstream_records = []

    signal_mask_array = None

    if signal_mask is not None:
        signal_mask_array = np.asarray(
            signal_mask,
            dtype=bool,
        )

        if signal_mask_array.shape != (
            len(feature_names),
        ):
            raise ValueError(
                "signal_mask must have one "
                "entry per feature."
            )

    for method, scores in methods.items():
        ranks = get_feature_ranks(
            scores
        )

        selected_indices = (
            select_top_features(
                scores=scores,
                top_k=config.top_k,
            )
        )

        downstream_metrics = (
            fit_downstream_models(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                selected_indices=selected_indices,
                config=config,
                contamination=contamination,
                random_state=random_state,
            )
        )

        record = {
            "repeat": repeat_index,
            "method": method,
            "selected_indices": ",".join(
                str(index)
                for index in selected_indices
            ),
            "selected_features": ",".join(
                feature_names[index]
                for index in selected_indices
            ),
            "n_selected_continuous": int(
                sum(
                    feature_types[index]
                    == "cont"
                    for index
                    in selected_indices
                )
            ),
            "n_selected_binary": int(
                sum(
                    feature_types[index]
                    == "bin"
                    for index
                    in selected_indices
                )
            ),
            **downstream_metrics,
        }

        if signal_mask_array is not None:
            signal_scores = scores[
                signal_mask_array
            ]

            noise_scores = scores[
                ~signal_mask_array
            ]

            record.update(
                {
                    "ranking_auc": float(
                        safe_roc_auc(
                            signal_mask_array.astype(
                                int
                            ),
                            scores,
                        )
                    ),
                    "signal_noise_gap": float(
                        signal_scores.min()
                        - noise_scores.max()
                    ),
                    "signal_noise_mean_difference": float(
                        signal_scores.mean()
                        - noise_scores.mean()
                    ),
                }
            )

        downstream_records.append(
            record
        )

        for feature_index, feature_name in enumerate(
            feature_names
        ):
            score_record = {
                "repeat": repeat_index,
                "method": method,
                "feature_index": feature_index,
                "feature": feature_name,
                "feature_type": feature_types[
                    feature_index
                ],
                "score": float(
                    scores[feature_index]
                ),
                "rank": int(
                    ranks[feature_index]
                ),
                "selected": bool(
                    feature_index
                    in selected_indices
                ),
            }

            if signal_mask_array is not None:
                score_record["signal"] = bool(
                    signal_mask_array[
                        feature_index
                    ]
                )

            score_records.append(
                score_record
            )

    return (
        pd.DataFrame(
            score_records
        ),
        pd.DataFrame(
            downstream_records
        ),
    )


def run_benchmark(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: Optional[
        BenchmarkConfig
    ] = None,
    signal_mask: Optional[
        np.ndarray
    ] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run repeated paired benchmarking.

    For real-world data, leave signal_mask=None.
    """
    if config is None:
        config = BenchmarkConfig()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            X,
            columns=feature_names,
        )

    if list(X.columns) != list(
        feature_names
    ):
        raise ValueError(
            "X columns must match feature_names."
        )

    y = np.asarray(
        y,
        dtype=int,
    )

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same "
            "number of rows."
        )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    all_scores = []
    all_downstream = []

    for repeat_index in range(
        config.n_repeats
    ):
        print(
            f"[{repeat_index + 1}/"
            f"{config.n_repeats}] "
            "running benchmark"
        )

        scores_df, downstream_df = (
            run_one_repeat(
                X=X,
                y=y,
                feature_names=feature_names,
                feature_types=feature_types,
                config=config,
                repeat_index=repeat_index,
                signal_mask=signal_mask,
            )
        )

        all_scores.append(
            scores_df
        )

        all_downstream.append(
            downstream_df
        )

    scores_df = pd.concat(
        all_scores,
        ignore_index=True,
    )

    downstream_df = pd.concat(
        all_downstream,
        ignore_index=True,
    )

    summary_df = summarize_benchmark(
        downstream_df
    )

    return (
        scores_df,
        downstream_df,
        summary_df,
    )


def summarize_benchmark(
    downstream_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize numeric benchmark metrics.
    """
    excluded = {
        "repeat",
        "method",
        "selected_indices",
        "selected_features",
    }

    numeric_columns = [
        column
        for column in downstream_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            downstream_df[column]
        )
    ]

    aggregations = {}

    for column in numeric_columns:
        aggregations[
            f"{column}_mean"
        ] = (
            column,
            "mean",
        )

        aggregations[
            f"{column}_sd"
        ] = (
            column,
            "std",
        )

    summary_df = (
        downstream_df
        .groupby(
            "method",
            as_index=False,
        )
        .agg(**aggregations)
    )

    method_order = {
        method: index
        for index, method
        in enumerate(
            METHOD_ORDER
        )
    }

    summary_df["_method_order"] = (
        summary_df["method"].map(
            method_order
        )
    )

    return (
        summary_df
        .sort_values(
            "_method_order"
        )
        .drop(
            columns="_method_order"
        )
        .reset_index(
            drop=True
        )
    )


def summarize_feature_scores(
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize feature scores and ranks.
    """
    group_columns = [
        "method",
        "feature_index",
        "feature",
        "feature_type",
    ]

    if "signal" in scores_df.columns:
        group_columns.append(
            "signal"
        )

    return (
        scores_df
        .groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            score_mean=(
                "score",
                "mean",
            ),
            score_sd=(
                "score",
                "std",
            ),
            rank_mean=(
                "rank",
                "mean",
            ),
            rank_sd=(
                "rank",
                "std",
            ),
            selection_rate=(
                "selected",
                "mean",
            ),
        )
        .sort_values(
            [
                "method",
                "rank_mean",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def save_benchmark_outputs(
    output_dir: str | Path,
    scores_df: pd.DataFrame,
    downstream_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Save benchmark outputs.
    """
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_df.to_csv(
        output_dir
        / "feature_scores.csv",
        index=False,
    )

    downstream_df.to_csv(
        output_dir
        / "downstream_metrics.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir
        / "benchmark_summary.csv",
        index=False,
    )

    feature_summary = (
        summarize_feature_scores(
            scores_df
        )
    )

    feature_summary.to_csv(
        output_dir
        / "feature_score_summary.csv",
        index=False,
    )
