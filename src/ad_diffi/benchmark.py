from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import json

import numpy as np
import pandas as pd
import sklearn

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import (
    train_test_split_mixed_type,
    validate_feature_metadata,
)


# ============================================================
# Version and method names
# ============================================================

BENCHMARK_IMPLEMENTATION_VERSION = "evaluable_na_v1"

ORIGINAL_METHOD = "Original DIFFI"
RAW_AD_DIFFI_METHOD = "Raw AD-DIFFI"
AD_DIFFI_METHOD = "Permutation-Z AD-DIFFI"
ALL_FEATURES_METHOD = "All features"

METHOD_ORDER = [
    ORIGINAL_METHOD,
    RAW_AD_DIFFI_METHOD,
    AD_DIFFI_METHOD,
    ALL_FEATURES_METHOD,
]

FoldPreprocessFunction = Callable[
    [
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
    Tuple[
        pd.DataFrame,
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
    ],
]


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for repeated real-world benchmarking."""

    n_repeats: int = 20
    test_size: float = 0.30

    n_estimators: int = 100
    max_samples: int = 256
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = -1

    # None: use training-fold label prevalence.
    model_contamination: Optional[float] = None

    top_k: int = 6
    downstream_max_iter: int = 1000
    base_seed: int = 42

    n_permutations: int = 10
    include_raw_ablation: bool = True
    include_all_features_baseline: bool = False

    # This controls evaluability, not denominator regularization.
    minimum_null_sd: float = 0.0

    permutation_output_dir: Optional[str] = None
    checkpoint_output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")

        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1).")

        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2.")

        if self.max_features != 1.0:
            raise ValueError(
                "The attribution implementation requires "
                "max_features=1.0."
            )

        if self.bootstrap:
            raise ValueError(
                "The attribution implementation requires "
                "bootstrap=False."
            )

        if self.model_contamination is not None:
            if not 0.0 < self.model_contamination <= 0.5:
                raise ValueError(
                    "model_contamination must be in (0, 0.5] "
                    "or None."
                )

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.downstream_max_iter < 1:
            raise ValueError(
                "downstream_max_iter must be at least 1."
            )

        if self.n_permutations < 2:
            raise ValueError(
                "n_permutations must be at least 2."
            )

        if (
            not np.isfinite(self.minimum_null_sd)
            or self.minimum_null_sd < 0
        ):
            raise ValueError(
                "minimum_null_sd must be finite and non-negative."
            )

        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be 0.")

        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative.")


# ============================================================
# Validation helpers
# ============================================================

def _validate_binary_labels(
    y: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(y)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(f"{name} must contain only 0 and 1.")

    values = values.astype(int)

    if np.unique(values).size != 2:
        raise ValueError(f"{name} must contain both classes.")

    return values


def _validate_score_vector(
    scores: np.ndarray,
    n_features: int,
    name: str,
    allow_nan: bool = False,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.shape != (n_features,):
        raise ValueError(
            f"{name} must have shape ({n_features},), "
            f"received {scores.shape}."
        )

    if np.any(np.isinf(scores)):
        raise ValueError(f"{name} contains infinity.")

    if not allow_nan and np.any(np.isnan(scores)):
        raise ValueError(f"{name} contains NaN.")

    return scores


def _validate_fitted_forest(
    iforest: IsolationForest,
    n_features: int,
) -> None:
    if (
        not hasattr(iforest, "estimators_")
        or len(iforest.estimators_) == 0
    ):
        raise ValueError("iforest must be fitted.")

    if iforest.n_features_in_ != n_features:
        raise ValueError("Forest and data feature counts differ.")

    if iforest.bootstrap:
        raise ValueError("bootstrap=False is required.")

    expected = np.arange(n_features)

    for feature_indices in iforest.estimators_features_:
        if not np.array_equal(feature_indices, expected):
            raise ValueError(
                "Attribution requires all features in original "
                "order for every estimator; use max_features=1.0."
            )


# ============================================================
# Isolation Forest helpers
# ============================================================

def resolve_contamination(
    y_train: np.ndarray,
    config: BenchmarkConfig,
) -> float:
    """
    Resolve contamination using a fixed value or training prevalence.

    Training-label prevalence makes this prevalence-informed,
    rather than entirely label-free.
    """
    if config.model_contamination is not None:
        return float(config.model_contamination)

    y_train = _validate_binary_labels(y_train, "y_train")
    contamination = float(y_train.mean())

    if not 0.0 < contamination <= 0.5:
        raise ValueError(
            "Training-fold prevalence must be in (0, 0.5]. "
            "Supply a prespecified contamination value if needed."
        )

    return contamination


def make_if_params(
    config: BenchmarkConfig,
    n_train: int,
    contamination: float,
) -> Dict[str, object]:
    if n_train < 2:
        raise ValueError("n_train must be at least 2.")

    return {
        "n_estimators": config.n_estimators,
        "max_samples": min(config.max_samples, n_train),
        "contamination": contamination,
        "max_features": config.max_features,
        "bootstrap": config.bootstrap,
        "n_jobs": config.n_jobs,
    }


def fit_isolation_forest(
    X_train: np.ndarray,
    config: BenchmarkConfig,
    contamination: float,
    random_state: int,
) -> IsolationForest:
    X_train = np.asarray(X_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train must be two-dimensional.")

    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train contains NaN or infinity.")

    return IsolationForest(
        **make_if_params(
            config=config,
            n_train=X_train.shape[0],
            contamination=contamination,
        ),
        random_state=random_state,
    ).fit(X_train)


def get_model_anomaly_mask(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.predict(np.asarray(X, dtype=float)) == -1,
        dtype=bool,
    )


def get_model_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    return -model.decision_function(
        np.asarray(X, dtype=float)
    )


# ============================================================
# Original DIFFI
# ============================================================

def _get_iic(
    estimator,
    predictions: np.ndarray,
    is_leaves: np.ndarray,
    adjust_iic: bool = True,
) -> np.ndarray:
    tree = estimator.tree_
    values = np.zeros(tree.node_count, dtype=float)

    if predictions.shape[0] == 0:
        return values

    membership = estimator.decision_path(predictions)
    counts = np.asarray(membership.sum(axis=0)).ravel()

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right or is_leaves[node]:
            values[node] = -1.0
            continue

        n_current = counts[node]

        if n_current <= 1:
            values[node] = -1.0
            continue

        n_left = counts[left]
        n_right = counts[right]

        if n_left == 0 or n_right == 0:
            values[node] = 0.0
            continue

        current_min = ceil(n_current / 2) / n_current
        current_max = (n_current - 1) / n_current
        imbalance = max(n_left, n_right) / n_current

        if adjust_iic and current_min != current_max:
            values[node] = (
                (imbalance - current_min)
                / (current_max - current_min)
                * 0.5
                + 0.5
            )
        else:
            values[node] = imbalance

    return values


def original_diffi_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    adjust_iic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preserve the supplied Original DIFFI scoring convention.

    AD-DIFFI's root-only eligibility rule is not applied here.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity.")

    n_features = X.shape[1]
    _validate_fitted_forest(iforest, n_features)

    cfi_outliers = np.zeros(n_features, dtype=float)
    cfi_inliers = np.zeros(n_features, dtype=float)
    counter_outliers = np.zeros(n_features, dtype=int)
    counter_inliers = np.zeros(n_features, dtype=int)

    anomaly_scores = iforest.decision_function(X)
    inbag_samples = iforest.estimators_samples_

    for tree_index, estimator in enumerate(iforest.estimators_):
        indices = inbag_samples[tree_index]
        X_inbag = X[indices]
        scores_inbag = anomaly_scores[indices]

        X_outliers = X_inbag[scores_inbag < 0]
        X_inliers = X_inbag[scores_inbag >= 0]

        if len(X_outliers) == 0 or len(X_inliers) == 0:
            continue

        tree = estimator.tree_
        left = tree.children_left
        right = tree.children_right
        node_depth = np.zeros(tree.node_count, dtype=int)
        is_leaves = left == right

        stack = [(0, 0)]

        while stack:
            node, depth = stack.pop()
            node_depth[node] = depth

            if not is_leaves[node]:
                stack.append((left[node], depth + 1))
                stack.append((right[node], depth + 1))

        def accumulate(
            X_subset: np.ndarray,
            cfi: np.ndarray,
            counter: np.ndarray,
        ) -> None:
            coefficients = _get_iic(
                estimator=estimator,
                predictions=X_subset,
                is_leaves=is_leaves,
                adjust_iic=adjust_iic,
            )

            membership = estimator.decision_path(
                X_subset
            ).tocsr()

            for row in range(len(X_subset)):
                nodes = membership.indices[
                    membership.indptr[row]:
                    membership.indptr[row + 1]
                ]

                depth = int(node_depth[nodes].max())

                if depth == 0:
                    continue

                for node in nodes:
                    feature = tree.feature[node]

                    if feature >= 0 and coefficients[node] != -1:
                        cfi[feature] += coefficients[node] / depth
                        counter[feature] += 1

        accumulate(
            X_outliers,
            cfi_outliers,
            counter_outliers,
        )
        accumulate(
            X_inliers,
            cfi_inliers,
            counter_inliers,
        )

    fi_out = np.divide(
        cfi_outliers,
        counter_outliers,
        out=np.zeros_like(cfi_outliers),
        where=counter_outliers > 0,
    )

    fi_in = np.divide(
        cfi_inliers,
        counter_inliers,
        out=np.zeros_like(cfi_inliers),
        where=counter_inliers > 0,
    )

    scores = np.divide(
        fi_out,
        fi_in,
        out=np.zeros_like(fi_out),
        where=fi_in > 0,
    )

    return scores, cfi_outliers, cfi_inliers


# ============================================================
# Raw enrichment-based AD-DIFFI
# ============================================================

def compute_raw_ad_diffi_fast(
    model: IsolationForest,
    X: np.ndarray,
    feature_types: Dict[int, str],
    outlier_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute numerical raw AD-DIFFI scores.

    Unsupported features return computational zero here.
    The benchmark reporting layer converts unsupported scores to NA.

    Continuous: all internal split nodes are eligible.
    Binary: only root split nodes are eligible.
    """
    X = np.asarray(X, dtype=float)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("X must contain observations.")

    if outlier_mask.shape != (n_samples,):
        raise ValueError("Invalid outlier_mask shape.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity.")

    if set(feature_types) != set(range(n_features)):
        raise ValueError("Incomplete feature-type metadata.")

    if any(
        value not in {"cont", "bin"}
        for value in feature_types.values()
    ):
        raise ValueError(
            "Feature types must be 'cont' or 'bin'."
        )

    _validate_fitted_forest(model, n_features)

    total_scores = np.zeros(n_features, dtype=float)
    valid_counts = np.zeros(n_features, dtype=int)

    for estimator in model.estimators_:
        tree = estimator.tree_
        left = tree.children_left
        right = tree.children_right

        internal = np.flatnonzero(left != right)

        if internal.size == 0:
            continue

        membership = estimator.decision_path(X).tocsr()
        occupancy = np.asarray(
            membership.sum(axis=0)
        ).ravel()

        outlier_counts = np.asarray(
            membership.T @ outlier_mask.astype(float)
        ).ravel()

        path_lengths = np.maximum(
            np.diff(membership.indptr) - 1,
            1,
        )

        weighted_occupancy = np.asarray(
            membership.T @ (1.0 / path_lengths)
        ).ravel()

        left_nodes = left[internal]
        right_nodes = right[internal]
        training_counts = tree.n_node_samples

        lambdas = 1.0 - (
            np.minimum(
                training_counts[left_nodes],
                training_counts[right_nodes],
            )
            / training_counts[internal]
        )

        left_n = occupancy[left_nodes]
        right_n = occupancy[right_nodes]

        left_prevalence = np.divide(
            outlier_counts[left_nodes],
            left_n,
            out=np.zeros(internal.size, dtype=float),
            where=left_n > 0,
        )

        right_prevalence = np.divide(
            outlier_counts[right_nodes],
            right_n,
            out=np.zeros(internal.size, dtype=float),
            where=right_n > 0,
        )

        enrichment = np.abs(
            left_prevalence - right_prevalence
        )
        enrichment[
            (left_n == 0) | (right_n == 0)
        ] = 0.0

        features = tree.feature[internal]

        eligible = np.array(
            [
                feature_types[int(feature)] == "cont"
                or node == 0
                for node, feature in zip(internal, features)
            ],
            dtype=bool,
        )

        nodes = internal[eligible]
        eligible_features = features[eligible]

        numerator = np.bincount(
            eligible_features,
            weights=(
                lambdas[eligible]
                * enrichment[eligible]
                * weighted_occupancy[nodes]
            ),
            minlength=n_features,
        )

        denominator = np.bincount(
            eligible_features,
            weights=occupancy[nodes],
            minlength=n_features,
        )

        tree_scores = np.divide(
            numerator,
            denominator,
            out=np.zeros(n_features, dtype=float),
            where=denominator > 0,
        )

        valid = denominator > 0
        total_scores[valid] += tree_scores[valid]
        valid_counts[valid] += 1

    return np.divide(
        total_scores,
        valid_counts,
        out=np.zeros(n_features, dtype=float),
        where=valid_counts > 0,
    )


def compute_ad_diffi_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    feature_types: Dict[int, str],
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    return compute_raw_ad_diffi_fast(
        model=iforest,
        X=X,
        feature_types=feature_types,
        outlier_mask=anomaly_mask,
    )


def _get_attribution_support(
    iforest: IsolationForest,
    feature_types: Dict[int, str],
    n_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Count structural attribution opportunities in the fixed forest.

    Called on the forest fitted to the same training observations
    used for observed attribution.
    """
    _validate_fitted_forest(iforest, n_features)

    root_features = np.asarray(
        [
            estimator.tree_.feature[0]
            for estimator in iforest.estimators_
        ],
        dtype=int,
    )

    n_root_trees = np.array(
        [
            np.sum(root_features == i)
            for i in range(n_features)
        ],
        dtype=int,
    )

    n_eligible_trees = np.zeros(n_features, dtype=int)

    for estimator in iforest.estimators_:
        tree = estimator.tree_
        internal = np.flatnonzero(
            tree.children_left != tree.children_right
        )

        eligible_features = {
            int(tree.feature[node])
            for node in internal
            if (
                feature_types[int(tree.feature[node])] == "cont"
                or node == 0
            )
        }

        for feature in eligible_features:
            n_eligible_trees[feature] += 1

    return root_features, n_root_trees, n_eligible_trees


# ============================================================
# Fixed-forest permutation standardization
# ============================================================

def compute_permutation_z_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    feature_types: Dict[int, str],
    raw_scores: np.ndarray,
    config: BenchmarkConfig,
    random_state: int,
    repeat_index: int,
    feature_names: List[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Return feature-specific Z-scores and diagnostics.

    Undefined Z-scores are NA, not zero or regularized values.
    The forest is fixed; model-defined outlier masks are recomputed
    after each one-column permutation.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a nonempty 2D matrix.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinity.")

    n_features = X.shape[1]

    if set(feature_types) != set(range(n_features)):
        raise ValueError("Incomplete feature-type metadata.")

    if any(
        value not in {"cont", "bin"}
        for value in feature_types.values()
    ):
        raise ValueError(
            "Feature types must be 'cont' or 'bin'."
        )

    raw_scores = _validate_score_vector(
        raw_scores,
        n_features,
        "raw_scores",
    )

    if len(feature_names) != n_features:
        raise ValueError("Invalid feature_names length.")

    (
        root_features,
        n_root_trees,
        n_eligible_trees,
    ) = _get_attribution_support(
        iforest,
        feature_types,
        n_features,
    )

    rng = np.random.default_rng(random_state)

    null_scores = np.empty(
        (n_features, config.n_permutations),
        dtype=float,
    )
    null_outlier_counts = np.empty(
        (n_features, config.n_permutations),
        dtype=int,
    )

    for feature_index in range(n_features):
        for permutation_index in range(config.n_permutations):
            X_permuted = X.copy()
            X_permuted[:, feature_index] = rng.permutation(
                X[:, feature_index]
            )

            mask = get_model_anomaly_mask(
                iforest,
                X_permuted,
            )

            scores = compute_ad_diffi_importance(
                iforest=iforest,
                X=X_permuted,
                feature_types=feature_types,
                anomaly_mask=mask,
            )

            null_scores[
                feature_index,
                permutation_index,
            ] = scores[feature_index]

            null_outlier_counts[
                feature_index,
                permutation_index,
            ] = int(mask.sum())

    if not np.all(np.isfinite(null_scores)):
        raise ValueError(
            "Non-finite permutation reference scores."
        )

    null_mean = null_scores.mean(axis=1)
    null_sd = null_scores.std(axis=1, ddof=1)
    null_range = np.ptp(null_scores, axis=1)

    # Numerical degeneracy check, not a statistical stability rule.
    reference_scale = np.maximum(
        np.max(np.abs(null_scores), axis=1),
        np.finfo(float).tiny,
    )
    numerical_tolerance = (
        100.0 * np.finfo(float).eps * reference_scale
    )
    numerically_constant = (
        null_range <= numerical_tolerance
    )

    no_support = n_eligible_trees == 0

    invalid_sd = (
        ~np.isfinite(null_sd)
        | (null_sd <= config.minimum_null_sd)
        | numerically_constant
    )

    raw_status = np.where(
        no_support,
        "no_eligible_split_support",
        "ok",
    ).astype(object)

    z_status = np.select(
        [
            no_support,
            ~np.isfinite(null_sd),
            numerically_constant,
            null_sd <= config.minimum_null_sd,
        ],
        [
            "no_eligible_split_support",
            "nonfinite_null_sd",
            "degenerate_null",
            "null_sd_below_threshold",
        ],
        default="ok",
    ).astype(object)

    valid_z = ~no_support & ~invalid_sd
    z_scores = np.full(n_features, np.nan, dtype=float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        np.divide(
            raw_scores - null_mean,
            null_sd,
            out=z_scores,
            where=valid_z,
        )

    nonfinite_z = valid_z & ~np.isfinite(z_scores)
    z_status[nonfinite_z] = "nonfinite_z"
    z_scores[nonfinite_z] = np.nan
    valid_z = np.isfinite(z_scores)

    n_unique_train = np.array(
        [
            np.unique(X[:, i]).size
            for i in range(n_features)
        ],
        dtype=int,
    )

    n_minority_train = np.full(n_features, np.nan)

    for i in range(n_features):
        if feature_types[i] == "bin":
            _, counts = np.unique(
                X[:, i],
                return_counts=True,
            )
            n_minority_train[i] = (
                int(counts.min())
                if len(counts) == 2
                else 0
            )

    diagnostics = pd.DataFrame({
        "repeat": repeat_index,
        "feature_index": np.arange(n_features),
        "feature": feature_names,
        "feature_type": [
            feature_types[i]
            for i in range(n_features)
        ],
        # Computational output retained for auditing.
        "raw_score": raw_scores,
        # Reportable raw score after support checking.
        "raw_score_evaluable": np.where(
            no_support,
            np.nan,
            raw_scores,
        ),
        "raw_status": raw_status,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_range": null_range,
        "null_numerical_tolerance": numerical_tolerance,
        "numerically_constant_null": numerically_constant,
        "valid_null_sd": ~invalid_sd,
        "valid_z": valid_z,
        "z_score": z_scores,
        "z_status": z_status,
        "n_root_trees": n_root_trees,
        "n_eligible_trees": n_eligible_trees,
        "n_unique_train": n_unique_train,
        "n_minority_train": n_minority_train,
        "n_permutations": config.n_permutations,
        "permuted_outlier_count_min": (
            null_outlier_counts.min(axis=1)
        ),
        "permuted_outlier_count_max": (
            null_outlier_counts.max(axis=1)
        ),
    })

    if config.permutation_output_dir is not None:
        directory = Path(config.permutation_output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        diagnostics.to_csv(
            directory
            / f"repeat_{repeat_index:03d}_diagnostics.csv",
            index=False,
            na_rep="NA",
        )

        np.savez_compressed(
            directory
            / f"repeat_{repeat_index:03d}_null_scores.npz",
            null_scores=null_scores,
            null_mean=null_mean,
            null_sd=null_sd,
            raw_scores=raw_scores,
            z_scores=z_scores,
            valid_z=valid_z,
            feature_names=np.asarray(feature_names),
            root_features=root_features,
            n_eligible_trees=n_eligible_trees,
            null_outlier_counts=null_outlier_counts,
            random_state=random_state,
        )

    return z_scores, diagnostics


# ============================================================
# Metrics and feature selection
# ============================================================

def safe_roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if np.unique(y_true).size < 2:
        return float("nan")

    return float(roc_auc_score(y_true, scores))


def safe_average_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if y_true.sum() == 0:
        return float("nan")

    return float(
        average_precision_score(y_true, scores)
    )


def select_top_features(
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Select exactly top-k finite scores; do not pad using NA."""
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if np.any(np.isinf(scores)):
        raise ValueError("scores contains infinity.")

    if not 1 <= top_k <= len(scores):
        raise ValueError(
            "top_k must be between 1 and n_features."
        )

    valid_indices = np.flatnonzero(np.isfinite(scores))

    if valid_indices.size < top_k:
        raise ValueError(
            f"Only {valid_indices.size} features are evaluable, "
            f"but top_k={top_k}. "
            "Unevaluable features are not used for padding. "
            "Inspect permutation diagnostics."
        )

    order = np.argsort(
        -scores[valid_indices],
        kind="stable",
    )

    return valid_indices[order[:top_k]]


def get_feature_ranks(
    scores: np.ndarray,
) -> np.ndarray:
    """Rank evaluable features and preserve NA for the rest."""
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if np.any(np.isinf(scores)):
        raise ValueError("scores contains infinity.")

    valid_indices = np.flatnonzero(np.isfinite(scores))

    order = valid_indices[
        np.argsort(
            -scores[valid_indices],
            kind="stable",
        )
    ]

    ranks = np.full(len(scores), np.nan, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)

    return ranks


def _classification_metrics(
    prefix: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> Dict[str, float]:
    return {
        f"{prefix}_auc": safe_roc_auc(y_true, scores),
        f"{prefix}_average_precision": (
            safe_average_precision(y_true, scores)
        ),
        f"{prefix}_precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
    }


# ============================================================
# Downstream models
# ============================================================

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
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if selected_indices.size == 0:
        raise ValueError(
            "selected_indices must not be empty."
        )

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    downstream_if = fit_isolation_forest(
        X_train=X_train_selected,
        config=config,
        contamination=contamination,
        random_state=random_state,
    )

    if_scores = get_model_anomaly_scores(
        downstream_if,
        X_test_selected,
    )
    if_predictions = get_model_anomaly_mask(
        downstream_if,
        X_test_selected,
    ).astype(int)

    result = _classification_metrics(
        prefix="if",
        y_true=y_test,
        scores=if_scores,
        predictions=if_predictions,
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=config.downstream_max_iter,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
    )

    logistic.fit(X_train_selected, y_train)

    logistic_scores = logistic.predict_proba(
        X_test_selected
    )[:, 1]
    logistic_predictions = (
        logistic_scores >= 0.5
    ).astype(int)

    result.update(
        _classification_metrics(
            prefix="lr",
            y_true=y_test,
            scores=logistic_scores,
            predictions=logistic_predictions,
        )
    )

    return result


# ============================================================
# One repeated split
# ============================================================

def run_one_repeat(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: BenchmarkConfig,
    repeat_index: int,
    signal_mask: Optional[np.ndarray] = None,
    fold_preprocess_fn: Optional[
        FoldPreprocessFunction
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    if list(X.columns) != list(feature_names):
        raise ValueError(
            "X columns must match feature_names."
        )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    y = _validate_binary_labels(y, "y")

    if len(X) != len(y):
        raise ValueError(
            "X and y have different row counts."
        )

    n_features = len(feature_names)

    if config.top_k >= n_features:
        raise ValueError(
            "Feature-selection comparison requires "
            "top_k < n_features."
        )

    signal_mask_array = None

    if signal_mask is not None:
        signal_mask_array = np.asarray(
            signal_mask,
            dtype=bool,
        )

        if signal_mask_array.shape != (n_features,):
            raise ValueError("Invalid signal_mask shape.")

        if (
            signal_mask_array.all()
            or (~signal_mask_array).all()
        ):
            raise ValueError(
                "signal_mask requires both signal "
                "and noise features."
            )

    random_state = config.base_seed + repeat_index

    X_train, X_test, y_train, y_test = (
        train_test_split_mixed_type(
            X=X,
            y=y,
            feature_types=feature_types,
            test_size=config.test_size,
            random_state=random_state,
        )
    )

    if fold_preprocess_fn is not None:
        X_train, X_test, y_train, y_test = (
            fold_preprocess_fn(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

    for name, data in [
        ("X_train", X_train),
        ("X_test", X_test),
    ]:
        if isinstance(data, pd.DataFrame):
            if list(data.columns) != list(feature_names):
                raise ValueError(
                    f"{name}: preprocessing changed "
                    "feature names, count, or order."
                )

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = _validate_binary_labels(
        y_train,
        "y_train",
    )
    y_test = _validate_binary_labels(
        y_test,
        "y_test",
    )

    for name, data in [
        ("X_train", X_train),
        ("X_test", X_test),
    ]:
        if data.ndim != 2 or data.shape[1] != n_features:
            raise ValueError(
                f"Unexpected {name} shape."
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"{name} contains NaN or infinity "
                "after preprocessing."
            )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training row counts do not match."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test row counts do not match."
        )

    contamination = resolve_contamination(
        y_train=y_train,
        config=config,
    )

    attribution_model = fit_isolation_forest(
        X_train=X_train,
        config=config,
        contamination=contamination,
        random_state=random_state,
    )

    train_anomaly_mask = get_model_anomaly_mask(
        attribution_model,
        X_train,
    )

    if (
        train_anomaly_mask.all()
        or (~train_anomaly_mask).all()
    ):
        raise ValueError(
            "Observed training data require both "
            "model-defined outliers and inliers."
        )

    start = perf_counter()
    original_scores, _, _ = original_diffi_importance(
        iforest=attribution_model,
        X=X_train,
        adjust_iic=True,
    )
    original_seconds = perf_counter() - start

    start = perf_counter()
    raw_scores = compute_ad_diffi_importance(
        iforest=attribution_model,
        X=X_train,
        feature_types=feature_types,
        anomaly_mask=train_anomaly_mask,
    )
    raw_seconds = perf_counter() - start

    start = perf_counter()
    z_scores, diagnostics = (
        compute_permutation_z_importance(
            iforest=attribution_model,
            X=X_train,
            feature_types=feature_types,
            raw_scores=raw_scores,
            config=config,
            random_state=random_state,
            repeat_index=repeat_index,
            feature_names=feature_names,
        )
    )
    permutation_seconds = perf_counter() - start

    methods = {
        ORIGINAL_METHOD: original_scores,
    }

    if config.include_raw_ablation:
        methods[RAW_AD_DIFFI_METHOD] = diagnostics[
            "raw_score_evaluable"
        ].to_numpy(dtype=float)

    methods[AD_DIFFI_METHOD] = z_scores

    scoring_seconds = {
        ORIGINAL_METHOD: original_seconds,
        RAW_AD_DIFFI_METHOD: raw_seconds,
        AD_DIFFI_METHOD: (
            raw_seconds + permutation_seconds
        ),
    }

    common_record = {
        "repeat": repeat_index,
        "random_state": random_state,
        "contamination": contamination,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "n_negative_train": int((y_train == 0).sum()),
        "n_negative_test": int((y_test == 0).sum()),
        "n_model_outlier_train": int(
            train_anomaly_mask.sum()
        ),
        "test_prevalence": float(y_test.mean()),
    }

    score_records = []
    downstream_records = []

    def evaluate_selected(
        method: str,
        selected_indices: np.ndarray,
        score_time: float,
    ) -> Dict[str, object]:
        metrics = fit_downstream_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            selected_indices=selected_indices,
            config=config,
            contamination=contamination,
            random_state=random_state,
        )

        return {
            **common_record,
            "method": method,
            "selected_indices": ",".join(
                str(int(i))
                for i in selected_indices
            ),
            "selected_features": ",".join(
                feature_names[i]
                for i in selected_indices
            ),
            "n_selected_features": int(
                len(selected_indices)
            ),
            "n_selected_continuous": int(sum(
                feature_types[i] == "cont"
                for i in selected_indices
            )),
            "n_selected_binary": int(sum(
                feature_types[i] == "bin"
                for i in selected_indices
            )),
            "attribution_seconds": float(score_time),
            **metrics,
        }

    for method, scores in methods.items():
        scores = _validate_score_vector(
            scores,
            n_features,
            method,
            allow_nan=True,
        )

        evaluable = np.isfinite(scores)

        if method == AD_DIFFI_METHOD:
            statuses = diagnostics[
                "z_status"
            ].to_numpy(dtype=str)
        elif method == RAW_AD_DIFFI_METHOD:
            statuses = diagnostics[
                "raw_status"
            ].to_numpy(dtype=str)
        else:
            # Preserved Original DIFFI convention.
            statuses = np.full(
                n_features,
                "ok",
                dtype=object,
            )

        ranks = get_feature_ranks(scores)

        selected_indices = select_top_features(
            scores=scores,
            top_k=config.top_k,
        )

        record = evaluate_selected(
            method,
            selected_indices,
            scoring_seconds[method],
        )

        record.update({
            "n_evaluable_features": int(
                evaluable.sum()
            ),
            "n_unevaluable_features": int(
                (~evaluable).sum()
            ),
        })

        if signal_mask_array is not None:
            if evaluable.all():
                signal_scores = scores[
                    signal_mask_array
                ]
                noise_scores = scores[
                    ~signal_mask_array
                ]

                record.update({
                    "ranking_auc": safe_roc_auc(
                        signal_mask_array.astype(int),
                        scores,
                    ),
                    "signal_noise_gap": float(
                        signal_scores.min()
                        - noise_scores.max()
                    ),
                    "signal_noise_mean_difference": float(
                        signal_scores.mean()
                        - noise_scores.mean()
                    ),
                })
            else:
                # Do not silently redefine full-feature AUC
                # as AUC among only evaluable features.
                record.update({
                    "ranking_auc": np.nan,
                    "signal_noise_gap": np.nan,
                    "signal_noise_mean_difference": np.nan,
                })

        downstream_records.append(record)
        selected_set = set(selected_indices.tolist())

        for i, feature_name in enumerate(feature_names):
            score_record = {
                "repeat": repeat_index,
                "method": method,
                "feature_index": i,
                "feature": feature_name,
                "feature_type": feature_types[i],
                "score": float(scores[i]),
                "rank": (
                    int(ranks[i])
                    if np.isfinite(ranks[i])
                    else np.nan
                ),
                "selected": i in selected_set,
                "evaluable": bool(evaluable[i]),
                "evaluation_status": str(statuses[i]),
                # Computational raw output, not always evaluable.
                "raw_ad_diffi": float(raw_scores[i]),
                "permutation_null_mean": float(
                    diagnostics.loc[i, "null_mean"]
                ),
                "permutation_null_sd": float(
                    diagnostics.loc[i, "null_sd"]
                ),
                "n_root_trees": int(
                    diagnostics.loc[i, "n_root_trees"]
                ),
                "n_eligible_trees": int(
                    diagnostics.loc[i, "n_eligible_trees"]
                ),
                "n_minority_train": float(
                    diagnostics.loc[i, "n_minority_train"]
                ),
            }

            if signal_mask_array is not None:
                score_record["signal"] = bool(
                    signal_mask_array[i]
                )

            score_records.append(score_record)

    if config.include_all_features_baseline:
        all_features_record = evaluate_selected(
            ALL_FEATURES_METHOD,
            np.arange(n_features),
            0.0,
        )

        # Not applicable: this baseline performs no attribution.
        all_features_record.update({
            "n_evaluable_features": np.nan,
            "n_unevaluable_features": np.nan,
        })

        downstream_records.append(
            all_features_record
        )

    return (
        pd.DataFrame(score_records),
        pd.DataFrame(downstream_records),
    )


# ============================================================
# Repeated benchmark
# ============================================================

def run_benchmark(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: Optional[BenchmarkConfig] = None,
    signal_mask: Optional[np.ndarray] = None,
    fold_preprocess_fn: Optional[
        FoldPreprocessFunction
    ] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if config is None:
        config = BenchmarkConfig()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            X,
            columns=feature_names,
        )

    if list(X.columns) != list(feature_names):
        raise ValueError(
            "X columns must match feature_names."
        )

    y = _validate_binary_labels(y, "y")

    if len(X) != len(y):
        raise ValueError(
            "X and y have different row counts."
        )

    validate_feature_metadata(
        feature_names=feature_names,
        feature_types=feature_types,
    )

    checkpoint_dir = None

    if config.checkpoint_output_dir is not None:
        checkpoint_dir = Path(
            config.checkpoint_output_dir
        )
        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = {
            "implementation_version": (
                BENCHMARK_IMPLEMENTATION_VERSION
            ),
            "config": asdict(config),
            "feature_names": list(feature_names),
            "feature_types": {
                str(i): value
                for i, value in feature_types.items()
            },
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "permutation_rule": (
                "fixed forest; one column shuffled; "
                "model outlier labels recomputed"
            ),
            "raw_averaging": (
                "conditional on positive eligible "
                "occurrences for continuous and binary features"
            ),
            "unevaluable_policy": (
                "NA score/rank; excluded from top-k; "
                "stop if fewer than top-k are evaluable"
            ),
            "null_numerical_tolerance": (
                "100 * float64 epsilon * "
                "max(max absolute null score, float64 tiny)"
            ),
            "original_diffi_policy": (
                "preserve supplied Original DIFFI convention; "
                "do not apply AD-DIFFI root-only support mask"
            ),
            "logistic_scaling": (
                "StandardScaler fitted on selected "
                "training features"
            ),
        }

        with open(
            checkpoint_dir / "benchmark_config.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    all_scores = []
    all_downstream = []

    for repeat_index in range(config.n_repeats):
        start = perf_counter()

        print(
            f"[{repeat_index + 1}/{config.n_repeats}] "
            "running benchmark",
            flush=True,
        )

        repeat_scores, repeat_downstream = (
            run_one_repeat(
                X=X,
                y=y,
                feature_names=feature_names,
                feature_types=feature_types,
                config=config,
                repeat_index=repeat_index,
                signal_mask=signal_mask,
                fold_preprocess_fn=fold_preprocess_fn,
            )
        )

        all_scores.append(repeat_scores)
        all_downstream.append(repeat_downstream)

        if checkpoint_dir is not None:
            pd.concat(
                all_scores,
                ignore_index=True,
            ).to_csv(
                checkpoint_dir
                / "feature_scores_partial.csv",
                index=False,
                na_rep="NA",
            )

            pd.concat(
                all_downstream,
                ignore_index=True,
            ).to_csv(
                checkpoint_dir
                / "downstream_metrics_partial.csv",
                index=False,
                na_rep="NA",
            )

        print(
            f"Completed repeat {repeat_index + 1}: "
            f"{perf_counter() - start:.1f} seconds",
            flush=True,
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

    return scores_df, downstream_df, summary_df


# ============================================================
# Summaries
# ============================================================

def summarize_benchmark(
    downstream_df: pd.DataFrame,
) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in downstream_df.columns
        if (
            column.startswith("if_")
            or column.startswith("lr_")
            or column.startswith("n_selected_")
            or column in {
                "attribution_seconds",
                "n_evaluable_features",
                "n_unevaluable_features",
            }
        )
        and pd.api.types.is_numeric_dtype(
            downstream_df[column]
        )
    ]

    aggregations = {}

    for column in numeric_columns:
        aggregations[f"{column}_mean"] = (
            column,
            "mean",
        )
        aggregations[f"{column}_sd"] = (
            column,
            "std",
        )

    summary_df = (
        downstream_df
        .groupby("method", as_index=False)
        .agg(**aggregations)
    )

    order = {
        method: i
        for i, method in enumerate(METHOD_ORDER)
    }

    summary_df["_method_order"] = (
        summary_df["method"].map(order)
    )

    return (
        summary_df
        .sort_values("_method_order")
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def summarize_feature_scores(
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "method",
        "feature_index",
        "feature",
        "feature_type",
    ]

    if "signal" in scores_df.columns:
        group_columns.append("signal")

    result = (
        scores_df
        .groupby(group_columns, as_index=False)
        .agg(
            n_repeats=("repeat", "size"),
            n_evaluable_repeats=("evaluable", "sum"),
            n_selected_repeats=("selected", "sum"),
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            rank_mean=("rank", "mean"),
            rank_sd=("rank", "std"),
            selection_rate=("selected", "mean"),
        )
    )

    result["evaluation_rate"] = (
        result["n_evaluable_repeats"]
        / result["n_repeats"]
    )

    result["selection_rate_when_evaluable"] = (
        result["n_selected_repeats"]
        / result["n_evaluable_repeats"].replace(
            0,
            np.nan,
        )
    )

    return (
        result
        .sort_values(["method", "rank_mean"])
        .reset_index(drop=True)
    )


def summarize_paired_comparisons(
    downstream_df: pd.DataFrame,
    proposed_method: str = AD_DIFFI_METHOD,
    reference_method: str = ORIGINAL_METHOD,
    metrics: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize paired repeat-level differences.

    Repeated splits are overlapping; no independent-sample
    confidence interval is constructed here.
    """
    if proposed_method == reference_method:
        raise ValueError(
            "Proposed and reference methods must differ."
        )

    if metrics is None:
        metrics = [
            "if_auc",
            "if_average_precision",
            "lr_auc",
            "lr_average_precision",
        ]

    missing = [
        metric
        for metric in metrics
        if metric not in downstream_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metrics: {missing}"
        )

    required_methods = {
        proposed_method,
        reference_method,
    }

    expected_repeats = set(
        downstream_df["repeat"].unique()
    )

    paired_records = []

    for repeat_index in sorted(expected_repeats):
        current = downstream_df.loc[
            (downstream_df["repeat"] == repeat_index)
            & downstream_df["method"].isin(
                required_methods
            )
        ]

        if current["method"].duplicated().any():
            raise ValueError(
                f"Duplicate method rows in repeat "
                f"{repeat_index}."
            )

        if set(current["method"]) != required_methods:
            raise ValueError(
                f"Incomplete pair in repeat "
                f"{repeat_index}."
            )

        current = current.set_index("method")

        record = {
            "repeat": repeat_index,
            "proposed_method": proposed_method,
            "reference_method": reference_method,
        }

        for metric in metrics:
            proposed = float(
                current.loc[proposed_method, metric]
            )
            reference = float(
                current.loc[reference_method, metric]
            )

            if not (
                np.isfinite(proposed)
                and np.isfinite(reference)
            ):
                raise ValueError(
                    f"Non-finite paired metric "
                    f"{metric}, repeat {repeat_index}."
                )

            record[f"{metric}_delta"] = (
                proposed - reference
            )

        paired_records.append(record)

    paired_df = pd.DataFrame(paired_records)
    summary_records = []

    for metric in metrics:
        differences = paired_df[
            f"{metric}_delta"
        ].to_numpy(dtype=float)

        summary_records.append({
            "metric": f"{metric}_delta",
            "proposed_method": proposed_method,
            "reference_method": reference_method,
            "n_pairs": len(differences),
            "mean": float(differences.mean()),
            "sd": (
                float(differences.std(ddof=1))
                if len(differences) > 1
                else float("nan")
            ),
            "win_rate": float(
                np.mean(differences > 1e-12)
            ),
            "tie_rate": float(
                np.mean(
                    np.abs(differences) <= 1e-12
                )
            ),
            "loss_rate": float(
                np.mean(differences < -1e-12)
            ),
        })

    return paired_df, pd.DataFrame(summary_records)


# ============================================================
# Output
# ============================================================

def save_benchmark_outputs(
    output_dir: str | Path,
    scores_df: pd.DataFrame,
    downstream_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "feature_scores.csv": scores_df,
        "downstream_metrics.csv": downstream_df,
        "benchmark_summary.csv": summary_df,
        "feature_score_summary.csv": (
            summarize_feature_scores(scores_df)
        ),
    }

    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            output_dir / filename,
            index=False,
            na_rep="NA",
        )

    paired_df, paired_summary = (
        summarize_paired_comparisons(
            downstream_df=downstream_df,
        )
    )

    paired_df.to_csv(
        output_dir
        / "paired_comparison_by_repeat.csv",
        index=False,
        na_rep="NA",
    )

    paired_summary.to_csv(
        output_dir
        / "paired_comparison_summary.csv",
        index=False,
        na_rep="NA",
    )
