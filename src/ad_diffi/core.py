from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from sklearn.ensemble import IsolationForest


FeatureTypes = Dict[int, str]

_CONTINUOUS = "cont"
_BINARY = "bin"

_VALID_FEATURE_TYPES = {
    _CONTINUOUS,
    _BINARY,
}


# ============================================================
# Input validation
# ============================================================

def _validate_feature_types(
    feature_types: FeatureTypes,
    n_features: int,
) -> None:
    """Validate complete feature-type metadata."""
    expected_indices = set(range(n_features))
    observed_indices = set(feature_types)

    missing_indices = expected_indices - observed_indices
    extra_indices = observed_indices - expected_indices

    if missing_indices:
        raise ValueError(
            f"Missing feature types: {sorted(missing_indices)}"
        )

    if extra_indices:
        raise ValueError(
            "Invalid feature indices: "
            f"{sorted(extra_indices, key=repr)}"
        )

    invalid_types = {
        feature: feature_types[feature]
        for feature in range(n_features)
        if feature_types[feature] not in _VALID_FEATURE_TYPES
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )


def _validate_anomaly_mask(
    anomaly_mask: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Validate a Boolean or numeric 0/1 outlier mask.

    Homogeneous masks are allowed and produce zero raw scores.
    """
    values = np.asarray(anomaly_mask)

    if values.shape != (n_samples,):
        raise ValueError(
            "anomaly_mask must have shape (n_samples,)."
        )

    if values.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(
            "anomaly_mask must contain Boolean or numeric 0/1 values."
        )

    if not np.all(np.isin(values, [0, 1])):
        raise ValueError(
            "anomaly_mask must contain only Boolean or 0/1 values."
        )

    return values.astype(
        bool,
        copy=False,
    )


def _validate_fitted_forest(
    iforest: IsolationForest,
    n_features: int,
) -> None:
    """Validate the supported fitted-forest configuration.

    All estimators must use all features in their original order.
    Bootstrap sampling is not supported.
    """
    if not isinstance(iforest, IsolationForest):
        raise TypeError(
            "iforest must be an sklearn IsolationForest."
        )

    estimators = getattr(
        iforest,
        "estimators_",
        None,
    )

    if estimators is None or len(estimators) == 0:
        raise ValueError(
            "iforest must be a fitted IsolationForest."
        )

    if not hasattr(iforest, "n_features_in_"):
        raise ValueError(
            "The fitted forest has no n_features_in_ attribute."
        )

    if iforest.n_features_in_ != n_features:
        raise ValueError(
            "X_data and iforest have different feature counts."
        )

    if iforest.bootstrap:
        raise ValueError(
            "This implementation requires bootstrap=False."
        )

    estimator_features = getattr(
        iforest,
        "estimators_features_",
        None,
    )

    if estimator_features is None:
        raise ValueError(
            "The fitted forest has no estimators_features_ attribute."
        )

    if len(estimator_features) != len(estimators):
        raise ValueError(
            "Inconsistent estimator feature metadata."
        )

    expected_features = np.arange(
        n_features,
        dtype=int,
    )

    for tree_index, selected_features in enumerate(
        estimator_features
    ):
        selected_features = np.asarray(
            selected_features,
            dtype=int,
        )

        if not np.array_equal(
            selected_features,
            expected_features,
        ):
            raise ValueError(
                "This implementation requires all features in their "
                "original order for every estimator. "
                "Use max_features=1.0. "
                f"Unsupported feature mapping in tree {tree_index}."
            )


def _validate_inputs(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate common raw AD-DIFFI inputs."""
    X_data = np.asarray(
        X_data,
        dtype=float,
    )

    if X_data.ndim != 2:
        raise ValueError(
            "X_data must be a two-dimensional array."
        )

    n_samples, n_features = X_data.shape

    if n_samples == 0 or n_features == 0:
        raise ValueError(
            "X_data must contain observations and features."
        )

    if not np.all(np.isfinite(X_data)):
        raise ValueError(
            "X_data must contain only finite values."
        )

    _validate_feature_types(
        feature_types,
        n_features,
    )

    for feature in range(n_features):
        if feature_types[feature] != _BINARY:
            continue

        if not np.all(
            np.isin(
                X_data[:, feature],
                [0, 1],
            )
        ):
            raise ValueError(
                f"Binary feature {feature} must contain only 0 and 1."
            )

    _validate_fitted_forest(
        iforest,
        n_features,
    )

    if anomaly_mask is None:
        return X_data, None

    validated_mask = _validate_anomaly_mask(
        anomaly_mask,
        n_samples,
    )

    return X_data, validated_mask


# ============================================================
# Reference implementation helpers
# ============================================================

def _node_lambdas(tree) -> np.ndarray:
    """Return fitted-tree split-imbalance coefficients.

    For an internal node v:
        lambda(v) = 1 - min(n_left, n_right) / n_parent.

    Leaf nodes receive zero.
    """
    lambdas = np.zeros(
        tree.node_count,
        dtype=float,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        n_parent = float(
            tree.n_node_samples[node]
        )

        if n_parent <= 0:
            continue

        n_left = tree.n_node_samples[left]
        n_right = tree.n_node_samples[right]

        lambdas[node] = 1.0 - (
            min(n_left, n_right) / n_parent
        )

    return lambdas


def _parent_array(tree) -> np.ndarray:
    """Return each node's parent index."""
    parent = np.full(
        tree.node_count,
        -1,
        dtype=int,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left != right:
            parent[left] = node
            parent[right] = node

    return parent


def _paths_for_rows(
    estimator,
    X_data: np.ndarray,
) -> list[list[int]]:
    """Return root-to-terminal paths for all observations."""
    tree = estimator.tree_
    parent = _parent_array(tree)

    terminal_nodes = estimator.apply(
        X_data
    ).astype(int)

    path_cache: dict[int, list[int]] = {}
    paths: list[list[int]] = []

    for terminal_node in terminal_nodes:
        terminal_node = int(terminal_node)

        if terminal_node not in path_cache:
            path: list[int] = []
            node = terminal_node

            while node >= 0:
                path.append(node)
                node = int(parent[node])

            path_cache[terminal_node] = path[::-1]

        paths.append(
            path_cache[terminal_node]
        )

    return paths


def _node_membership(
    estimator,
    X_data: np.ndarray,
) -> np.ndarray:
    """Return dense Boolean observation-by-node membership."""
    return (
        estimator.decision_path(X_data)
        .toarray()
        .astype(bool)
    )


def _node_outlier_enrichment_from_membership(
    tree,
    node_membership: np.ndarray,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Return child-node outlier-prevalence contrasts.

    Empty-child and leaf-node enrichment is zero.
    """
    enrichment = np.zeros(
        tree.node_count,
        dtype=float,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        in_left = node_membership[:, left]
        in_right = node_membership[:, right]

        n_left = int(in_left.sum())
        n_right = int(in_right.sum())

        if n_left == 0 or n_right == 0:
            continue

        outlier_rate_left = float(
            anomaly_mask[in_left].mean()
        )

        outlier_rate_right = float(
            anomaly_mask[in_right].mean()
        )

        enrichment[node] = abs(
            outlier_rate_left - outlier_rate_right
        )

    return enrichment


def _eligible_node(
    node: int,
    feature: int,
    feature_types: FeatureTypes,
) -> bool:
    """Allow continuous splits at all nodes, binary splits at root."""
    feature_type = feature_types[feature]

    return (
        feature_type == _CONTINUOUS
        or (
            feature_type == _BINARY
            and node == 0
        )
    )


def _feature_enrichment_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return observation-level numerators and denominators.

    An eligible occurrence contributes lambda(v) * E(v) / h_i
    to its numerator and one to its denominator.
    """
    n_samples, n_features = X_data.shape
    tree = estimator.tree_

    lambdas = _node_lambdas(tree)

    paths = _paths_for_rows(
        estimator,
        X_data,
    )

    node_membership = _node_membership(
        estimator,
        X_data,
    )

    enrichment = _node_outlier_enrichment_from_membership(
        tree,
        node_membership,
        anomaly_mask,
    )

    numerator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    denominator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    for row, path in enumerate(paths):
        internal_path = [
            node
            for node in path
            if tree.feature[node] >= 0
        ]

        path_length = max(
            len(internal_path),
            1,
        )

        for node in internal_path:
            feature = int(
                tree.feature[node]
            )

            if not _eligible_node(
                node,
                feature,
                feature_types,
            ):
                continue

            numerator[row, feature] += (
                lambdas[node]
                * enrichment[node]
                / path_length
            )

            denominator[row, feature] += 1.0

    return numerator, denominator


# ============================================================
# Observation-wise reference implementation
# ============================================================

def compute_enrichment_ad_diffi_reference(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute raw scores using the observation-wise reference.

    Retained for numerical-equivalence tests and debugging.
    """
    X_data, validated_mask = _validate_inputs(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    if validated_mask is None:
        raise ValueError(
            "anomaly_mask is required."
        )

    anomaly_mask = validated_mask
    n_features = X_data.shape[1]

    if anomaly_mask.all() or not anomaly_mask.any():
        return np.zeros(
            n_features,
            dtype=float,
        )

    total_tree_scores = np.zeros(
        n_features,
        dtype=float,
    )

    valid_tree_counts = np.zeros(
        n_features,
        dtype=int,
    )

    for estimator in iforest.estimators_:
        numerator, denominator = _feature_enrichment_terms(
            estimator,
            X_data,
            feature_types,
            anomaly_mask,
        )

        tree_numerator = numerator.sum(axis=0)
        tree_denominator = denominator.sum(axis=0)

        tree_scores = np.divide(
            tree_numerator,
            tree_denominator,
            out=np.zeros_like(tree_numerator),
            where=tree_denominator > 0,
        )

        valid = tree_denominator > 0

        total_tree_scores[valid] += tree_scores[valid]
        valid_tree_counts[valid] += 1

    forest_importance = np.divide(
        total_tree_scores,
        valid_tree_counts,
        out=np.zeros_like(total_tree_scores),
        where=valid_tree_counts > 0,
    )

    if not np.all(np.isfinite(forest_importance)):
        raise ValueError(
            "Raw AD-DIFFI produced non-finite scores."
        )

    return forest_importance


# ============================================================
# Fast sparse implementation: all features
# ============================================================

def compute_enrichment_ad_diffi_fast(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute all raw scores using sparse node-level aggregation.

    Split imbalance uses fitted-tree training counts.
    Enrichment uses supplied evaluation data and anomaly labels.
    Path weighting is the inverse number of internal splits.

    Continuous splits are eligible at all internal nodes.
    Binary splits are eligible only at the root.

    Scores are normalized within each tree, then averaged over
    trees with positive feature-specific denominators.
    """
    X_data, validated_mask = _validate_inputs(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    if validated_mask is None:
        raise ValueError(
            "anomaly_mask is required."
        )

    anomaly_mask = validated_mask
    n_features = X_data.shape[1]

    if anomaly_mask.all() or not anomaly_mask.any():
        return np.zeros(
            n_features,
            dtype=float,
        )

    feature_is_continuous = np.asarray(
        [
            feature_types[index] == _CONTINUOUS
            for index in range(n_features)
        ],
        dtype=bool,
    )

    total_tree_scores = np.zeros(
        n_features,
        dtype=float,
    )

    valid_tree_counts = np.zeros(
        n_features,
        dtype=int,
    )

    outlier_values = anomaly_mask.astype(
        float,
        copy=False,
    )

    for estimator in iforest.estimators_:
        tree = estimator.tree_

        left = tree.children_left
        right = tree.children_right

        internal_nodes = np.flatnonzero(
            left != right
        )

        if internal_nodes.size == 0:
            continue

        membership = (
            estimator.decision_path(X_data)
            .tocsr()
        )

        occupancy = np.asarray(
            membership.sum(axis=0)
        ).ravel()

        outlier_counts = np.asarray(
            membership.T @ outlier_values
        ).ravel()

        path_lengths = np.maximum(
            np.diff(membership.indptr) - 1,
            1,
        )

        weighted_occupancy = np.asarray(
            membership.T @ (1.0 / path_lengths)
        ).ravel()

        left_nodes = left[internal_nodes]
        right_nodes = right[internal_nodes]

        training_counts = tree.n_node_samples

        lambdas = 1.0 - (
            np.minimum(
                training_counts[left_nodes],
                training_counts[right_nodes],
            )
            / training_counts[internal_nodes]
        )

        left_occupancy = occupancy[left_nodes]
        right_occupancy = occupancy[right_nodes]

        left_outlier_rate = np.divide(
            outlier_counts[left_nodes],
            left_occupancy,
            out=np.zeros(
                internal_nodes.size,
                dtype=float,
            ),
            where=left_occupancy > 0,
        )

        right_outlier_rate = np.divide(
            outlier_counts[right_nodes],
            right_occupancy,
            out=np.zeros(
                internal_nodes.size,
                dtype=float,
            ),
            where=right_occupancy > 0,
        )

        enrichment = np.abs(
            left_outlier_rate - right_outlier_rate
        )

        empty_child = (
            (left_occupancy == 0)
            | (right_occupancy == 0)
        )

        enrichment[empty_child] = 0.0

        split_features = tree.feature[internal_nodes]

        eligible = (
            feature_is_continuous[split_features]
            | (internal_nodes == 0)
        )

        eligible_nodes = internal_nodes[eligible]
        eligible_features = split_features[eligible]

        if eligible_nodes.size == 0:
            continue

        numerator = np.bincount(
            eligible_features,
            weights=(
                lambdas[eligible]
                * enrichment[eligible]
                * weighted_occupancy[eligible_nodes]
            ),
            minlength=n_features,
        )

        denominator = np.bincount(
            eligible_features,
            weights=occupancy[eligible_nodes],
            minlength=n_features,
        )

        tree_scores = np.divide(
            numerator,
            denominator,
            out=np.zeros(
                n_features,
                dtype=float,
            ),
            where=denominator > 0,
        )

        valid = denominator > 0

        total_tree_scores[valid] += tree_scores[valid]
        valid_tree_counts[valid] += 1

    forest_importance = np.divide(
        total_tree_scores,
        valid_tree_counts,
        out=np.zeros(
            n_features,
            dtype=float,
        ),
        where=valid_tree_counts > 0,
    )

    if not np.all(np.isfinite(forest_importance)):
        raise ValueError(
            "Raw AD-DIFFI produced non-finite scores."
        )

    return forest_importance


# ============================================================
# Target-only evaluator for permutation references
# ============================================================

# TARGET_RAW_OPTIMIZATION_V1

def _build_target_raw_evaluator(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    feature_index: int,
):
    """Build a cached evaluator for one target feature.

    This internal API is intended for rearrangements of validated
    input data under the same unchanged fitted forest.

    Construction validates the original data and forest.
    Subsequent evaluations validate shape and anomaly masks but
    do not repeat full data-value or forest validation.

    Binary targets use only trees splitting on that target at root.
    Continuous targets use all trees with eligible target splits.

    The returned evaluator does not predict anomaly labels.
    The caller must recompute labels using the complete forest
    after every rearrangement.
    """
    X_data, _ = _validate_inputs(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
    )

    n_features = X_data.shape[1]

    if (
        isinstance(feature_index, (bool, np.bool_))
        or not isinstance(feature_index, (int, np.integer))
        or not 0 <= feature_index < n_features
    ):
        raise ValueError(
            "Invalid feature_index."
        )

    plans = []

    for estimator in iforest.estimators_:
        tree = estimator.tree_

        internal_nodes = np.flatnonzero(
            tree.children_left != tree.children_right
        )

        selected = (
            tree.feature[internal_nodes] == feature_index
        )

        if feature_types[feature_index] == _BINARY:
            selected &= internal_nodes == 0

        nodes = internal_nodes[selected]

        if nodes.size == 0:
            continue

        left_nodes = tree.children_left[nodes]
        right_nodes = tree.children_right[nodes]
        training_counts = tree.n_node_samples

        lambdas = 1.0 - (
            np.minimum(
                training_counts[left_nodes],
                training_counts[right_nodes],
            )
            / training_counts[nodes]
        )

        plans.append(
            (
                estimator,
                nodes,
                left_nodes,
                right_nodes,
                lambdas,
            )
        )

    def evaluate(
        X_eval: np.ndarray,
        anomaly_mask: np.ndarray,
    ) -> float:
        X_eval = np.asarray(
            X_eval,
            dtype=float,
        )

        if (
            X_eval.ndim != 2
            or X_eval.shape[1] != n_features
            or X_eval.shape[0] == 0
        ):
            raise ValueError(
                "Invalid evaluation-data shape."
            )

        mask = _validate_anomaly_mask(
            anomaly_mask,
            X_eval.shape[0],
        )

        if mask.all() or not mask.any():
            return 0.0

        outlier_values = mask.astype(
            float,
            copy=False,
        )

        score_total = 0.0
        valid_tree_count = 0

        for (
            estimator,
            nodes,
            left_nodes,
            right_nodes,
            lambdas,
        ) in plans:
            membership = (
                estimator.decision_path(X_eval)
                .tocsr()
            )

            occupancy = np.asarray(
                membership.sum(axis=0)
            ).ravel()

            outlier_counts = np.asarray(
                membership.T @ outlier_values
            ).ravel()

            path_lengths = np.maximum(
                np.diff(membership.indptr) - 1,
                1,
            )

            weighted_occupancy = np.asarray(
                membership.T @ (1.0 / path_lengths)
            ).ravel()

            left_occupancy = occupancy[left_nodes]
            right_occupancy = occupancy[right_nodes]

            left_outlier_rate = np.divide(
                outlier_counts[left_nodes],
                left_occupancy,
                out=np.zeros(
                    nodes.size,
                    dtype=float,
                ),
                where=left_occupancy > 0,
            )

            right_outlier_rate = np.divide(
                outlier_counts[right_nodes],
                right_occupancy,
                out=np.zeros(
                    nodes.size,
                    dtype=float,
                ),
                where=right_occupancy > 0,
            )

            enrichment = np.abs(
                left_outlier_rate - right_outlier_rate
            )

            enrichment[
                (left_occupancy == 0)
                | (right_occupancy == 0)
            ] = 0.0

            denominator = float(
                occupancy[nodes].sum()
            )

            if denominator <= 0:
                continue

            numerator = float(
                np.sum(
                    lambdas
                    * enrichment
                    * weighted_occupancy[nodes]
                )
            )

            score_total += numerator / denominator
            valid_tree_count += 1

        score = (
            score_total / valid_tree_count
            if valid_tree_count > 0
            else 0.0
        )

        if not np.isfinite(score):
            raise ValueError(
                "Non-finite target raw score."
            )

        return float(score)

    return evaluate


# ============================================================
# Public raw AD-DIFFI API
# ============================================================

def compute_enrichment_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute all raw, unstandardized AD-DIFFI scores.

    The fitted forest must use bootstrap=False and every feature
    in its original order for every estimator.

    This function does not refit the forest, infer anomaly labels,
    construct permutation references, or standardize scores.

    Homogeneous masks produce zero scores.
    Features without eligible occurrences receive zero.

    The observation-wise reference remains available for
    numerical-equivalence testing.
    """
    return compute_enrichment_ad_diffi_fast(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )


# Backward-compatible aliases: all return raw scores.
compute_raw_ad_diffi = compute_enrichment_ad_diffi
calculate_ad_diffi = compute_enrichment_ad_diffi


__all__ = [
    "FeatureTypes",
    "compute_enrichment_ad_diffi",
    "compute_enrichment_ad_diffi_reference",
    "compute_enrichment_ad_diffi_fast",
    "compute_raw_ad_diffi",
    "calculate_ad_diffi",
]
