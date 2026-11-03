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
    """Validate and return a Boolean model-defined outlier mask.

    The supplied mask must contain only Boolean or numeric 0/1 values.

    Homogeneous masks are allowed. If all observations are inliers or
    all observations are outliers, the child-node outlier-prevalence
    contrasts are zero, and raw AD-DIFFI scores are zero.

    Requirements concerning the presence of both model-defined groups
    in observed training data should be checked by the caller.
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
    """Validate the forest configuration supported by this implementation.

    All features must be supplied to every estimator in their original
    order. Feature subsampling is not supported.

    Bootstrap sampling is excluded from the supported configuration.
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
# Tree-level quantities
# ============================================================

def _node_lambdas(tree) -> np.ndarray:
    """Return raw split-imbalance coefficients for all tree nodes.

    For every internal node v,

        lambda(v) = 1 - min(n_left, n_right) / n_parent.

    Counts are taken from the fitted tree, not recomputed from the
    evaluation data.

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
            min(n_left, n_right)
            / n_parent
        )

    return lambdas


def _parent_array(tree) -> np.ndarray:
    """Return the parent-node index for every tree node."""
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
    """Return root-to-terminal node paths for all observations."""
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
    """Return a Boolean observation-by-node membership matrix."""
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
    """Return child-node model-defined outlier-prevalence contrasts.

    For every internal node v,

        E(v) = |P(outlier | left child of v)
             - P(outlier | right child of v)|.

    Membership and outlier prevalence are computed from the supplied
    evaluation data and mask.

    Leaf nodes and nodes with an empty child receive zero.
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

        n_left = int(
            in_left.sum()
        )
        n_right = int(
            in_right.sum()
        )

        if n_left == 0 or n_right == 0:
            continue

        outlier_rate_left = float(
            anomaly_mask[in_left].mean()
        )

        outlier_rate_right = float(
            anomaly_mask[in_right].mean()
        )

        enrichment[node] = abs(
            outlier_rate_left
            - outlier_rate_right
        )

    return enrichment


def _eligible_node(
    node: int,
    feature: int,
    feature_types: FeatureTypes,
) -> bool:
    """Return whether a split node is eligible under raw AD-DIFFI.

    Continuous features are eligible at all internal split nodes.
    Binary features are eligible only at the root.
    """
    feature_type = feature_types[feature]

    return (
        feature_type == _CONTINUOUS
        or (
            feature_type == _BINARY
            and node == 0
        )
    )


# ============================================================
# Observation-level contributions
# ============================================================

def _feature_enrichment_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-observation raw AD-DIFFI numerators and denominators.

    Each eligible occurrence of feature f contributes

        lambda(v) * E(v) / h_i

    to the numerator, and one to the denominator.

    h_i is the number of internal split nodes on observation i's path.
    """
    n_samples, n_features = X_data.shape
    tree = estimator.tree_

    lambdas = _node_lambdas(
        tree
    )

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
# Public raw AD-DIFFI API
# ============================================================

def compute_enrichment_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute unstandardized raw enrichment-based AD-DIFFI scores.

    Parameters
    ----------
    iforest
        A fitted IsolationForest with bootstrap=False and all features
        supplied to every estimator in their original order.
    X_data
        Finite two-dimensional evaluation data. Binary columns must
        contain only 0 and 1.
    feature_types
        Complete mapping from column indices to "cont" or "bin".
    anomaly_mask
        Boolean or numeric 0/1 model-defined outlier indicators for
        X_data. The mask is supplied by the caller; this function does
        not infer or modify outlier labels.

    Returns
    -------
    np.ndarray
        Nonnegative raw feature scores with shape (n_features,).

    Notes
    -----
    Node contributions combine fitted-tree split imbalance,
    child-node outlier-prevalence contrasts, and inverse
    observation-specific path length.

    Continuous features are eligible at all internal nodes.
    Binary features are eligible only at the root.

    Within each tree, contributions are divided by the number of
    eligible observation-node occurrences for the feature.

    Forest-level scores are averaged only over trees with positive
    eligible occurrence counts for the feature. Features without any
    eligible occurrences receive zero.

    Homogeneous outlier masks produce zero scores. No artificial
    outliers or inliers are introduced.

    This function does not refit the forest and does not perform
    permutation, hybrid reference estimation, or Z standardization.
    """
    X_data, validated_mask = _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
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

        tree_numerator = numerator.sum(
            axis=0
        )

        tree_denominator = denominator.sum(
            axis=0
        )

        tree_scores = np.divide(
            tree_numerator,
            tree_denominator,
            out=np.zeros_like(
                tree_numerator
            ),
            where=tree_denominator > 0,
        )

        valid = tree_denominator > 0

        total_tree_scores[valid] += tree_scores[valid]
        valid_tree_counts[valid] += 1

    forest_importance = np.divide(
        total_tree_scores,
        valid_tree_counts,
        out=np.zeros_like(
            total_tree_scores
        ),
        where=valid_tree_counts > 0,
    )

    if not np.all(
        np.isfinite(forest_importance)
    ):
        raise ValueError(
            "Raw AD-DIFFI produced non-finite scores."
        )

    return forest_importance


# Backward-compatible aliases.
# All three public names return RAW, unstandardized scores.
compute_raw_ad_diffi = compute_enrichment_ad_diffi
calculate_ad_diffi = compute_enrichment_ad_diffi


__all__ = [
    "FeatureTypes",
    "compute_enrichment_ad_diffi",
    "compute_raw_ad_diffi",
    "calculate_ad_diffi",
]
