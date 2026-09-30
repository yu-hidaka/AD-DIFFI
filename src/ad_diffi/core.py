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
            f"Invalid feature indices: {sorted(extra_indices)}"
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
    """Validate and return a Boolean model-defined outlier mask."""
    anomaly_mask = np.asarray(
        anomaly_mask,
        dtype=bool,
    )

    if anomaly_mask.shape != (n_samples,):
        raise ValueError(
            "anomaly_mask must have shape (n_samples,)."
        )

    if anomaly_mask.all() or (~anomaly_mask).all():
        raise ValueError(
            "Both model-defined outliers and inliers are required."
        )

    return anomaly_mask


def _validate_inputs(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate common AD-DIFFI inputs."""
    X_data = np.asarray(
        X_data,
        dtype=float,
    )

    if X_data.ndim != 2:
        raise ValueError(
            "X_data must be a two-dimensional array."
        )

    if (
        not hasattr(iforest, "estimators_")
        or not iforest.estimators_
    ):
        raise ValueError(
            "iforest must be a fitted IsolationForest."
        )

    n_samples, n_features = X_data.shape

    if (
        hasattr(iforest, "n_features_in_")
        and iforest.n_features_in_ != n_features
    ):
        raise ValueError(
            "X_data and iforest have different feature counts."
        )

    _validate_feature_types(
        feature_types,
        n_features,
    )

    if anomaly_mask is None:
        return X_data, None

    return X_data, _validate_anomaly_mask(
        anomaly_mask,
        n_samples,
    )


def _node_lambdas(tree) -> np.ndarray:
    """Return raw split-imbalance coefficients for all tree nodes.

    For every internal node v,

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

    terminal_nodes = estimator.apply(X_data).astype(int)

    path_cache: dict[int, list[int]] = {}
    paths: list[list[int]] = []

    for terminal_node in terminal_nodes:
        terminal_node = int(terminal_node)

        if terminal_node not in path_cache:
            path = []
            node = terminal_node

            while node >= 0:
                path.append(node)
                node = parent[node]

            path_cache[terminal_node] = path[::-1]

        paths.append(path_cache[terminal_node])

    return paths


def _node_membership(
    estimator,
    X_data: np.ndarray,
) -> np.ndarray:
    """Return a Boolean observation-by-node membership matrix."""
    return estimator.decision_path(
        X_data
    ).toarray().astype(bool)


def _node_outlier_enrichment_from_membership(
    tree,
    node_membership: np.ndarray,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Return child-node outlier-prevalence contrasts.

    For every internal node v,

        E(v) = |P(outlier | left child of v)
             - P(outlier | right child of v)|.

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
            outlier_rate_left
            - outlier_rate_right
        )

    return enrichment


def _eligible_node(
    node: int,
    feature: int,
    feature_types: FeatureTypes,
) -> bool:
    """Return whether a split node is eligible under AD-DIFFI."""
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
    """Return per-observation AD-DIFFI numerators and denominators.

    For an eligible node v on the path of observation x in tree t, the
    contribution is

        lambda(v) * E(v) / max(h_t(x), 1),

    where lambda(v) is the raw split-imbalance coefficient, E(v) is the
    absolute difference in model-defined outlier prevalence between the child
    nodes, and h_t(x) is the number of internal split nodes on the path.

    Continuous features are eligible at all internal split nodes. Binary
    features are eligible only when selected at the root node.
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
            feature = int(tree.feature[node])

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


def compute_enrichment_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute raw enrichment-based AD-DIFFI feature scores.

    This is the sole formal AD-DIFFI score used in the analysis.

    For each fitted tree, the method averages the feature-specific eligible
    node contributions

        lambda(v) * E(v) / max(h_t(x), 1)

    over feature-specific path occurrences. Here, lambda(v) is raw split
    imbalance, E(v) is node-level model-defined outlier enrichment, and
    h_t(x) is the observation-specific number of internal path nodes.

    Continuous features contribute at every internal split node where they
    are selected. Binary features contribute only when selected at the root
    node under the Root-Split-Only rule.

    Tree-specific feature means are averaged across the forest. Larger values
    indicate stronger structural attribution to split imbalance and
    model-defined outlier enrichment under the fitted Isolation Forest.

    Parameters
    ----------
    iforest:
        A fitted sklearn IsolationForest.

    X_data:
        Two-dimensional data array used for attribution.

    feature_types:
        Mapping from feature index to ``"cont"`` or ``"bin"``.

    anomaly_mask:
        Boolean model-defined outlier mask. In the standard workflow, use
        ``iforest.predict(X_data) == -1``.

    Returns
    -------
    np.ndarray
        One raw enrichment-based AD-DIFFI score per feature.
    """
    X_data, anomaly_mask = _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    if anomaly_mask is None:
        raise ValueError(
            "anomaly_mask is required."
        )

    n_features = X_data.shape[1]

    forest_importance = np.zeros(
        n_features,
        dtype=float,
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

        forest_importance += tree_scores

    return forest_importance / len(
        iforest.estimators_
    )
