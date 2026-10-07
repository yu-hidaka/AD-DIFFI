"""
Original DIFFI baseline implementation.

This module contains only the original DIFFI implementation.
AD-DIFFI, RSO, and Z-score normalization are implemented elsewhere.

The implementation follows the public DIFFI code:
- tree-wise in-bag observations;
- tree-wise outlier/inlier separation;
- IIC-adjusted path contributions;
- separate outlier and inlier cumulative feature importance;
- global feature importance as outlier contribution divided by inlier
  contribution.
"""

from __future__ import annotations

import time
from math import ceil
from typing import Any, Callable, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


def _get_iic(
    estimator: Any,
    predictions: np.ndarray,
    is_leaves: np.ndarray,
    adjust_iic: bool = True,
) -> np.ndarray:
    """
    Compute the Induced Imbalance Coefficient (IIC) for one isolation tree.
    """
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0

    tree = estimator.tree_

    n_nodes = tree.node_count
    lambda_values = np.zeros(
        n_nodes,
        dtype=float,
    )

    children_left = tree.children_left
    children_right = tree.children_right

    if predictions.shape[0] == 0:
        return lambda_values

    node_indicator = (
        estimator
        .decision_path(predictions)
        .toarray()
    )

    n_samples_node = np.sum(
        node_indicator,
        axis=0,
    )

    for node in range(n_nodes):
        n_current = n_samples_node[node]

        if (
            n_current <= 1
            or is_leaves[node]
        ):
            lambda_values[node] = -1.0
            continue

        n_left = n_samples_node[
            children_left[node]
        ]

        n_right = n_samples_node[
            children_right[node]
        ]

        if n_left == 0 or n_right == 0:
            lambda_values[node] = epsilon
            continue

        current_min = (
            0.5
            if n_current % 2 == 0
            else ceil(n_current / 2)
            / n_current
        )

        current_max = (
            n_current - 1
        ) / n_current

        imbalance = max(
            n_left,
            n_right,
        ) / n_current

        if (
            adjust_iic
            and current_min != current_max
        ):
            lambda_values[node] = (
                (imbalance - current_min)
                / (current_max - current_min)
                * (desired_max - desired_min)
                + desired_min
            )
        else:
            lambda_values[node] = imbalance

    return lambda_values


def _tree_depths_and_leaves(
    estimator: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate node depths and leaf indicators for one isolation tree.
    """
    tree = estimator.tree_

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(
        n_nodes,
        dtype=np.int64,
    )

    is_leaves = np.zeros(
        n_nodes,
        dtype=bool,
    )

    stack = [(0, -1)]

    while stack:
        node_id, parent_depth = stack.pop()

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

    return node_depth, is_leaves


def _default_tree_score(
    iforest: IsolationForest,
    tree_index: int,
    X: np.ndarray,
) -> np.ndarray:
    """
    Fallback tree-level score.

    The public DIFFI implementation uses a project-specific
    decision_function_single_tree function. This fallback uses the
    per-tree path depth only and is intended for compatibility checks,
    not as a replacement when the original helper is available.
    """
    estimator = iforest.estimators_[
        tree_index
    ]

    node_indicator = (
        estimator
        .decision_path(X)
        .toarray()
    )

    leaf_nodes = np.argmax(
        node_indicator,
        axis=1,
    )

    node_depth, _ = _tree_depths_and_leaves(
        estimator
    )

    depths = node_depth[leaf_nodes]

    return -depths.astype(float)


def _get_tree_score_function(
    tree_score_function: Optional[
        Callable[
            [
                IsolationForest,
                int,
                np.ndarray,
            ],
            np.ndarray,
        ]
    ],
):
    if tree_score_function is not None:
        return tree_score_function

    return _default_tree_score


def _accumulate_path_contributions(
    estimator: Any,
    X_subset: np.ndarray,
    is_leaves: np.ndarray,
    node_depth: np.ndarray,
    cfi: np.ndarray,
    counters: np.ndarray,
    adjust_iic: bool,
) -> None:
    """
    Accumulate IIC-adjusted path contributions for one observation subset.
    """
    if X_subset.shape[0] == 0:
        return

    tree = estimator.tree_
    split_features = tree.feature

    lambda_values = _get_iic(
        estimator=estimator,
        predictions=X_subset,
        is_leaves=is_leaves.copy(),
        adjust_iic=adjust_iic,
    )

    node_indicator = (
        estimator
        .decision_path(X_subset)
        .toarray()
    )

    for row in range(X_subset.shape[0]):
        path = np.where(
            node_indicator[row] == 1
        )[0]

        if len(path) == 0:
            continue

        leaf_depth = node_depth[
            path[-1]
        ]

        if leaf_depth <= 0:
            continue

        for node in path:
            feature_index = split_features[node]
            lambda_value = lambda_values[node]

            if feature_index < 0:
                continue

            if lambda_value == -1:
                continue

            cfi[feature_index] += (
                lambda_value / leaf_depth
            )

            counters[feature_index] += 1


def original_diffi_importance(
    iforest: IsolationForest,
    X: np.ndarray,
    adjust_iic: bool = True,
    tree_score_function: Optional[
        Callable[
            [
                IsolationForest,
                int,
                np.ndarray,
            ],
            np.ndarray,
        ]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate Original DIFFI global feature importance.

    Parameters
    ----------
    iforest:
        A fitted sklearn IsolationForest.

    X:
        Data matrix used by the fitted forest.

    adjust_iic:
        Whether to apply IIC adjustment.

    tree_score_function:
        Function used to calculate tree-level outlier/inlier scores.
        The expected signature is:

            tree_score_function(iforest, tree_index, X)

        If omitted, a path-depth fallback is used. For exact reproduction
        of the public DIFFI implementation, pass the original
        decision_function_single_tree function.

    Returns
    -------
    fi:
        Global Original DIFFI feature-importance ratio.

    cfi_outliers:
        Raw cumulative feature contributions from outliers.

    cfi_inliers:
        Raw cumulative feature contributions from inliers.

    elapsed_time:
        Runtime in seconds.
    """
    start_time = time.time()

    X = np.asarray(
        X,
        dtype=float,
    )

    n_features = X.shape[1]

    estimators = iforest.estimators_
    in_bag_samples = (
        iforest.estimators_samples_
    )

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

    tree_score = _get_tree_score_function(
        tree_score_function
    )

    for tree_index, estimator in enumerate(
        estimators
    ):
        in_bag_indices = list(
            in_bag_samples[tree_index]
        )

        X_in_bag = X[in_bag_indices]

        scores_in_bag = tree_score(
            iforest,
            tree_index,
            X_in_bag,
        )

        X_outliers = X_in_bag[
            scores_in_bag < 0
        ]

        X_inliers = X_in_bag[
            scores_in_bag > 0
        ]

        if (
            X_outliers.shape[0] == 0
            or X_inliers.shape[0] == 0
        ):
            continue

        node_depth, is_leaves = (
            _tree_depths_and_leaves(
                estimator
            )
        )

        _accumulate_path_contributions(
            estimator=estimator,
            X_subset=X_outliers,
            is_leaves=is_leaves,
            node_depth=node_depth,
            cfi=cfi_outliers,
            counters=counter_outliers,
            adjust_iic=adjust_iic,
        )

        _accumulate_path_contributions(
            estimator=estimator,
            X_subset=X_inliers,
            is_leaves=is_leaves,
            node_depth=node_depth,
            cfi=cfi_inliers,
            counters=counter_inliers,
            adjust_iic=adjust_iic,
        )

    fi_outliers = np.divide(
        cfi_outliers,
        counter_outliers,
        out=np.zeros_like(
            cfi_outliers
        ),
        where=counter_outliers > 0,
    )

    fi_inliers = np.divide(
        cfi_inliers,
        counter_inliers,
        out=np.zeros_like(
            cfi_inliers
        ),
        where=counter_inliers > 0,
    )

    fi = np.divide(
        fi_outliers,
        fi_inliers,
        out=np.zeros_like(
            fi_outliers
        ),
        where=fi_inliers != 0,
    )

    elapsed_time = (
        time.time() - start_time
    )

    return (
        fi,
        cfi_outliers,
        cfi_inliers,
        elapsed_time,
    )


def diffi_ib(
    iforest: IsolationForest,
    X: np.ndarray,
    adjust_iic: bool = True,
    tree_score_function: Optional[
        Callable[
            [
                IsolationForest,
                int,
                np.ndarray,
            ],
            np.ndarray,
        ]
    ] = None,
) -> Tuple[np.ndarray, float]:
    """
    Backward-compatible public API.

    Returns
    -------
    fi:
        Original DIFFI global feature importance.

    elapsed_time:
        Runtime in seconds.
    """
    fi, _, _, elapsed_time = (
        original_diffi_importance(
            iforest=iforest,
            X=X,
            adjust_iic=adjust_iic,
            tree_score_function=tree_score_function,
        )
    )

    return fi, elapsed_time


def local_diffi(
    iforest: IsolationForest,
    x: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Calculate local DIFFI feature importance for one observation.
    """
    start_time = time.time()

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(1, -1)

    n_features = x.shape[1]

    cfi = np.zeros(
        n_features,
        dtype=float,
    )

    counters = np.zeros(
        n_features,
        dtype=int,
    )

    max_samples = iforest.max_samples_

    max_depth = int(
        np.ceil(
            np.log2(max_samples)
        )
    )

    for estimator in iforest.estimators_:
        node_depth, is_leaves = (
            _tree_depths_and_leaves(
                estimator
            )
        )

        node_indicator = (
            estimator
            .decision_path(x)
            .toarray()
        )

        path = np.where(
            node_indicator[0] == 1
        )[0]

        if len(path) == 0:
            continue

        leaf_depth = node_depth[
            path[-1]
        ]

        if leaf_depth <= 0:
            continue

        for node in path:
            if is_leaves[node]:
                continue

            feature_index = (
                estimator.tree_.feature[node]
            )

            if feature_index < 0:
                continue

            contribution = (
                1.0 / leaf_depth
                - 1.0 / max_depth
            )

            cfi[feature_index] += contribution
            counters[feature_index] += 1

    fi = np.divide(
        cfi,
        counters,
        out=np.zeros_like(cfi),
        where=counters > 0,
    )

    elapsed_time = (
        time.time() - start_time
    )

    return fi, elapsed_time
