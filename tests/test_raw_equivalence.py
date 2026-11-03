from __future__ import annotations

import numpy as np
import pytest

from sklearn.ensemble import IsolationForest

from ad_diffi.core import compute_raw_ad_diffi


def sparse_raw_reference(
    model,
    X,
    feature_types,
    anomaly_mask,
):
    """Independent sparse aggregation for equivalence testing.

    This function is a test oracle, not a public production API.
    It deliberately does not use core.py's contribution helpers.
    """
    X = np.asarray(X, dtype=float)
    anomaly_mask = np.asarray(
        anomaly_mask,
        dtype=bool,
    )

    n_features = X.shape[1]

    total_scores = np.zeros(
        n_features,
        dtype=float,
    )

    valid_tree_counts = np.zeros(
        n_features,
        dtype=int,
    )

    for estimator in model.estimators_:
        tree = estimator.tree_

        left = tree.children_left
        right = tree.children_right

        internal = np.flatnonzero(
            left != right
        )

        if internal.size == 0:
            continue

        membership = (
            estimator.decision_path(X).tocsr()
        )

        occupancy = np.asarray(
            membership.sum(axis=0)
        ).ravel()

        outlier_counts = np.asarray(
            membership.T
            @ anomaly_mask.astype(float)
        ).ravel()

        path_lengths = np.maximum(
            np.diff(membership.indptr) - 1,
            1,
        )

        weighted_occupancy = np.asarray(
            membership.T
            @ (1.0 / path_lengths)
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

        left_rate = np.divide(
            outlier_counts[left_nodes],
            left_n,
            out=np.zeros(
                internal.size,
                dtype=float,
            ),
            where=left_n > 0,
        )

        right_rate = np.divide(
            outlier_counts[right_nodes],
            right_n,
            out=np.zeros(
                internal.size,
                dtype=float,
            ),
            where=right_n > 0,
        )

        enrichment = np.abs(
            left_rate - right_rate
        )

        enrichment[
            (left_n == 0) | (right_n == 0)
        ] = 0.0

        split_features = tree.feature[internal]

        eligible = np.asarray(
            [
                feature_types[int(feature)] == "cont"
                or node == 0
                for node, feature in zip(
                    internal,
                    split_features,
                )
            ],
            dtype=bool,
        )

        nodes = internal[eligible]
        features = split_features[eligible]

        numerator = np.bincount(
            features,
            weights=(
                lambdas[eligible]
                * enrichment[eligible]
                * weighted_occupancy[nodes]
            ),
            minlength=n_features,
        )

        denominator = np.bincount(
            features,
            weights=occupancy[nodes],
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

        total_scores[valid] += tree_scores[valid]
        valid_tree_counts[valid] += 1

    return np.divide(
        total_scores,
        valid_tree_counts,
        out=np.zeros(
            n_features,
            dtype=float,
        ),
        where=valid_tree_counts > 0,
    )


@pytest.fixture
def equivalence_case():
    rng = np.random.default_rng(123)

    X = rng.normal(
        size=(36, 4)
    )

    X[:, 2] = rng.binomial(
        1,
        0.25,
        len(X),
    )

    # Include a rare binary feature with one positive observation.
    X[:, 3] = 0
    X[0, 3] = 1

    feature_types = {
        0: "cont",
        1: "cont",
        2: "bin",
        3: "bin",
    }

    model = IsolationForest(
        n_estimators=16,
        max_samples=24,
        contamination=0.2,
        max_features=1.0,
        bootstrap=False,
        random_state=123,
    ).fit(X)

    return model, X, feature_types


@pytest.mark.parametrize(
    "permuted_feature",
    [None, 0, 1, 2, 3],
)
def test_sparse_and_core_match(
    equivalence_case,
    permuted_feature,
):
    model, X, feature_types = equivalence_case

    X_eval = X.copy()

    if permuted_feature is not None:
        rng = np.random.default_rng(
            1000 + permuted_feature
        )

        X_eval[:, permuted_feature] = (
            rng.permutation(
                X[:, permuted_feature]
            )
        )

    mask = model.predict(X_eval) == -1

    expected = compute_raw_ad_diffi(
        model,
        X_eval,
        feature_types,
        mask,
    )

    actual = sparse_raw_reference(
        model,
        X_eval,
        feature_types,
        mask,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mask_kind",
    ["all_inlier", "all_outlier", "alternating"],
)
def test_sparse_and_core_match_for_supplied_masks(
    equivalence_case,
    mask_kind,
):
    model, X, feature_types = equivalence_case

    if mask_kind == "all_inlier":
        mask = np.zeros(
            len(X),
            dtype=bool,
        )
    elif mask_kind == "all_outlier":
        mask = np.ones(
            len(X),
            dtype=bool,
        )
    else:
        mask = np.arange(len(X)) % 2 == 0

    expected = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        mask,
    )

    actual = sparse_raw_reference(
        model,
        X,
        feature_types,
        mask,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_sparse_and_core_match_with_empty_children(
    equivalence_case,
):
    model, X, feature_types = equivalence_case

    X_eval = np.repeat(
        X[:1],
        repeats=6,
        axis=0,
    )

    mask = np.array(
        [False, True, False, True, False, True]
    )

    expected = compute_raw_ad_diffi(
        model,
        X_eval,
        feature_types,
        mask,
    )

    actual = sparse_raw_reference(
        model,
        X_eval,
        feature_types,
        mask,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        expected,
        np.zeros(X.shape[1]),
    )


def test_binary_without_eligible_root_has_zero_score(
    equivalence_case,
):
    model, X, _ = equivalence_case

    root_features = {
        int(estimator.tree_.feature[0])
        for estimator in model.estimators_
    }

    # Build a single-tree forest view by retaining one estimator.
    # All other features then have no eligible root in that view.
    import copy

    single_tree_model = copy.copy(model)
    single_tree_model.estimators_ = [
        model.estimators_[0]
    ]
    single_tree_model.estimators_features_ = [
        model.estimators_features_[0]
    ]

    root_feature = int(
        single_tree_model.estimators_[0]
        .tree_.feature[0]
    )

    target = next(
        index
        for index in [2, 3]
        if index != root_feature
    )

    feature_types = {
        0: "cont",
        1: "cont",
        2: "bin",
        3: "bin",
    }

    mask = np.arange(len(X)) % 2 == 0

    score = compute_raw_ad_diffi(
        single_tree_model,
        X,
        feature_types,
        mask,
    )

    assert score[target] == 0
