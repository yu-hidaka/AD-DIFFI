from __future__ import annotations

import numpy as np
import pytest

from sklearn.ensemble import IsolationForest

from ad_diffi.core import (
    _eligible_node,
    _node_lambdas,
    _node_membership,
    _node_outlier_enrichment_from_membership,
    _paths_for_rows,
    calculate_ad_diffi,
    compute_enrichment_ad_diffi,
    compute_raw_ad_diffi,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(32, 4)
    )

    X[:, 2] = rng.binomial(
        1,
        0.3,
        len(X),
    )

    X[:, 3] = rng.binomial(
        1,
        0.5,
        len(X),
    )

    feature_types = {
        0: "cont",
        1: "cont",
        2: "bin",
        3: "bin",
    }

    return X, feature_types


@pytest.fixture
def fitted_case(sample_data):
    X, feature_types = sample_data

    model = IsolationForest(
        n_estimators=12,
        max_samples=24,
        contamination=0.2,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
    ).fit(X)

    mask = model.predict(X) == -1

    assert mask.any()
    assert not mask.all()

    return model, X, feature_types, mask


def test_public_aliases():
    assert compute_raw_ad_diffi is compute_enrichment_ad_diffi
    assert calculate_ad_diffi is compute_enrichment_ad_diffi


def test_raw_output_shape_and_range(fitted_case):
    model, X, feature_types, mask = fitted_case

    scores = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        mask,
    )

    assert scores.shape == (X.shape[1],)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0)

    # Each eligible contribution is lambda * enrichment / h.
    assert np.all(scores <= 1.0 + 1e-12)


@pytest.mark.parametrize(
    "all_outliers",
    [False, True],
)
def test_homogeneous_masks_return_zero(
    fitted_case,
    all_outliers,
):
    model, X, feature_types, _ = fitted_case

    mask = np.full(
        len(X),
        all_outliers,
        dtype=bool,
    )

    scores = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        mask,
    )

    np.testing.assert_array_equal(
        scores,
        np.zeros(X.shape[1]),
    )


def test_numeric_zero_one_mask_matches_boolean(fitted_case):
    model, X, feature_types, mask = fitted_case

    expected = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        mask,
    )

    actual = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        mask.astype(int),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "invalid_value",
    [2.0, -1.0, 0.5, np.nan, np.inf],
)
def test_invalid_mask_values_raise(
    fitted_case,
    invalid_value,
):
    model, X, feature_types, mask = fitted_case

    invalid_mask = mask.astype(float)
    invalid_mask[0] = invalid_value

    with pytest.raises(
        ValueError,
        match="0/1",
    ):
        compute_raw_ad_diffi(
            model,
            X,
            feature_types,
            invalid_mask,
        )


def test_invalid_mask_shape_raises(fitted_case):
    model, X, feature_types, mask = fitted_case

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        compute_raw_ad_diffi(
            model,
            X,
            feature_types,
            mask[:, None],
        )


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_nonfinite_data_raise(
    fitted_case,
    invalid_value,
):
    model, X, feature_types, mask = fitted_case

    invalid_X = X.copy()
    invalid_X[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        compute_raw_ad_diffi(
            model,
            invalid_X,
            feature_types,
            mask,
        )


def test_empty_data_raise(fitted_case):
    model, X, feature_types, _ = fitted_case

    with pytest.raises(
        ValueError,
        match="observations and features",
    ):
        compute_raw_ad_diffi(
            model,
            X[:0],
            feature_types,
            np.zeros(0, dtype=bool),
        )


def test_nonbinary_value_raises(fitted_case):
    model, X, feature_types, mask = fitted_case

    invalid_X = X.copy()
    invalid_X[0, 2] = 0.5

    with pytest.raises(
        ValueError,
        match="Binary feature",
    ):
        compute_raw_ad_diffi(
            model,
            invalid_X,
            feature_types,
            mask,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {0: "cont", 1: "cont", 2: "bin"},
        {
            0: "cont",
            1: "cont",
            2: "bin",
            3: "bin",
            4: "cont",
        },
        {
            0: "cont",
            1: "invalid",
            2: "bin",
            3: "bin",
        },
    ],
)
def test_invalid_feature_metadata_raise(
    fitted_case,
    metadata,
):
    model, X, _, mask = fitted_case

    with pytest.raises(ValueError):
        compute_raw_ad_diffi(
            model,
            X,
            metadata,
            mask,
        )


def test_unfitted_forest_raises(sample_data):
    X, feature_types = sample_data

    model = IsolationForest()
    mask = np.arange(len(X)) % 2 == 0

    with pytest.raises(
        ValueError,
        match="fitted",
    ):
        compute_raw_ad_diffi(
            model,
            X,
            feature_types,
            mask,
        )


def test_feature_subsampling_raises(sample_data):
    X, feature_types = sample_data

    model = IsolationForest(
        n_estimators=4,
        max_features=0.5,
        contamination=0.2,
        random_state=42,
    ).fit(X)

    mask = model.predict(X) == -1

    with pytest.raises(
        ValueError,
        match="max_features=1.0",
    ):
        compute_raw_ad_diffi(
            model,
            X,
            feature_types,
            mask,
        )


def test_bootstrap_raises(sample_data):
    X, feature_types = sample_data

    model = IsolationForest(
        n_estimators=4,
        bootstrap=True,
        contamination=0.2,
        random_state=42,
    ).fit(X)

    mask = model.predict(X) == -1

    with pytest.raises(
        ValueError,
        match="bootstrap=False",
    ):
        compute_raw_ad_diffi(
            model,
            X,
            feature_types,
            mask,
        )


def test_feature_count_mismatch_raises(fitted_case):
    model, X, _, mask = fitted_case

    with pytest.raises(
        ValueError,
        match="feature counts",
    ):
        compute_raw_ad_diffi(
            model,
            X[:, :2],
            {0: "cont", 1: "cont"},
            mask,
        )


def test_eligibility_rule():
    types = {
        0: "cont",
        1: "bin",
    }

    assert _eligible_node(0, 0, types)
    assert _eligible_node(5, 0, types)
    assert _eligible_node(0, 1, types)
    assert not _eligible_node(5, 1, types)


def test_node_lambdas_match_training_counts(fitted_case):
    model, _, _, _ = fitted_case
    tree = model.estimators_[0].tree_

    lambdas = _node_lambdas(tree)

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            assert lambdas[node] == 0
        else:
            expected = 1.0 - (
                min(
                    tree.n_node_samples[left],
                    tree.n_node_samples[right],
                )
                / tree.n_node_samples[node]
            )

            assert lambdas[node] == pytest.approx(
                expected
            )


def test_paths_match_membership(fitted_case):
    model, X, _, _ = fitted_case
    estimator = model.estimators_[0]

    paths = _paths_for_rows(
        estimator,
        X,
    )

    membership = _node_membership(
        estimator,
        X,
    )

    leaves = estimator.apply(X)

    for row, path in enumerate(paths):
        assert path[0] == 0
        assert path[-1] == leaves[row]

        assert set(path) == set(
            np.flatnonzero(membership[row])
        )


def test_empty_child_enrichment_is_zero(fitted_case):
    model, X, _, _ = fitted_case
    estimator = model.estimators_[0]

    # Repeated identical rows follow one branch at every split.
    repeated_X = np.repeat(
        X[:1],
        repeats=4,
        axis=0,
    )

    membership = _node_membership(
        estimator,
        repeated_X,
    )

    mask = np.array(
        [False, True, False, True]
    )

    enrichment = (
        _node_outlier_enrichment_from_membership(
            estimator.tree_,
            membership,
            mask,
        )
    )

    np.testing.assert_array_equal(
        enrichment,
        np.zeros(estimator.tree_.node_count),
    )
