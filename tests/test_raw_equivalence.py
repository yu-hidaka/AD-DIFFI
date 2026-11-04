from __future__ import annotations

import numpy as np
import pytest

from sklearn.ensemble import IsolationForest

from ad_diffi.core import (
    compute_enrichment_ad_diffi_fast,
    compute_enrichment_ad_diffi_reference,
    compute_raw_ad_diffi,
)


RTOL = 1e-10
ATOL = 1e-12


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def equivalence_case():
    """Return a fitted forest and mixed continuous-binary data."""
    rng = np.random.default_rng(123)

    X = rng.normal(
        size=(36, 4)
    )

    X[:, 2] = rng.binomial(
        1,
        0.25,
        len(X),
    )

    # A rare binary feature with one positive observation.
    X[:, 3] = 0.0
    X[0, 3] = 1.0

    feature_types = {
        0: "cont",
        1: "cont",
        2: "bin",
        3: "bin",
    }

    model = IsolationForest(
        n_estimators=12,
        max_samples=24,
        contamination=0.2,
        max_features=1.0,
        bootstrap=False,
        random_state=123,
    ).fit(X)

    return model, X, feature_types


# ============================================================
# Comparison helper
# ============================================================

def assert_raw_implementations_match(
    model,
    X,
    feature_types,
    anomaly_mask,
):
    """Compare the reference, fast implementation, and public API."""
    reference_scores = (
        compute_enrichment_ad_diffi_reference(
            iforest=model,
            X_data=X,
            feature_types=feature_types,
            anomaly_mask=anomaly_mask,
        )
    )

    fast_scores = compute_enrichment_ad_diffi_fast(
        iforest=model,
        X_data=X,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    public_scores = compute_raw_ad_diffi(
        iforest=model,
        X_data=X,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    expected_shape = (X.shape[1],)

    for scores in (
        reference_scores,
        fast_scores,
        public_scores,
    ):
        assert scores.shape == expected_shape
        assert np.isfinite(scores).all()
        assert np.all(scores >= 0.0)

    np.testing.assert_allclose(
        fast_scores,
        reference_scores,
        rtol=RTOL,
        atol=ATOL,
        equal_nan=False,
        err_msg=(
            "The fast implementation differs from "
            "the observation-wise reference."
        ),
    )

    np.testing.assert_allclose(
        public_scores,
        reference_scores,
        rtol=RTOL,
        atol=ATOL,
        equal_nan=False,
        err_msg=(
            "The public raw API differs from "
            "the observation-wise reference."
        ),
    )

    return reference_scores, fast_scores, public_scores


# ============================================================
# Observed and permuted data
# ============================================================

@pytest.mark.parametrize(
    "permuted_feature",
    [None, 0, 1, 2, 3],
)
@pytest.mark.parametrize(
    "mask_kind",
    [
        "model",
        "alternating",
        "all_inlier",
        "all_outlier",
    ],
)
def test_fast_matches_reference_on_observed_and_permuted_data(
    equivalence_case,
    permuted_feature,
    mask_kind,
):
    """Compare implementations using identical data and masks."""
    model, X, feature_types = equivalence_case

    X_eval = X.copy()

    if permuted_feature is not None:
        rng = np.random.default_rng(
            2000 + permuted_feature
        )

        X_eval[:, permuted_feature] = (
            rng.permutation(
                X[:, permuted_feature]
            )
        )

    if mask_kind == "model":
        anomaly_mask = model.predict(
            X_eval
        ) == -1

    elif mask_kind == "alternating":
        anomaly_mask = (
            np.arange(len(X_eval)) % 2 == 0
        )

    elif mask_kind == "all_inlier":
        anomaly_mask = np.zeros(
            len(X_eval),
            dtype=bool,
        )

    else:
        anomaly_mask = np.ones(
            len(X_eval),
            dtype=bool,
        )

    reference, fast, public = (
        assert_raw_implementations_match(
            model,
            X_eval,
            feature_types,
            anomaly_mask,
        )
    )

    if mask_kind in {
        "all_inlier",
        "all_outlier",
    }:
        expected = np.zeros(
            X.shape[1],
            dtype=float,
        )

        for scores in (
            reference,
            fast,
            public,
        ):
            np.testing.assert_array_equal(
                scores,
                expected,
            )


@pytest.mark.parametrize(
    "seed",
    [7, 42, 2026],
)
def test_fast_matches_reference_across_forest_seeds(
    seed,
):
    """Check agreement across independently fitted forests."""
    rng = np.random.default_rng(seed)

    X = rng.normal(
        size=(28, 4)
    )

    X[:, 2] = rng.binomial(
        1,
        0.2,
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

    model = IsolationForest(
        n_estimators=8,
        max_samples=20,
        contamination=0.2,
        max_features=1.0,
        bootstrap=False,
        random_state=seed,
    ).fit(X)

    anomaly_mask = model.predict(X) == -1

    assert_raw_implementations_match(
        model,
        X,
        feature_types,
        anomaly_mask,
    )


# ============================================================
# Edge cases
# ============================================================

def test_fast_matches_reference_with_empty_children(
    equivalence_case,
):
    """Repeated identical rows leave one child empty at each split."""
    model, X, feature_types = equivalence_case

    X_eval = np.repeat(
        X[:1],
        repeats=6,
        axis=0,
    )

    anomaly_mask = np.array(
        [
            False,
            True,
            False,
            True,
            False,
            True,
        ]
    )

    reference, fast, public = (
        assert_raw_implementations_match(
            model,
            X_eval,
            feature_types,
            anomaly_mask,
        )
    )

    expected = np.zeros(
        X.shape[1],
        dtype=float,
    )

    for scores in (
        reference,
        fast,
        public,
    ):
        np.testing.assert_array_equal(
            scores,
            expected,
        )


def test_fast_matches_reference_with_constant_binary_feature(
    equivalence_case,
):
    """A constant binary evaluation column remains supported."""
    model, X, feature_types = equivalence_case

    X_eval = X.copy()
    X_eval[:, 3] = 0.0

    anomaly_mask = (
        np.arange(len(X_eval)) % 2 == 0
    )

    reference, fast, public = (
        assert_raw_implementations_match(
            model,
            X_eval,
            feature_types,
            anomaly_mask,
        )
    )

    for scores in (
        reference,
        fast,
        public,
    ):
        assert scores[3] == 0.0


def test_binary_feature_without_eligible_root_has_zero_score(
    equivalence_case,
):
    """Binary features not used at the root receive zero scores."""
    _, X, feature_types = equivalence_case

    model = IsolationForest(
        n_estimators=1,
        max_samples=24,
        contamination=0.2,
        max_features=1.0,
        bootstrap=False,
        random_state=321,
    ).fit(X)

    root_feature = int(
        model.estimators_[0].tree_.feature[0]
    )

    target_feature = next(
        feature
        for feature in [2, 3]
        if feature != root_feature
    )

    anomaly_mask = (
        np.arange(len(X)) % 2 == 0
    )

    reference, fast, public = (
        assert_raw_implementations_match(
            model,
            X,
            feature_types,
            anomaly_mask,
        )
    )

    for scores in (
        reference,
        fast,
        public,
    ):
        assert scores[target_feature] == 0.0


def test_forests_without_internal_splits_return_zero():
    """A forest fitted on constant data has no eligible splits."""
    X = np.zeros(
        (12, 3),
        dtype=float,
    )

    feature_types = {
        0: "cont",
        1: "cont",
        2: "bin",
    }

    model = IsolationForest(
        n_estimators=4,
        max_samples=12,
        contamination=0.25,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
    ).fit(X)

    anomaly_mask = (
        np.arange(len(X)) % 2 == 0
    )

    reference, fast, public = (
        assert_raw_implementations_match(
            model,
            X,
            feature_types,
            anomaly_mask,
        )
    )

    expected = np.zeros(
        X.shape[1],
        dtype=float,
    )

    for scores in (
        reference,
        fast,
        public,
    ):
        np.testing.assert_array_equal(
            scores,
            expected,
        )


def test_numeric_mask_matches_boolean_mask(
    equivalence_case,
):
    """Numeric 0/1 masks preserve the same raw scores."""
    model, X, feature_types = equivalence_case

    boolean_mask = (
        np.arange(len(X)) % 2 == 0
    )

    numeric_mask = boolean_mask.astype(
        np.int64
    )

    boolean_scores = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        boolean_mask,
    )

    _, _, numeric_scores = (
        assert_raw_implementations_match(
            model,
            X,
            feature_types,
            numeric_mask,
        )
    )

    np.testing.assert_allclose(
        numeric_scores,
        boolean_scores,
        rtol=RTOL,
        atol=ATOL,
        equal_nan=False,
    )


# ============================================================
# Input and fitted-tree preservation
# ============================================================

def test_raw_scoring_does_not_modify_inputs_or_trees(
    equivalence_case,
):
    """Neither implementation may mutate data, masks, or tree arrays."""
    model, X, feature_types = equivalence_case

    anomaly_mask = (
        np.arange(len(X)) % 2 == 0
    )

    original_X = X.copy()
    original_mask = anomaly_mask.copy()
    original_feature_types = feature_types.copy()

    original_tree_arrays = [
        {
            "feature": estimator.tree_.feature.copy(),
            "threshold": estimator.tree_.threshold.copy(),
            "children_left": (
                estimator.tree_.children_left.copy()
            ),
            "children_right": (
                estimator.tree_.children_right.copy()
            ),
            "n_node_samples": (
                estimator.tree_.n_node_samples.copy()
            ),
        }
        for estimator in model.estimators_
    ]

    assert_raw_implementations_match(
        model,
        X,
        feature_types,
        anomaly_mask,
    )

    np.testing.assert_array_equal(
        X,
        original_X,
    )

    np.testing.assert_array_equal(
        anomaly_mask,
        original_mask,
    )

    assert feature_types == original_feature_types

    for estimator, original_arrays in zip(
        model.estimators_,
        original_tree_arrays,
    ):
        for array_name, expected in original_arrays.items():
            np.testing.assert_array_equal(
                getattr(
                    estimator.tree_,
                    array_name,
                ),
                expected,
            )


def test_repeated_raw_calls_are_reproducible(
    equivalence_case,
):
    """Repeated evaluations of identical inputs return identical scores."""
    model, X, feature_types = equivalence_case

    anomaly_mask = model.predict(X) == -1

    first = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        anomaly_mask,
    )

    second = compute_raw_ad_diffi(
        model,
        X,
        feature_types,
        anomaly_mask,
    )

    np.testing.assert_array_equal(
        second,
        first,
    )
