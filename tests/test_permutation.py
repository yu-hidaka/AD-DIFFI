from __future__ import annotations

import copy

import numpy as np
import pytest

from sklearn.ensemble import IsolationForest

import ad_diffi.permutation as permutation_module

from ad_diffi.permutation import (
    _collect_feature_reference,
    _resolve_reference_plan,
    apply_permutation_z,
    compute_hybrid_importance,
    estimate_permutation_baseline,
    get_model_anomaly_mask,
    summarize_permutation_reference,
)


@pytest.fixture
def small_case():
    rng = np.random.default_rng(7)

    X = np.column_stack(
        [
            rng.normal(size=12),
            np.zeros(12),
        ]
    )

    X[0, 1] = 1

    feature_types = {
        0: "cont",
        1: "bin",
    }

    model = IsolationForest(
        n_estimators=6,
        max_samples=12,
        contamination=0.25,
        max_features=1.0,
        bootstrap=False,
        random_state=7,
    ).fit(X)

    return model, X, feature_types


def make_stored_reference(
    scores,
    mode="monte_carlo",
):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    return {
        "mode": mode,
        "feature_index": 0,
        "n_distinct_assignments": None,
        "scores": scores,
        "outlier_counts": np.ones(
            len(scores),
            dtype=int,
        ),
        "elapsed_seconds": np.arange(
            1,
            len(scores) + 1,
            dtype=float,
        ),
    }


def test_reference_plan():
    singleton = np.array(
        [1, 0, 0, 0, 0]
    )

    assert _resolve_reference_plan(
        singleton,
        "bin",
        "hybrid",
        exact_limit=5,
    ) == ("exact", 5)

    assert _resolve_reference_plan(
        singleton,
        "bin",
        "hybrid",
        exact_limit=4,
    ) == ("monte_carlo", 5)

    assert _resolve_reference_plan(
        singleton,
        "bin",
        "monte_carlo",
        exact_limit=100,
    ) == ("monte_carlo", 5)

    assert _resolve_reference_plan(
        np.arange(5),
        "cont",
        "hybrid",
        exact_limit=100,
    ) == ("monte_carlo", None)


def test_constant_binary_has_one_assignment():
    assert _resolve_reference_plan(
        np.zeros(5),
        "bin",
        "hybrid",
        exact_limit=1,
    ) == ("exact", 1)


def test_exact_summary_uses_population_sd():
    reference = make_stored_reference(
        [0.0, 1.0, 2.0, 3.0],
        mode="exact",
    )

    summary = summarize_permutation_reference(
        reference
    )

    assert summary["sd_ddof"] == 0
    assert summary["n_evaluations"] == 4

    assert summary["null_mean"] == pytest.approx(
        1.5
    )

    assert summary["null_sd"] == pytest.approx(
        np.std(
            [0.0, 1.0, 2.0, 3.0],
            ddof=0,
        )
    )


def test_mc_summary_uses_sample_sd_and_prefix():
    reference = make_stored_reference(
        [0.0, 1.0, 2.0, 100.0]
    )

    summary = summarize_permutation_reference(
        reference,
        n_permutations=3,
    )

    assert summary["sd_ddof"] == 1
    assert summary["n_evaluations"] == 3
    assert summary["null_mean"] == pytest.approx(1.0)
    assert summary["null_sd"] == pytest.approx(1.0)


def test_exact_summary_does_not_truncate():
    reference = make_stored_reference(
        [0.0, 1.0, 2.0, 3.0],
        mode="exact",
    )

    summary = summarize_permutation_reference(
        reference,
        n_permutations=2,
    )

    assert summary["n_evaluations"] == 4


def test_identical_scores_have_exactly_zero_sd():
    reference = make_stored_reference(
        [0.1] * 10
    )

    summary = summarize_permutation_reference(
        reference
    )

    assert summary["null_sd"] == 0.0
    assert not summary["valid_null_sd"]


def test_single_assignment_exact_sd_is_zero():
    reference = make_stored_reference(
        [0.2],
        mode="exact",
    )

    summary = summarize_permutation_reference(
        reference
    )

    assert summary["null_sd"] == 0.0
    assert not summary["valid_null_sd"]


@pytest.mark.parametrize(
    "budget",
    [1, 5],
)
def test_invalid_mc_prefix_raises(budget):
    reference = make_stored_reference(
        [0.0, 1.0, 2.0, 3.0]
    )

    with pytest.raises(ValueError):
        summarize_permutation_reference(
            reference,
            n_permutations=budget,
        )


def test_nonfinite_reference_raises():
    reference = make_stored_reference(
        [0.0, np.nan]
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        summarize_permutation_reference(
            reference
        )


def test_minimum_sd_flags_but_does_not_modify_sd():
    reference = make_stored_reference(
        [0.0, 1.0, 2.0]
    )

    summary = summarize_permutation_reference(
        reference,
        minimum_null_sd=2.0,
    )

    assert summary["null_sd"] == pytest.approx(1.0)
    assert not summary["valid_null_sd"]


def test_z_standardization():
    baseline = {
        "null_mean": np.array([1.0, 2.0]),
        "null_sd": np.array([2.0, 0.5]),
        "minimum_null_sd": 0.0,
    }

    z = apply_permutation_z(
        np.array([5.0, 1.0]),
        baseline,
    )

    np.testing.assert_allclose(
        z,
        [2.0, -2.0],
    )


def test_zero_sd_is_not_regularized():
    baseline = {
        "null_mean": np.array([1.0, 2.0]),
        "null_sd": np.array([0.0, 1.0]),
    }

    z = apply_permutation_z(
        np.array([2.0, 3.0]),
        baseline,
        strict=False,
    )

    assert np.isnan(z[0])
    assert z[1] == pytest.approx(1.0)

    with pytest.raises(
        ValueError,
        match="undefined",
    ):
        apply_permutation_z(
            np.array([2.0, 3.0]),
            baseline,
            strict=True,
        )


def test_exact_enumeration_and_recomputed_masks(
    monkeypatch,
):
    class DummyModel:
        def predict(self, X):
            return np.where(
                X[:, 0] == 1,
                -1,
                1,
            )

    observed_columns = []

    def fake_raw(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    ):
        np.testing.assert_array_equal(
            anomaly_mask,
            X_data[:, 0] == 1,
        )

        # The non-target column must remain unchanged.
        np.testing.assert_array_equal(
            X_data[:, 1],
            np.arange(4),
        )

        observed_columns.append(
            X_data[:, 0].copy()
        )

        return np.array(
            [
                np.dot(
                    X_data[:, 0],
                    np.arange(4),
                ),
                0.0,
            ]
        )

    monkeypatch.setattr(
        permutation_module,
        "compute_raw_ad_diffi",
        fake_raw,
    )

    X = np.column_stack(
        [
            [1, 0, 0, 0],
            np.arange(4),
        ]
    )

    reference = _collect_feature_reference(
        iforest=DummyModel(),
        X_data=X,
        feature_types={0: "bin", 1: "cont"},
        feature_index=0,
        n_permutations=10,
        seed_sequence=np.random.SeedSequence(42),
        reference_mode="exact",
        n_distinct_assignments=4,
    )

    np.testing.assert_array_equal(
        reference["scores"],
        [0.0, 1.0, 2.0, 3.0],
    )

    assert len(
        {
            tuple(column)
            for column in observed_columns
        }
    ) == 4

    assert all(
        column.sum() == 1
        for column in observed_columns
    )


def test_mc_preserves_marginal_and_other_columns(
    monkeypatch,
):
    class DummyModel:
        def predict(self, X):
            return np.ones(
                len(X),
                dtype=int,
            )

    X = np.column_stack(
        [
            [1, 1, 0, 0, 0],
            np.arange(5),
        ]
    )

    call_count = 0

    def fake_raw(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    ):
        nonlocal call_count
        call_count += 1

        np.testing.assert_array_equal(
            np.sort(X_data[:, 0]),
            np.sort(X[:, 0]),
        )

        np.testing.assert_array_equal(
            X_data[:, 1],
            X[:, 1],
        )

        assert not anomaly_mask.any()

        return np.array(
            [
                np.dot(
                    X_data[:, 0],
                    np.arange(5),
                ),
                0.0,
            ]
        )

    monkeypatch.setattr(
        permutation_module,
        "compute_raw_ad_diffi",
        fake_raw,
    )

    _collect_feature_reference(
        iforest=DummyModel(),
        X_data=X,
        feature_types={0: "bin", 1: "cont"},
        feature_index=0,
        n_permutations=8,
        seed_sequence=np.random.SeedSequence(42),
        reference_mode="monte_carlo",
        n_distinct_assignments=10,
    )

    assert call_count == 8


def test_same_seed_reproduces_mc_prefix(small_case):
    model, X, feature_types = small_case

    short = estimate_permutation_baseline(
        model,
        X,
        feature_types,
        n_permutations=3,
        random_state=42,
        mode="monte_carlo",
    )

    long = estimate_permutation_baseline(
        model,
        X,
        feature_types,
        n_permutations=5,
        random_state=42,
        mode="monte_carlo",
    )

    for j in range(X.shape[1]):
        np.testing.assert_array_equal(
            short["references"][j]["scores"],
            long["references"][j]["scores"][:3],
        )

        np.testing.assert_array_equal(
            short["references"][j]["outlier_counts"],
            long["references"][j]["outlier_counts"][:3],
        )


def test_hybrid_branch_and_no_forest_mutation(small_case):
    model, X, feature_types = small_case

    original_X = X.copy()

    original_trees = [
        (
            estimator.tree_.feature.copy(),
            estimator.tree_.threshold.copy(),
            estimator.tree_.n_node_samples.copy(),
        )
        for estimator in model.estimators_
    ]

    baseline = estimate_permutation_baseline(
        model,
        X,
        feature_types,
        n_permutations=3,
        exact_limit=12,
        random_state=42,
        mode="hybrid",
    )

    assert baseline["references"][0]["mode"] == "monte_carlo"
    assert baseline["references"][1]["mode"] == "exact"

    assert len(
        baseline["references"][0]["scores"]
    ) == 3

    assert len(
        baseline["references"][1]["scores"]
    ) == 12

    np.testing.assert_array_equal(
        X,
        original_X,
    )

    for estimator, original in zip(
        model.estimators_,
        original_trees,
    ):
        np.testing.assert_array_equal(
            estimator.tree_.feature,
            original[0],
        )
        np.testing.assert_array_equal(
            estimator.tree_.threshold,
            original[1],
        )
        np.testing.assert_array_equal(
            estimator.tree_.n_node_samples,
            original[2],
        )


def test_exact_limit_zero_disables_enumeration(small_case):
    model, X, feature_types = small_case

    baseline = estimate_permutation_baseline(
        model,
        X,
        feature_types,
        n_permutations=3,
        exact_limit=0,
        random_state=42,
    )

    assert all(
        reference["mode"] == "monte_carlo"
        for reference in baseline["references"]
    )


def test_hybrid_diagnostics_for_constant_feature():
    X = np.column_stack(
        [
            np.arange(12, dtype=float),
            np.zeros(12),
        ]
    )

    model = IsolationForest(
        n_estimators=4,
        contamination=0.25,
        random_state=42,
    ).fit(X)

    z, diagnostics, baseline = compute_hybrid_importance(
        iforest=model,
        X_data=X,
        feature_types={0: "cont", 1: "bin"},
        feature_names=["continuous", "constant_binary"],
        n_permutations=3,
        exact_limit=100,
        random_state=42,
        strict=False,
    )

    row = diagnostics.loc[
        diagnostics["feature"] == "constant_binary"
    ].iloc[0]

    assert row["mode"] == "exact"
    assert row["n_evaluations"] == 1
    assert row["null_sd"] == 0.0
    assert not row["valid_z"]
    assert row["status"] == "constant_feature"
    assert np.isnan(z[1])

    assert len(
        baseline["references"][1]["scores"]
    ) == 1

    with pytest.raises(
        ValueError,
        match="undefined",
    ):
        compute_hybrid_importance(
            iforest=model,
            X_data=X,
            feature_types={0: "cont", 1: "bin"},
            n_permutations=3,
            random_state=42,
            strict=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_permutations": 1},
        {"exact_limit": -1},
        {"random_state": -1},
        {"mode": "invalid"},
        {"minimum_null_sd": -1.0},
    ],
)
def test_invalid_configuration_raises(
    small_case,
    kwargs,
):
    model, X, feature_types = small_case

    with pytest.raises(ValueError):
        estimate_permutation_baseline(
            model,
            X,
            feature_types,
            **kwargs,
        )


def test_outlier_mask_matches_predict(small_case):
    model, X, _ = small_case

    np.testing.assert_array_equal(
        get_model_anomaly_mask(model, X),
        model.predict(X) == -1,
    )
