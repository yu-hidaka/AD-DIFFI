from __future__ import annotations

from itertools import combinations
from math import comb
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .core import (
    FeatureTypes,
    _validate_inputs,
    compute_raw_ad_diffi,
)


ReferenceResult = Dict[str, Any]
PermutationBaseline = Dict[str, Any]

_VALID_MODES = {
    "hybrid",
    "monte_carlo",
}


# ============================================================
# Validation helpers
# ============================================================

def _validate_integer(
    value: int,
    name: str,
    minimum: int,
) -> int:
    """Validate and return an integer configuration value."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be an integer, not Boolean."
        )

    if not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer."
        )

    value = int(value)

    if value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    return value


def _validate_minimum_null_sd(
    minimum_null_sd: float,
) -> float:
    """Validate the threshold used to flag undefined Z-scores."""
    value = float(minimum_null_sd)

    if not np.isfinite(value) or value < 0:
        raise ValueError(
            "minimum_null_sd must be finite and nonnegative."
        )

    return value


def _validate_feature_names(
    feature_names: Optional[Sequence[str]],
    n_features: int,
) -> List[str]:
    """Return complete, unique feature names."""
    if feature_names is None:
        return [
            f"feature_{index}"
            for index in range(n_features)
        ]

    names = list(feature_names)

    if len(names) != n_features:
        raise ValueError(
            "feature_names must match the number of columns."
        )

    if not all(
        isinstance(name, str)
        for name in names
    ):
        raise ValueError(
            "feature_names must contain strings."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "feature_names must be unique."
        )

    return names


def _validate_score_vector(
    scores: np.ndarray,
    n_features: int,
    name: str,
) -> np.ndarray:
    """Validate a finite feature-score vector."""
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if scores.shape != (n_features,):
        raise ValueError(
            f"{name} must have shape ({n_features},)."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return scores


# ============================================================
# Model-defined outlier labels
# ============================================================

def get_model_anomaly_mask(
    iforest: IsolationForest,
    X_data: np.ndarray,
) -> np.ndarray:
    """Return the fitted model's outlier mask without modification.

    Homogeneous masks are preserved. No artificial outliers or
    inliers are introduced.
    """
    return np.asarray(
        iforest.predict(X_data) == -1,
        dtype=bool,
    )


# ============================================================
# Reference planning
# ============================================================

def _resolve_reference_plan(
    column: np.ndarray,
    feature_type: str,
    mode: str,
    exact_limit: int,
) -> Tuple[str, Optional[int]]:
    """Choose exact enumeration or Monte Carlo permutation.

    For a binary column containing m ones among n observations,
    the number of distinct assignments is C(n, m).

    Continuous features always use Monte Carlo permutation.
    """
    n_distinct_assignments: Optional[int] = None

    if feature_type == "bin":
        n_samples = len(column)
        n_ones = int(np.count_nonzero(column == 1))

        n_distinct_assignments = comb(
            n_samples,
            n_ones,
        )

        if (
            mode == "hybrid"
            and n_distinct_assignments <= exact_limit
        ):
            return (
                "exact",
                n_distinct_assignments,
            )

    return (
        "monte_carlo",
        n_distinct_assignments,
    )


# ============================================================
# Feature-specific reference generation
# ============================================================

def _collect_feature_reference(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    feature_index: int,
    n_permutations: int,
    seed_sequence: np.random.SeedSequence,
    reference_mode: str,
    n_distinct_assignments: Optional[int],
) -> ReferenceResult:
    """Evaluate one feature's fixed-forest reference distribution.

    Only the target feature is rearranged. All other columns and the
    fitted forest remain unchanged.

    Outlier labels are recomputed after every rearrangement. Only the
    target feature's raw score enters its reference distribution.
    """
    n_samples = X_data.shape[0]
    column = X_data[:, feature_index]

    if reference_mode == "exact":
        if n_distinct_assignments is None:
            raise ValueError(
                "Exact enumeration requires an assignment count."
            )

        n_evaluations = n_distinct_assignments

        n_ones = int(
            np.count_nonzero(column == 1)
        )
        n_zeros = n_samples - n_ones

        minority_value = (
            1.0
            if n_ones <= n_zeros
            else 0.0
        )

        majority_value = 1.0 - minority_value

        minority_count = min(
            n_ones,
            n_zeros,
        )

        arrangements = combinations(
            range(n_samples),
            minority_count,
        )

        rng = None

    else:
        n_evaluations = n_permutations
        arrangements = range(n_permutations)

        rng = np.random.default_rng(
            seed_sequence
        )

        minority_value = None
        majority_value = None

    null_scores = np.empty(
        n_evaluations,
        dtype=float,
    )

    outlier_counts = np.empty(
        n_evaluations,
        dtype=int,
    )

    elapsed_seconds = np.empty(
        n_evaluations,
        dtype=float,
    )

    start = perf_counter()

    for evaluation_index, arrangement in enumerate(
        arrangements
    ):
        X_permuted = X_data.copy()

        if reference_mode == "exact":
            X_permuted[:, feature_index] = majority_value

            if len(arrangement) > 0:
                X_permuted[
                    list(arrangement),
                    feature_index,
                ] = minority_value

        else:
            if rng is None:
                raise RuntimeError(
                    "Monte Carlo RNG was not initialized."
                )

            X_permuted[:, feature_index] = (
                rng.permutation(column)
            )

        anomaly_mask = get_model_anomaly_mask(
            iforest,
            X_permuted,
        )

        raw_scores = compute_raw_ad_diffi(
            iforest=iforest,
            X_data=X_permuted,
            feature_types=feature_types,
            anomaly_mask=anomaly_mask,
        )

        raw_scores = _validate_score_vector(
            raw_scores,
            X_data.shape[1],
            "permuted raw scores",
        )

        null_scores[evaluation_index] = (
            raw_scores[feature_index]
        )

        outlier_counts[evaluation_index] = int(
            anomaly_mask.sum()
        )

        elapsed_seconds[evaluation_index] = (
            perf_counter() - start
        )

    return {
        "mode": reference_mode,
        "feature_index": feature_index,
        "n_distinct_assignments": n_distinct_assignments,
        "scores": null_scores,
        "outlier_counts": outlier_counts,
        "elapsed_seconds": elapsed_seconds,
    }


# ============================================================
# Reference summaries
# ============================================================

def summarize_permutation_reference(
    reference: ReferenceResult,
    n_permutations: Optional[int] = None,
    minimum_null_sd: float = 0.0,
) -> Dict[str, Any]:
    """Summarize a complete reference or an MC prefix.

    Exact references always use every enumerated assignment.

    For a Monte Carlo reference, n_permutations may select a prefix.
    This permits comparisons of candidate budgets without recomputing
    the same initial permutations.

    Exact SD uses ddof=0 because all distinct assignments are included.
    Monte Carlo SD uses ddof=1 to estimate reference variance.
    """
    minimum_null_sd = _validate_minimum_null_sd(
        minimum_null_sd
    )

    reference_mode = reference["mode"]

    if reference_mode not in {
        "exact",
        "monte_carlo",
    }:
        raise ValueError(
            "Invalid reference mode."
        )

    scores = np.asarray(
        reference["scores"],
        dtype=float,
    )

    outlier_counts = np.asarray(
        reference["outlier_counts"],
        dtype=int,
    )

    elapsed_seconds = np.asarray(
        reference["elapsed_seconds"],
        dtype=float,
    )

    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            "Reference scores must be a nonempty 1D array."
        )

    if (
        outlier_counts.shape != scores.shape
        or elapsed_seconds.shape != scores.shape
    ):
        raise ValueError(
            "Inconsistent reference array shapes."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Reference scores contain non-finite values."
        )

    if reference_mode == "exact":
        n_used = scores.size
        ddof = 0

    else:
        if n_permutations is None:
            n_used = scores.size
        else:
            n_used = _validate_integer(
                n_permutations,
                "n_permutations",
                minimum=2,
            )

        if n_used < 2:
            raise ValueError(
                "Monte Carlo summaries require at least two scores."
            )

        if n_used > scores.size:
            raise ValueError(
                "Requested prefix exceeds the stored reference."
            )

        ddof = 1

    selected_scores = scores[:n_used]

    null_mean = float(
        selected_scores.mean()
    )

    # Identical stored values define zero empirical variation.
    # Avoid tiny numerical SDs caused by floating-point summation.
    if np.all(
        selected_scores == selected_scores[0]
    ):
        null_sd = 0.0
    else:
        null_sd = float(
            selected_scores.std(
                ddof=ddof,
            )
        )

    valid_null_sd = bool(
        np.isfinite(null_sd)
        and null_sd > minimum_null_sd
    )

    return {
        "mode": reference_mode,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "valid_null_sd": valid_null_sd,
        "sd_ddof": ddof,
        "n_evaluations": int(n_used),
        "permuted_outlier_count_min": int(
            outlier_counts[:n_used].min()
        ),
        "permuted_outlier_count_max": int(
            outlier_counts[:n_used].max()
        ),
        "elapsed_seconds": float(
            elapsed_seconds[n_used - 1]
        ),
    }


# ============================================================
# Public reference API
# ============================================================

def estimate_permutation_baseline(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    n_permutations: int = 200,
    exact_limit: int = 10000,
    random_state: int = 42,
    mode: str = "hybrid",
    minimum_null_sd: float = 0.0,
) -> PermutationBaseline:
    """Estimate feature-specific fixed-forest reference distributions.

    Parameters
    ----------
    iforest
        A fitted IsolationForest supported by ad_diffi.core.
    X_data
        Evaluation data with shape (n_samples, n_features).
    feature_types
        Complete mapping from feature indices to "cont" or "bin".
    n_permutations
        Monte Carlo budget. Ignored for exact-enumeration features.
        The default of 200 is provisional, not a validated guarantee.
    exact_limit
        Maximum number of binary assignments to enumerate.
        Setting this to zero disables exact enumeration.
    random_state
        Nonnegative integer used to initialize feature-specific
        random streams.
    mode
        "hybrid": enumerate eligible binary references, otherwise MC.
        "monte_carlo": use MC for every feature.
    minimum_null_sd
        SDs at or below this threshold are flagged as undefined.
        The threshold does not replace or modify the denominator.

    Returns
    -------
    dict
        Per-feature means, SDs, validity flags, summaries, and complete
        reference arrays.

    Notes
    -----
    Feature-specific random streams are independent of the selected
    Monte Carlo budget. With identical inputs and random_state, a
    larger budget extends each feature's previous MC sequence.

    MC draws may repeat the same assignment and may include the
    observed assignment. Exact enumeration includes every distinct
    assignment once.

    No outlier labels are fixed across rearrangements.
    """
    n_permutations = _validate_integer(
        n_permutations,
        "n_permutations",
        minimum=2,
    )

    exact_limit = _validate_integer(
        exact_limit,
        "exact_limit",
        minimum=0,
    )

    random_state = _validate_integer(
        random_state,
        "random_state",
        minimum=0,
    )

    minimum_null_sd = _validate_minimum_null_sd(
        minimum_null_sd
    )

    if mode not in _VALID_MODES:
        raise ValueError(
            "mode must be 'hybrid' or 'monte_carlo'."
        )

    X_data, _ = _validate_inputs(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
    )

    n_features = X_data.shape[1]

    feature_seeds = np.random.SeedSequence(
        random_state
    ).spawn(n_features)

    references: List[ReferenceResult] = []
    summaries: List[Dict[str, Any]] = []

    for feature_index in range(n_features):
        reference_mode, n_assignments = (
            _resolve_reference_plan(
                column=X_data[:, feature_index],
                feature_type=feature_types[feature_index],
                mode=mode,
                exact_limit=exact_limit,
            )
        )

        reference = _collect_feature_reference(
            iforest=iforest,
            X_data=X_data,
            feature_types=feature_types,
            feature_index=feature_index,
            n_permutations=n_permutations,
            seed_sequence=feature_seeds[feature_index],
            reference_mode=reference_mode,
            n_distinct_assignments=n_assignments,
        )

        summary = summarize_permutation_reference(
            reference=reference,
            minimum_null_sd=minimum_null_sd,
        )

        references.append(
            reference
        )

        summaries.append(
            summary
        )

    return {
        "null_mean": np.asarray(
            [
                item["null_mean"]
                for item in summaries
            ],
            dtype=float,
        ),
        "null_sd": np.asarray(
            [
                item["null_sd"]
                for item in summaries
            ],
            dtype=float,
        ),
        "valid_null_sd": np.asarray(
            [
                item["valid_null_sd"]
                for item in summaries
            ],
            dtype=bool,
        ),
        "references": references,
        "summaries": summaries,
        "mode": mode,
        "n_permutations": n_permutations,
        "exact_limit": exact_limit,
        "random_state": random_state,
        "minimum_null_sd": minimum_null_sd,
    }


# ============================================================
# Z standardization
# ============================================================

def apply_permutation_z(
    raw_scores: np.ndarray,
    baseline: PermutationBaseline,
    minimum_null_sd: Optional[float] = None,
    strict: bool = True,
) -> np.ndarray:
    """Standardize observed raw scores against feature references.

    Undefined SDs produce NaN. If strict=True, any undefined SD raises
    ValueError instead.

    No epsilon is added and no SD is replaced with one.
    """
    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if raw_scores.ndim != 1:
        raise ValueError(
            "raw_scores must be one-dimensional."
        )

    raw_scores = _validate_score_vector(
        raw_scores,
        raw_scores.size,
        "raw_scores",
    )

    null_mean = np.asarray(
        baseline["null_mean"],
        dtype=float,
    )

    null_sd = np.asarray(
        baseline["null_sd"],
        dtype=float,
    )

    if (
        null_mean.shape != raw_scores.shape
        or null_sd.shape != raw_scores.shape
    ):
        raise ValueError(
            "Raw scores and reference arrays must have matching shapes."
        )

    if minimum_null_sd is None:
        minimum_null_sd = baseline.get(
            "minimum_null_sd",
            0.0,
        )

    minimum_null_sd = _validate_minimum_null_sd(
        minimum_null_sd
    )

    valid = (
        np.isfinite(null_mean)
        & np.isfinite(null_sd)
        & (null_sd > minimum_null_sd)
    )

    z_scores = np.full(
        raw_scores.shape,
        np.nan,
        dtype=float,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        np.divide(
            raw_scores - null_mean,
            null_sd,
            out=z_scores,
            where=valid,
        )

    valid_z = (
        valid
        & np.isfinite(z_scores)
    )

    z_scores[~valid_z] = np.nan

    if strict and not np.all(valid_z):
        invalid_indices = np.flatnonzero(
            ~valid_z
        ).tolist()

        raise ValueError(
            "Permutation-Z is undefined for feature indices "
            f"{invalid_indices}. Inspect the reference diagnostics. "
            "SDs are not automatically replaced or regularized."
        )

    return z_scores


# ============================================================
# Unified Hybrid API
# ============================================================

def compute_hybrid_importance(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    feature_names: Optional[Sequence[str]] = None,
    n_permutations: int = 200,
    exact_limit: int = 10000,
    random_state: int = 42,
    minimum_null_sd: float = 0.0,
    strict: bool = True,
) -> Tuple[
    np.ndarray,
    pd.DataFrame,
    PermutationBaseline,
]:
    """Compute raw scores and Hybrid-standardized AD-DIFFI scores.

    Returns
    -------
    z_scores
        Feature-specific standardized scores.
    diagnostics
        Feature-level raw scores, reference summaries, and validity
        diagnostics.
    baseline
        Complete reference arrays and configuration metadata.

    Notes
    -----
    This function does not require both model-defined groups in the
    observed data. If that is a study requirement, check it before
    calling this function.

    strict=False is recommended when diagnostics must be saved before
    deciding how to handle undefined scores. Undefined scores must not
    be silently used for feature ranking.
    """
    X_data, _ = _validate_inputs(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
    )

    n_samples, n_features = X_data.shape

    names = _validate_feature_names(
        feature_names,
        n_features,
    )

    observed_mask = get_model_anomaly_mask(
        iforest,
        X_data,
    )

    raw_scores = compute_raw_ad_diffi(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=observed_mask,
    )

    raw_scores = _validate_score_vector(
        raw_scores,
        n_features,
        "observed raw scores",
    )

    baseline = estimate_permutation_baseline(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        n_permutations=n_permutations,
        exact_limit=exact_limit,
        random_state=random_state,
        mode="hybrid",
        minimum_null_sd=minimum_null_sd,
    )

    z_scores = apply_permutation_z(
        raw_scores=raw_scores,
        baseline=baseline,
        strict=False,
    )

    root_features = np.asarray(
        [
            estimator.tree_.feature[0]
            for estimator in iforest.estimators_
        ],
        dtype=int,
    )

    records: List[Dict[str, Any]] = []

    for feature_index in range(n_features):
        column = X_data[:, feature_index]
        feature_type = feature_types[feature_index]

        summary = baseline["summaries"][
            feature_index
        ]

        reference = baseline["references"][
            feature_index
        ]

        valid_z = bool(
            np.isfinite(
                z_scores[feature_index]
            )
        )

        n_root_trees = int(
            np.sum(
                root_features == feature_index
            )
        )

        n_unique = int(
            np.unique(column).size
        )

        if feature_type == "bin":
            n_ones = int(
                np.count_nonzero(column == 1)
            )
            n_zeros = n_samples - n_ones
            minority_count: Optional[int] = min(
                n_ones,
                n_zeros,
            )
        else:
            minority_count = None

        if valid_z:
            status = "ok"
        elif n_unique == 1:
            status = "constant_feature"
        elif (
            feature_type == "bin"
            and n_root_trees == 0
        ):
            status = "no_eligible_root_tree"
        elif not summary["valid_null_sd"]:
            status = "undefined_null_sd"
        else:
            status = "nonfinite_z"

        records.append({
            "feature_index": feature_index,
            "feature": names[feature_index],
            "feature_type": feature_type,
            "raw_score": float(
                raw_scores[feature_index]
            ),
            **summary,
            "z_score": float(
                z_scores[feature_index]
            ),
            "valid_z": valid_z,
            "status": status,
            "n_distinct_assignments": reference[
                "n_distinct_assignments"
            ],
            "requested_mc_permutations": (
                baseline["n_permutations"]
            ),
            "exact_limit": baseline["exact_limit"],
            "n_root_trees": n_root_trees,
            "n_unique": n_unique,
            "minority_count": minority_count,
            "n_observed_outliers": int(
                observed_mask.sum()
            ),
            "observed_mask_homogeneous": bool(
                observed_mask.all()
                or not observed_mask.any()
            ),
        })

    diagnostics = pd.DataFrame(
        records
    )

    if strict and not diagnostics["valid_z"].all():
        invalid_features = diagnostics.loc[
            ~diagnostics["valid_z"],
            "feature",
        ].tolist()

        raise ValueError(
            "Hybrid Permutation-Z is undefined for "
            f"{invalid_features}. "
            "Call with strict=False to obtain and save diagnostics. "
            "Do not rank features using undefined Z-scores."
        )

    return (
        z_scores,
        diagnostics,
        baseline,
    )


__all__ = [
    "get_model_anomaly_mask",
    "summarize_permutation_reference",
    "estimate_permutation_baseline",
    "apply_permutation_z",
    "compute_hybrid_importance",
]
