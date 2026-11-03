#!/usr/bin/env python3
"""Validate the Monte Carlo budget used by Hybrid AD-DIFFI.

For each simulated dataset and fitted Isolation Forest:

1. Keep the forest fixed.
2. Construct an independent, larger reference distribution.
3. Generate several independent candidate Monte Carlo streams.
4. Evaluate prefixes of each candidate stream.
5. Compare reference SDs, Z-scores, rankings, and top-k selections.

Exact-enumeration features are evaluated once and reused across
candidate budgets and streams. Their results are excluded from
MC-specific numerical-error summaries.

The larger Monte Carlo reference is an approximation, not ground truth.

Run from the repository root, for example:

    python scripts/validate_permutation_budget.py \
        --scenarios S1 S3 S8 \
        --fits 2 \
        --streams 3 \
        --budgets 10 25 50 100 200 \
        --reference-permutations 1000 \
        --output results/permutation_budget_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest


# ============================================================
# Repository imports
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ad_diffi.core import compute_raw_ad_diffi

from ad_diffi.permutation import (
    _collect_feature_reference,
    apply_permutation_z,
    estimate_permutation_baseline,
    get_model_anomaly_mask,
    summarize_permutation_reference,
)


# ============================================================
# Simulation definitions
# ============================================================

FEATURE_NAMES = [
    "X1_Strong_Cont",
    "X2_Weak_Cont",
    "X3_Moderate_Cont",
    "X4_Strong_Bin",
    "X5_Weak_Bin",
    "X6_Cont_Noise",
    "X7_Bin_Noise_A",
    "X8_Bin_Noise_B",
]

FEATURE_TYPES = {
    0: "cont",
    1: "cont",
    2: "cont",
    3: "bin",
    4: "bin",
    5: "cont",
    6: "bin",
    7: "bin",
}

BASE_SCENARIO = {
    "contamination": 0.05,
    "subgroup_fraction": 0.5,
    "signal_scale": 1.0,
    "binary_noise_p": 0.05,
    "normal_binary_p": 0.05,
    "correlation": 0.0,
    "n_samples": 1000,
}

SCENARIOS = {
    "S1": {
        **BASE_SCENARIO,
    },
    "S2": {
        **BASE_SCENARIO,
        "signal_scale": 0.5,
    },
    "S3": {
        **BASE_SCENARIO,
        "binary_noise_p": 0.01,
    },
    "S4": {
        **BASE_SCENARIO,
        "binary_noise_p": 0.50,
    },
    "S5": {
        **BASE_SCENARIO,
        "contamination": 0.10,
    },
    "S6": {
        **BASE_SCENARIO,
        "contamination": 0.02,
    },
    "S7": {
        **BASE_SCENARIO,
        "correlation": 0.5,
    },
    "S8": {
        **BASE_SCENARIO,
        "n_samples": 300,
    },
    "S9": {
        **BASE_SCENARIO,
        "subgroup_fraction": 0.8,
    },
}

SCENARIO_NAMES = {
    "S1": "S1_calibrated_subgroups",
    "S2": "S2_weak_signals",
    "S3": "S3_rare_binary_noise",
    "S4": "S4_common_binary_noise",
    "S5": "S5_higher_contamination",
    "S6": "S6_lower_contamination",
    "S7": "S7_correlated_noise",
    "S8": "S8_small_sample",
    "S9": "S9_unequal_subgroups",
}


def generate_dataset(
    seed: int,
    scenario: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the revised Chapter 5 simulation data.

    In S7, X6 shares residual variation with X1 but has no
    anomaly-specific marginal mean shift.
    """
    rng = np.random.default_rng(seed)

    n = int(scenario["n_samples"])
    contamination = float(scenario["contamination"])
    scale = float(scenario["signal_scale"])
    rho = float(scenario["correlation"])

    if not -1.0 < rho < 1.0:
        raise ValueError(
            "correlation must lie strictly between -1 and 1."
        )

    n_anomaly = int(round(n * contamination))
    n_normal = n - n_anomaly

    if n_anomaly == 0 or n_normal == 0:
        raise ValueError(
            "Both normal and anomalous observations are required."
        )

    n_a = int(
        round(
            n_anomaly
            * scenario["subgroup_fraction"]
        )
    )

    anomaly_indices = np.arange(n_normal, n)
    group_a = anomaly_indices[:n_a]
    group_b = anomaly_indices[n_a:]

    y = np.zeros(n, dtype=bool)
    y[n_normal:] = True

    X = np.empty(
        (n, len(FEATURE_NAMES)),
        dtype=float,
    )

    x1_residual = rng.normal(0.0, 1.0, n)

    X[:, 0] = 10.0 + x1_residual
    X[group_a, 0] += 10.0 * scale

    X[:, 1] = rng.normal(10.0, 1.0, n)
    X[y, 1] = rng.normal(
        10.0 + 3.0 * scale,
        1.0,
        n_anomaly,
    )

    X[:, 2] = rng.normal(10.0, 1.0, n)
    X[group_b, 2] = rng.normal(
        10.0 + 6.0 * scale,
        1.0,
        len(group_b),
    )

    p_normal = float(scenario["normal_binary_p"])

    X[:, 3] = rng.binomial(1, p_normal, n)
    X[group_a, 3] = rng.binomial(
        1,
        min(p_normal + 0.75 * scale, 0.99),
        len(group_a),
    )

    X[:, 4] = rng.binomial(1, p_normal, n)
    X[group_b, 4] = rng.binomial(
        1,
        min(p_normal + 0.30 * scale, 0.99),
        len(group_b),
    )

    X[:, 5] = (
        10.0
        + rho * x1_residual
        + np.sqrt(1.0 - rho**2)
        * rng.normal(0.0, 1.0, n)
    )

    p_noise = float(scenario["binary_noise_p"])

    X[:, 6] = rng.binomial(1, p_noise, n)
    X[:, 7] = rng.binomial(1, p_noise, n)

    order = rng.permutation(n)

    return X[order], y[order]


# ============================================================
# Reproducibility helpers
# ============================================================

def make_seed(
    base_seed: int,
    *coordinates: int,
) -> int:
    """Create a reproducible integer seed for a named task."""
    sequence = np.random.SeedSequence(
        [base_seed, *coordinates]
    )

    return int(
        sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def repository_commit() -> str | None:
    """Return the repository commit when Git metadata is available."""
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None


def function_hash(function) -> str:
    """Return a hash of the function source."""
    source = inspect.getsource(function)

    return hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()


def save_reference(
    path: Path,
    reference: Dict[str, Any],
) -> None:
    """Save feature-reference arrays without pickle."""
    np.savez_compressed(
        path,
        scores=reference["scores"],
        outlier_counts=reference["outlier_counts"],
        elapsed_seconds=reference["elapsed_seconds"],
        mode=np.asarray(reference["mode"]),
        feature_index=reference["feature_index"],
    )


# ============================================================
# Score and ranking diagnostics
# ============================================================

def reference_z(
    raw_score: float,
    summary: Dict[str, Any],
) -> float:
    """Return an unregularized Z-score or NaN."""
    if not summary["valid_null_sd"]:
        return np.nan

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        z = (
            raw_score - summary["null_mean"]
        ) / summary["null_sd"]

    return float(z) if np.isfinite(z) else np.nan


def compare_rankings(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    """Compare complete rankings using signed Z-scores.

    If any score is undefined, the full ranking comparison is
    undefined. Features are not silently dropped.

    Ties at the top-k boundary are broken by stable column order.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            "Ranking vectors must have matching shapes."
        )

    all_valid = bool(
        np.isfinite(scores_a).all()
        and np.isfinite(scores_b).all()
    )

    result = {
        "all_features_valid": all_valid,
        "spearman": np.nan,
        "topk_overlap": np.nan,
        "topk_jaccard": np.nan,
    }

    if not all_valid:
        return result

    if (
        np.unique(scores_a).size > 1
        and np.unique(scores_b).size > 1
    ):
        result["spearman"] = float(
            spearmanr(
                scores_a,
                scores_b,
            ).statistic
        )

    top_a = set(
        np.argsort(
            -scores_a,
            kind="stable",
        )[:top_k].tolist()
    )

    top_b = set(
        np.argsort(
            -scores_b,
            kind="stable",
        )[:top_k].tolist()
    )

    intersection = len(top_a & top_b)
    union = len(top_a | top_b)

    result["topk_overlap"] = intersection / top_k
    result["topk_jaccard"] = intersection / union

    return result


# ============================================================
# One dataset / fitted forest
# ============================================================

def run_one_fit(
    scenario_id: str,
    fit_id: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Evaluate candidate budgets conditional on one fitted forest."""
    scenario = SCENARIOS[scenario_id]
    scenario_number = int(scenario_id[1:])

    fit_dir = (
        output_dir
        / f"{scenario_id}_fit_{fit_id:03d}"
    )
    fit_dir.mkdir(parents=True, exist_ok=False)

    data_seed = make_seed(
        args.seed,
        scenario_number,
        fit_id,
        0,
    )

    forest_seed = make_seed(
        args.seed,
        scenario_number,
        fit_id,
        1,
    )

    reference_seed = make_seed(
        args.seed,
        scenario_number,
        fit_id,
        2,
    )

    X, y = generate_dataset(
        data_seed,
        scenario,
    )

    model = IsolationForest(
        n_estimators=args.trees,
        max_samples=min(args.max_samples, len(X)),
        contamination=scenario["contamination"],
        max_features=1.0,
        bootstrap=False,
        n_jobs=args.jobs,
        random_state=forest_seed,
    ).fit(X)

    observed_mask = get_model_anomaly_mask(
        model,
        X,
    )

    if observed_mask.all() or not observed_mask.any():
        raise ValueError(
            f"{scenario_id}, fit {fit_id}: observed data lack "
            "both model-defined outliers and inliers."
        )

    raw_scores = compute_raw_ad_diffi(
        iforest=model,
        X_data=X,
        feature_types=FEATURE_TYPES,
        anomaly_mask=observed_mask,
    )

    np.savez_compressed(
        fit_dir / "observed_data.npz",
        X=X,
        y=y,
        observed_mask=observed_mask,
        raw_scores=raw_scores,
        data_seed=data_seed,
        forest_seed=forest_seed,
    )

    print(
        f"[{scenario_id}, fit {fit_id + 1}] "
        "building independent reference",
        flush=True,
    )

    independent_baseline = estimate_permutation_baseline(
        iforest=model,
        X_data=X,
        feature_types=FEATURE_TYPES,
        n_permutations=args.reference_permutations,
        exact_limit=args.exact_limit,
        random_state=reference_seed,
        mode="hybrid",
        minimum_null_sd=args.minimum_null_sd,
    )

    independent_z = apply_permutation_z(
        raw_scores=raw_scores,
        baseline=independent_baseline,
        strict=False,
    )

    root_features = np.asarray(
        [
            estimator.tree_.feature[0]
            for estimator in model.estimators_
        ],
        dtype=int,
    )

    records: List[Dict[str, Any]] = []

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):
        reference = independent_baseline[
            "references"
        ][feature_index]

        reference_summary = independent_baseline[
            "summaries"
        ][feature_index]

        save_reference(
            fit_dir
            / f"reference_feature_{feature_index:02d}.npz",
            reference,
        )

        feature_type = FEATURE_TYPES[feature_index]
        column = X[:, feature_index]

        n_unique = int(np.unique(column).size)
        n_root_trees = int(
            np.sum(root_features == feature_index)
        )

        if feature_type == "bin":
            n_ones = int(np.count_nonzero(column == 1))
            minority_count = min(
                n_ones,
                len(X) - n_ones,
            )
        else:
            minority_count = None

        reference_valid = bool(
            np.isfinite(independent_z[feature_index])
        )

        for stream_id in range(args.streams):
            if reference["mode"] == "exact":
                # Exact work is shared, not repeated per stream.
                candidate = reference
                candidate_seed = None

            else:
                candidate_seed = make_seed(
                    args.seed,
                    scenario_number,
                    fit_id,
                    3,
                    feature_index,
                    stream_id,
                )

                # Reuse the same engine as permutation.py.
                # No raw-score or reference logic is duplicated here.
                candidate = _collect_feature_reference(
                    iforest=model,
                    X_data=X,
                    feature_types=FEATURE_TYPES,
                    feature_index=feature_index,
                    n_permutations=max(args.budgets),
                    seed_sequence=np.random.SeedSequence(
                        candidate_seed
                    ),
                    reference_mode="monte_carlo",
                    n_distinct_assignments=reference[
                        "n_distinct_assignments"
                    ],
                )

                save_reference(
                    fit_dir
                    / (
                        f"candidate_feature_{feature_index:02d}"
                        f"_stream_{stream_id:02d}.npz"
                    ),
                    candidate,
                )

            for budget in args.budgets:
                summary = summarize_permutation_reference(
                    reference=candidate,
                    n_permutations=budget,
                    minimum_null_sd=args.minimum_null_sd,
                )

                z = reference_z(
                    raw_scores[feature_index],
                    summary,
                )

                reference_sd = reference_summary["null_sd"]

                if reference_summary["valid_null_sd"]:
                    sd_relative_error = (
                        abs(
                            summary["null_sd"]
                            - reference_sd
                        )
                        / reference_sd
                    )

                    mean_error_in_reference_sd = (
                        abs(
                            summary["null_mean"]
                            - reference_summary["null_mean"]
                        )
                        / reference_sd
                    )
                else:
                    sd_relative_error = np.nan
                    mean_error_in_reference_sd = np.nan

                if (
                    np.isfinite(z)
                    and reference_valid
                ):
                    z_abs_error = abs(
                        z - independent_z[feature_index]
                    )
                else:
                    z_abs_error = np.nan

                records.append({
                    "scenario": scenario_id,
                    "scenario_name": SCENARIO_NAMES[
                        scenario_id
                    ],
                    "fit_id": fit_id,
                    "stream_id": stream_id,
                    "B": budget,
                    "feature_index": feature_index,
                    "feature": feature_name,
                    "feature_type": feature_type,
                    "raw_score": raw_scores[feature_index],
                    **summary,
                    "z_score": z,
                    "valid_z": bool(np.isfinite(z)),
                    "reference_mean": reference_summary[
                        "null_mean"
                    ],
                    "reference_sd": reference_sd,
                    "reference_z": independent_z[
                        feature_index
                    ],
                    "reference_valid_z": reference_valid,
                    "reference_n_evaluations": (
                        reference_summary["n_evaluations"]
                    ),
                    "sd_relative_error": sd_relative_error,
                    "mean_error_in_reference_sd": (
                        mean_error_in_reference_sd
                    ),
                    "z_abs_error": z_abs_error,
                    "n_root_trees": n_root_trees,
                    "n_unique": n_unique,
                    "minority_count": minority_count,
                    "data_seed": data_seed,
                    "forest_seed": forest_seed,
                    "reference_seed": reference_seed,
                    "candidate_seed": candidate_seed,
                })

        pd.DataFrame(records).to_csv(
            fit_dir / "estimates_partial.csv",
            index=False,
        )

        print(
            f"[{scenario_id}, fit {fit_id + 1}] "
            f"feature {feature_index + 1}/{len(FEATURE_NAMES)} "
            f"completed ({reference['mode']})",
            flush=True,
        )

    estimates = pd.DataFrame(records)

    estimates.to_csv(
        fit_dir / "estimates.csv",
        index=False,
    )

    reference_rank_records = []
    stream_rank_records = []

    for budget in args.budgets:
        table = (
            estimates.loc[
                estimates["B"] == budget
            ]
            .pivot(
                index="feature_index",
                columns="stream_id",
                values="z_score",
            )
            .reindex(range(len(FEATURE_NAMES)))
        )

        for stream_id in range(args.streams):
            reference_rank_records.append({
                "scenario": scenario_id,
                "fit_id": fit_id,
                "B": budget,
                "stream_id": stream_id,
                **compare_rankings(
                    table[stream_id].to_numpy(),
                    independent_z,
                    args.top_k,
                ),
            })

        for stream_a, stream_b in combinations(
            range(args.streams),
            2,
        ):
            stream_rank_records.append({
                "scenario": scenario_id,
                "fit_id": fit_id,
                "B": budget,
                "stream_a": stream_a,
                "stream_b": stream_b,
                **compare_rankings(
                    table[stream_a].to_numpy(),
                    table[stream_b].to_numpy(),
                    args.top_k,
                ),
            })

    pd.DataFrame(reference_rank_records).to_csv(
        fit_dir / "rank_vs_reference.csv",
        index=False,
    )

    pd.DataFrame(stream_rank_records).to_csv(
        fit_dir / "rank_between_streams.csv",
        index=False,
    )

    # Hypothetical cost of one Hybrid run at each candidate B.
    # Excludes fitting, observed raw scoring, and file I/O.
    candidate_costs = (
        estimates
        .groupby(
            ["B", "stream_id"],
            as_index=False,
        )
        .agg(
            n_evaluations=("n_evaluations", "sum"),
            reference_seconds=("elapsed_seconds", "sum"),
        )
    )

    candidate_costs.to_csv(
        fit_dir / "candidate_costs.csv",
        index=False,
    )

    (fit_dir / "estimates_partial.csv").unlink(
        missing_ok=True
    )

    # Only folders with this marker enter the final summaries.
    (fit_dir / "COMPLETE").write_text(
        "completed\n",
        encoding="utf-8",
    )


# ============================================================
# Aggregate summaries
# ============================================================

def concatenate_completed(
    completed_dirs: Sequence[Path],
    filename: str,
) -> pd.DataFrame:
    """Load one output table from every completed fit."""
    return pd.concat(
        [
            pd.read_csv(directory / filename)
            for directory in completed_dirs
        ],
        ignore_index=True,
    )


def summarize_rank_table(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize rank comparisons while retaining validity counts."""
    return (
        table
        .groupby(
            ["scenario", "B"],
            as_index=False,
        )
        .agg(
            n_comparisons=("all_features_valid", "size"),
            n_valid_comparisons=("all_features_valid", "sum"),
            valid_comparison_fraction=(
                "all_features_valid",
                "mean",
            ),
            spearman_median=("spearman", "median"),
            spearman_min=("spearman", "min"),
            topk_overlap_median=("topk_overlap", "median"),
            topk_overlap_min=("topk_overlap", "min"),
        )
    )


def summarize_outputs(
    output_dir: Path,
) -> None:
    """Summarize only fully completed fits."""
    completed_dirs = sorted(
        marker.parent
        for marker in output_dir.glob(
            "S*_fit_*/COMPLETE"
        )
    )

    if not completed_dirs:
        raise ValueError(
            "No completed fits are available."
        )

    estimates = concatenate_completed(
        completed_dirs,
        "estimates.csv",
    )

    feature_summary = (
        estimates
        .groupby(
            [
                "scenario",
                "fit_id",
                "B",
                "feature_index",
                "feature",
                "feature_type",
                "mode",
            ],
            as_index=False,
        )
        .agg(
            n_streams=("stream_id", "size"),
            valid_z_fraction=("valid_z", "mean"),
            reference_valid_z=("reference_valid_z", "first"),
            sd_median=("null_sd", "median"),
            sd_min=("null_sd", "min"),
            sd_max=("null_sd", "max"),
            sd_between_streams=("null_sd", "std"),
            z_min=("z_score", "min"),
            z_max=("z_score", "max"),
            z_between_streams=("z_score", "std"),
            sd_relative_error_median=(
                "sd_relative_error",
                "median",
            ),
            sd_relative_error_max=(
                "sd_relative_error",
                "max",
            ),
            z_abs_error_median=("z_abs_error", "median"),
            z_abs_error_max=("z_abs_error", "max"),
            n_root_trees=("n_root_trees", "first"),
            minority_count=("minority_count", "first"),
        )
    )

    feature_summary["sd_relative_spread"] = (
        feature_summary["sd_between_streams"]
        / feature_summary["sd_median"].replace(
            0,
            np.nan,
        )
    )

    # Exact rows are deliberately excluded from MC budget summaries.
    mc_summary = feature_summary.loc[
        feature_summary["mode"] == "monte_carlo"
    ]

    mc_budget_summary = (
        mc_summary
        .groupby(
            ["scenario", "B"],
            as_index=False,
        )
        .agg(
            n_feature_fit_cells=("feature", "size"),
            valid_z_fraction_min=(
                "valid_z_fraction",
                "min",
            ),
            reference_valid_fraction=(
                "reference_valid_z",
                "mean",
            ),
            sd_relative_spread_max=(
                "sd_relative_spread",
                "max",
            ),
            sd_relative_error_max=(
                "sd_relative_error_max",
                "max",
            ),
            z_abs_error_max=(
                "z_abs_error_max",
                "max",
            ),
        )
    )

    validity_counts = (
        estimates
        .groupby(
            ["scenario", "B", "mode"],
            as_index=False,
        )
        .agg(
            n_rows=("feature", "size"),
            n_valid_z=("valid_z", "sum"),
            n_valid_reference_z=(
                "reference_valid_z",
                "sum",
            ),
        )
    )

    rank_vs_reference = concatenate_completed(
        completed_dirs,
        "rank_vs_reference.csv",
    )

    rank_between_streams = concatenate_completed(
        completed_dirs,
        "rank_between_streams.csv",
    )

    costs = concatenate_completed(
        completed_dirs,
        "candidate_costs.csv",
    )

    outputs = {
        "all_estimates.csv": estimates,
        "feature_summary.csv": feature_summary,
        "mc_budget_summary.csv": mc_budget_summary,
        "validity_counts.csv": validity_counts,
        "rank_vs_reference.csv": rank_vs_reference,
        "rank_vs_reference_summary.csv": (
            summarize_rank_table(rank_vs_reference)
        ),
        "rank_between_streams.csv": rank_between_streams,
        "rank_between_streams_summary.csv": (
            summarize_rank_table(rank_between_streams)
        ),
        "candidate_costs.csv": costs,
    }

    for filename, table in outputs.items():
        table.to_csv(
            output_dir / filename,
            index=False,
        )


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=["S1", "S3", "S8"],
    )

    parser.add_argument(
        "--fits",
        type=int,
        default=2,
        help="Number of dataset/forest replications per scenario.",
    )

    parser.add_argument(
        "--streams",
        type=int,
        default=3,
        help="Independent candidate MC streams per fixed forest.",
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 200],
    )

    parser.add_argument(
        "--reference-permutations",
        type=int,
        default=1000,
        help="Independent, larger MC reference budget.",
    )

    parser.add_argument(
        "--exact-limit",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--trees",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--minimum-null-sd",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261103,
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/permutation_budget_pilot"
        ),
    )

    args = parser.parse_args()

    args.budgets = sorted(set(args.budgets))
    args.scenarios = list(dict.fromkeys(args.scenarios))

    if args.fits < 1:
        parser.error("--fits must be at least 1.")

    if args.streams < 2:
        parser.error("--streams must be at least 2.")

    if min(args.budgets) < 2:
        parser.error("Every candidate budget must be at least 2.")

    if args.reference_permutations <= max(args.budgets):
        parser.error(
            "--reference-permutations must exceed "
            "the largest candidate budget."
        )

    if args.exact_limit < 0:
        parser.error("--exact-limit must be nonnegative.")

    if args.trees < 1:
        parser.error("--trees must be at least 1.")

    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2.")

    if not 1 <= args.top_k <= len(FEATURE_NAMES):
        parser.error(
            "--top-k must be between 1 and the feature count."
        )

    if (
        not np.isfinite(args.minimum_null_sd)
        or args.minimum_null_sd < 0
    ):
        parser.error(
            "--minimum-null-sd must be finite and nonnegative."
        )

    if args.seed < 0:
        parser.error("--seed must be nonnegative.")

    if args.jobs == 0:
        parser.error("--jobs must not be zero.")

    return args


def main() -> None:
    args = parse_arguments()
    output_dir = args.output.resolve()

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory to avoid mixing runs."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        **vars(args),
        "output": str(output_dir),
        "repository_commit": repository_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "raw_function_sha256": function_hash(
            compute_raw_ad_diffi
        ),
        "reference_function_sha256": function_hash(
            estimate_permutation_baseline
        ),
        "feature_names": FEATURE_NAMES,
        "feature_types": FEATURE_TYPES,
        "scenario_definitions": {
            name: SCENARIOS[name]
            for name in args.scenarios
        },
        "reference_rule": (
            "fixed forest; target column rearranged; "
            "model-defined outlier mask recomputed"
        ),
        "exact_sd_ddof": 0,
        "monte_carlo_sd_ddof": 1,
        "undefined_sd_policy": (
            "NaN Z-score; no replacement or regularization"
        ),
        "candidate_design": (
            "nested prefixes within stream; "
            "independent larger reference"
        ),
    }

    (output_dir / "config.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for scenario_id in args.scenarios:
        for fit_id in range(args.fits):
            run_one_fit(
                scenario_id=scenario_id,
                fit_id=fit_id,
                args=args,
                output_dir=output_dir,
            )

            # Refresh summaries after every completed fit.
            summarize_outputs(output_dir)

    print(
        f"Completed. Outputs: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
