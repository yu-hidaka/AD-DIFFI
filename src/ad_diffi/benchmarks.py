"""
Paired real-world benchmark evaluation for Original DIFFI and AD-DIFFI.

Design
------
1. Stratified train/test split within every repetition.
2. Training-only preprocessing.
3. One shared Isolation Forest per repetition.
4. Feature importance calculated on the training data only.
5. Top-k features selected separately for each attribution method.
6. Downstream models evaluated on held-out test data.
7. Paired replicate-level differences and MCSE reported.

This module does not download data and does not generate synthetic fallback data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================


@dataclass
class BenchmarkConfig:
    """Configuration for one real-world benchmark analysis."""

    n_repeats: int = 20
    test_size: float = 0.30

    n_estimators: int = 100
    max_samples: int = 256
    contamination: float = 0.05
    max_features: float = 1.0
    bootstrap: bool = False

    top_k: int = 6

    n_noise: int = 100
    n_noise_samples: int = 2000

    base_seed: int = 42


# ============================================================
# Isolation Forest helpers
# ============================================================


def make_if_params(
    config: BenchmarkConfig,
    n_train: int,
) -> Dict[str, Any]:
    """
    Create Isolation Forest parameters for a training dataset.
    """
    return {
        "n_estimators": config.n_estimators,
        "max_samples": min(
            config.max_samples,
            n_train,
        ),
        "contamination": config.contamination,
        "max_features": config.max_features,
        "bootstrap": config.bootstrap,
        "n_jobs": -1,
    }


def fit_isolation_forest(
    X_train: np.ndarray,
    if_params: Dict[str, Any],
    random_state: int,
) -> IsolationForest:
    """
    Fit one Isolation Forest.
    """
    params = if_params.copy()
    params.pop("random_state", None)

    model = IsolationForest(
        **params,
        random_state=random_state,
    )

    model.fit(X_train)

    return model


def split_outlier_inlier(
    model: IsolationForest,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split observations according to the fitted forest decision function.

    Negative decision_function values are treated as predicted outliers and
    positive values as predicted inliers.
    """
    scores = model.decision_function(X)

    outlier_mask = scores < 0
    inlier_mask = scores > 0

    if outlier_mask.sum() == 0:
        raise RuntimeError(
            "The fitted Isolation Forest produced no predicted outliers."
        )

    if inlier_mask.sum() == 0:
        raise RuntimeError(
            "The fitted Isolation Forest produced no predicted inliers."
        )

    return outlier_mask, inlier_mask


# ============================================================
# Preprocessing
# ============================================================


def fit_train_scaler(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only and transform both partitions.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, scaler


def make_stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    random_state: int,
):
    """
    Create a stratified train/test split using row indices.
    """
    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return train_indices, test_indices


# ============================================================
# Feature-score and Z-score helpers
# ============================================================


def make_mixed_noise(
    n_samples: int,
    feature_types: Dict[int, str],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate mixed continuous/binary noise data.

    Continuous features follow N(0,1).
    Binary features follow Bernoulli(0.5).
    """
    n_features = len(feature_types)

    X_noise = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    for feature_index, feature_type in feature_types.items():
        if feature_type == "cont":
            X_noise[:, feature_index] = rng.normal(
                loc=0.0,
                scale=1.0,
                size=n_samples,
            )
        elif feature_type == "bin":
            X_noise[:, feature_index] = rng.binomial(
                n=1,
                p=0.5,
                size=n_samples,
            )
        else:
            raise ValueError(
                f"Unknown feature type: {feature_type}"
            )

    return X_noise


def estimate_noise_baseline(
    feature_types: Dict[int, str],
    if_params: Dict[str, Any],
    diffi_func_ad: Callable,
    n_iter: int,
    n_samples: int,
    random_seed: int,
) -> Dict[str, float]:
    """
    Estimate continuous and binary AD-DIFFI noise baselines.

    The baseline is estimated independently for each benchmark repetition.
    """
    score_matrix = []

    for iteration in range(n_iter):
        seed = random_seed + iteration
        rng = np.random.default_rng(seed)

        X_noise = make_mixed_noise(
            n_samples=n_samples,
            feature_types=feature_types,
            rng=rng,
        )

        model = fit_isolation_forest(
            X_train=X_noise,
            if_params=if_params,
            random_state=seed,
        )

        raw_scores = diffi_func_ad(
            model,
            X_noise,
            feature_types,
        )

        score_matrix.append(
            np.asarray(
                raw_scores,
                dtype=float,
            )
        )

    matrix = np.vstack(score_matrix)

    continuous_indices = [
        index
        for index, feature_type in feature_types.items()
        if feature_type == "cont"
    ]

    binary_indices = [
        index
        for index, feature_type in feature_types.items()
        if feature_type == "bin"
    ]

    if not continuous_indices:
        continuous_mean = 0.0
        continuous_sd = 1.0
    else:
        continuous_values = matrix[
            :,
            continuous_indices,
        ]

        continuous_mean = float(
            continuous_values.mean()
        )

        continuous_sd = float(
            continuous_values.std(
                ddof=1
            )
        )

    if not binary_indices:
        binary_mean = 0.0
        binary_sd = 1.0
    else:
        binary_values = matrix[
            :,
            binary_indices,
        ]

        binary_mean = float(
            binary_values.mean()
        )

        binary_sd = float(
            binary_values.std(
                ddof=1
            )
        )

    if continuous_sd <= 0:
        continuous_sd = 1.0

    if binary_sd <= 0:
        binary_sd = 1.0

    return {
        "continuous_mean": continuous_mean,
        "continuous_sd": continuous_sd,
        "binary_mean": binary_mean,
        "binary_sd": binary_sd,
    }


def z_standardize_scores(
    raw_scores: np.ndarray,
    feature_types: Dict[int, str],
    baseline: Dict[str, float],
) -> np.ndarray:
    """
    Apply type-specific empirical Z-score normalization.
    """
    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    z_scores = np.zeros_like(
        raw_scores,
        dtype=float,
    )

    for feature_index, feature_type in feature_types.items():
        if feature_type == "cont":
            mean = baseline["continuous_mean"]
            sd = baseline["continuous_sd"]
        elif feature_type == "bin":
            mean = baseline["binary_mean"]
            sd = baseline["binary_sd"]
        else:
            raise ValueError(
                f"Unknown feature type: {feature_type}"
            )

        z_scores[feature_index] = (
            raw_scores[feature_index] - mean
        ) / sd

    return z_scores


# ============================================================
# Ranking helpers
# ============================================================


def rank_scores(
    scores: np.ndarray,
    feature_names: List[str],
) -> pd.Series:
    """
    Rank features from 1 (highest score) to p (lowest score).

    Average ranks are used for tied scores.
    """
    score_series = pd.Series(
        np.asarray(scores, dtype=float),
        index=feature_names,
    )

    return score_series.rank(
        ascending=False,
        method="average",
    )


def make_rank_table(
    feature_names: List[str],
    feature_types: Dict[int, str],
    original_scores: np.ndarray,
    ad_scores: np.ndarray,
    repeat: int,
) -> pd.DataFrame:
    """
    Create one repetition-level feature-rank table.
    """
    original_scores = np.asarray(
        original_scores,
        dtype=float,
    )

    ad_scores = np.asarray(
        ad_scores,
        dtype=float,
    )

    original_ranks = rank_scores(
        original_scores,
        feature_names,
    )

    ad_ranks = rank_scores(
        ad_scores,
        feature_names,
    )

    feature_type_names = [
        feature_types[index]
        for index in range(len(feature_names))
    ]

    table = pd.DataFrame(
        {
            "repeat": repeat,
            "Feature": feature_names,
            "Type": feature_type_names,
            "Original_DIFFI": original_scores,
            "AD_DIFFI_RSO_Z": ad_scores,
            "Rank_Original": original_ranks.values,
            "Rank_AD_DIFFI": ad_ranks.values,
        }
    )

    table["Rank_Change"] = (
        table["Rank_AD_DIFFI"]
        - table["Rank_Original"]
    )

    return table


def select_top_k_features(
    scores: np.ndarray,
    feature_names: List[str],
    k: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Select top-k features using stable descending-score ordering.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    if k > len(feature_names):
        k = len(feature_names)

    order = np.argsort(
        -scores,
        kind="stable",
    )

    selected_indices = order[:k]

    selected_features = [
        feature_names[index]
        for index in selected_indices
    ]

    return selected_features, selected_indices


# ============================================================
# Feature-score calculation
# ============================================================


def calculate_feature_scores(
    model: IsolationForest,
    X_train: np.ndarray,
    feature_types: Dict[int, str],
    diffi_func_orig: Callable,
    diffi_func_ad: Callable,
):
    """
    Calculate Original DIFFI and raw AD-DIFFI scores on training data.
    """
    split_outlier_inlier(
        model,
        X_train,
    )

    original_result = diffi_func_orig(
        model,
        X_train,
    )

    if isinstance(
        original_result,
        (tuple, list),
    ):
        original_scores = np.asarray(
            original_result[0],
            dtype=float,
        )
    else:
        original_scores = np.asarray(
            original_result,
            dtype=float,
        )

    ad_scores = np.asarray(
        diffi_func_ad(
            model,
            X_train,
            feature_types,
        ),
        dtype=float,
    )

    return original_scores, ad_scores


# ============================================================
# Downstream evaluation
# ============================================================


def evaluate_if_selected_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_indices: np.ndarray,
    if_params: Dict[str, Any],
    random_state: int,
) -> float:
    """
    Fit an Isolation Forest using only selected training features and evaluate
    anomaly-score AUC on held-out test observations.
    """
    params = if_params.copy()
    params.pop("random_state", None)

    model = IsolationForest(
        **params,
        random_state=random_state,
    )

    model.fit(
        X_train[:, selected_indices]
    )

    anomaly_scores = -model.decision_function(
        X_test[:, selected_indices]
    )

    return float(
        roc_auc_score(
            y_test,
            anomaly_scores,
        )
    )


def evaluate_logistic_selected_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_indices: np.ndarray,
    random_state: int,
) -> float:
    """
    Fit logistic regression on selected training features and evaluate
    predicted-probability AUC on held-out test observations.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(
        X_train[:, selected_indices],
        y_train,
    )

    probabilities = model.predict_proba(
        X_test[:, selected_indices]
    )[:, 1]

    return float(
        roc_auc_score(
            y_test,
            probabilities,
        )
    )


# ============================================================
# One paired repetition
# ============================================================


def run_one_repeat(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: BenchmarkConfig,
    diffi_func_orig: Callable,
    diffi_func_ad: Callable,
    repeat: int,
):
    """
    Run one paired train/test benchmark repetition.
    """
    seed = config.base_seed + repeat

    train_indices, test_indices = train_test_split(
        np.arange(len(y)),
        test_size=config.test_size,
        random_state=seed,
        stratify=y,
    )

    X_train_raw = X.iloc[
        train_indices
    ].to_numpy(
        dtype=float
    )

    X_test_raw = X.iloc[
        test_indices
    ].to_numpy(
        dtype=float
    )

    y_train = y[train_indices]
    y_test = y[test_indices]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_raw
    )

    X_test = scaler.transform(
        X_test_raw
    )

    if_params = make_if_params(
        config=config,
        n_train=X_train.shape[0],
    )

    model = fit_isolation_forest(
        X_train=X_train,
        if_params=if_params,
        random_state=seed,
    )

    original_scores, ad_raw_scores = (
        calculate_feature_scores(
            model=model,
            X_train=X_train,
            feature_types=feature_types,
            diffi_func_orig=diffi_func_orig,
            diffi_func_ad=diffi_func_ad,
        )
    )

    baseline = estimate_noise_baseline(
        feature_types=feature_types,
        if_params=if_params,
        diffi_func_ad=diffi_func_ad,
        n_iter=config.n_noise,
        n_samples=config.n_noise_samples,
        random_seed=seed + 100000,
    )

    ad_z_scores = z_standardize_scores(
        raw_scores=ad_raw_scores,
        feature_types=feature_types,
        baseline=baseline,
    )

    rank_table = make_rank_table(
        feature_names=feature_names,
        feature_types=feature_types,
        original_scores=original_scores,
        ad_scores=ad_z_scores,
        repeat=repeat,
    )

    original_features, original_indices = (
        select_top_k_features(
            scores=original_scores,
            feature_names=feature_names,
            k=config.top_k,
        )
    )

    ad_features, ad_indices = (
        select_top_k_features(
            scores=ad_z_scores,
            feature_names=feature_names,
            k=config.top_k,
        )
    )

    if_auc_original = evaluate_if_selected_features(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        selected_indices=original_indices,
        if_params=if_params,
        random_state=seed,
    )

    if_auc_ad = evaluate_if_selected_features(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        selected_indices=ad_indices,
        if_params=if_params,
        random_state=seed,
    )

    lr_auc_original = evaluate_logistic_selected_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_indices=original_indices,
        random_state=seed,
    )

    lr_auc_ad = evaluate_logistic_selected_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selected_indices=ad_indices,
        random_state=seed,
    )

    repeat_result = {
        "repeat": repeat,
        "seed": seed,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "if_auc_original": if_auc_original,
        "if_auc_ad": if_auc_ad,
        "if_auc_difference": (
            if_auc_ad - if_auc_original
        ),
        "lr_auc_original": lr_auc_original,
        "lr_auc_ad": lr_auc_ad,
        "lr_auc_difference": (
            lr_auc_ad - lr_auc_original
        ),
        "original_top_features": (
            original_features
        ),
        "ad_top_features": ad_features,
        "continuous_baseline_mean": (
            baseline["continuous_mean"]
        ),
        "continuous_baseline_sd": (
            baseline["continuous_sd"]
        ),
        "binary_baseline_mean": (
            baseline["binary_mean"]
        ),
        "binary_baseline_sd": (
            baseline["binary_sd"]
        ),
    }

    rank_table["original_top_k"] = (
        rank_table["Feature"].isin(
            original_features
        )
    )

    rank_table["ad_top_k"] = (
        rank_table["Feature"].isin(
            ad_features
        )
    )

    return rank_table, repeat_result


# ============================================================
# Summary helpers
# ============================================================


def _mcse(values: pd.Series) -> float:
    values = values.dropna()

    if len(values) < 2:
        return np.nan

    return float(
        values.std(ddof=1)
        / np.sqrt(len(values))
    )


def _win_rate(values: pd.Series) -> float:
    values = values.dropna()

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values > 0)
    )


def summarize_rankings(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize ranks across repetitions.
    """
    summary = (
        rankings
        .groupby(
            [
                "Feature",
                "Type",
            ],
            as_index=False,
        )
        .agg(
            original_rank_mean=(
                "Rank_Original",
                "mean",
            ),
            original_rank_sd=(
                "Rank_Original",
                "std",
            ),
            ad_rank_mean=(
                "Rank_AD_DIFFI",
                "mean",
            ),
            ad_rank_sd=(
                "Rank_AD_DIFFI",
                "std",
            ),
            rank_change_mean=(
                "Rank_Change",
                "mean",
            ),
            rank_change_sd=(
                "Rank_Change",
                "std",
            ),
            original_top_k_rate=(
                "original_top_k",
                "mean",
            ),
            ad_top_k_rate=(
                "ad_top_k",
                "mean",
            ),
        )
    )

    return summary


def summarize_metrics(
    repeat_results: pd.DataFrame,
    n_repeats: int,
) -> pd.DataFrame:
    """
    Summarize downstream metrics and paired differences.
    """
    records = []

    metric_specs = [
        (
            "IF AUC",
            "if_auc_original",
            "if_auc_ad",
            "if_auc_difference",
        ),
        (
            "Logistic Regression AUC",
            "lr_auc_original",
            "lr_auc_ad",
            "lr_auc_difference",
        ),
    ]

    for (
        metric_name,
        original_column,
        ad_column,
        difference_column,
    ) in metric_specs:
        differences = repeat_results[
            difference_column
        ]

        records.append(
            {
                "metric": metric_name,
                "original_mean": (
                    repeat_results[
                        original_column
                    ].mean()
                ),
                "original_sd": (
                    repeat_results[
                        original_column
                    ].std(ddof=1)
                ),
                "ad_diffi_mean": (
                    repeat_results[
                        ad_column
                    ].mean()
                ),
                "ad_diffi_sd": (
                    repeat_results[
                        ad_column
                    ].std(ddof=1)
                ),
                "paired_difference_mean": (
                    differences.mean()
                ),
                "paired_difference_sd": (
                    differences.std(ddof=1)
                ),
                "paired_difference_mcse": (
                    differences.std(ddof=1)
                    / np.sqrt(n_repeats)
                ),
                "ad_diffi_win_rate": (
                    _win_rate(differences)
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Main benchmark runner
# ============================================================


def run_benchmark(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    config: BenchmarkConfig,
    diffi_func_orig: Callable,
    diffi_func_ad: Callable,
):
    """
    Run the complete paired benchmark.
    """
    ranking_tables = []
    repeat_records = []

    for repeat in range(config.n_repeats):
        rank_table, repeat_result = (
            run_one_repeat(
                X=X,
                y=y,
                feature_names=feature_names,
                feature_types=feature_types,
                config=config,
                diffi_func_orig=diffi_func_orig,
                diffi_func_ad=diffi_func_ad,
                repeat=repeat,
            )
        )

        ranking_tables.append(rank_table)
        repeat_records.append(repeat_result)

    rankings = pd.concat(
        ranking_tables,
        ignore_index=True,
    )

    repeat_results = pd.DataFrame(
        repeat_records
    )

    rank_summary = summarize_rankings(
        rankings
    )

    metric_summary = summarize_metrics(
        repeat_results=repeat_results,
        n_repeats=config.n_repeats,
    )

    rank_correlation, rank_pvalue = (
        spearmanr(
            rankings["Rank_Original"],
            rankings["Rank_AD_DIFFI"],
        )
    )

    mean_abs_rank_change = float(
        rankings["Rank_Change"]
        .abs()
        .mean()
    )

    return {
        "rankings_by_repeat": rankings,
        "rank_summary": rank_summary,
        "repeat_results": repeat_results,
        "metric_summary": metric_summary,
        "spearman_rank_correlation": (
            float(rank_correlation)
        ),
        "spearman_rank_pvalue": (
            float(rank_pvalue)
        ),
        "mean_abs_rank_change": (
            mean_abs_rank_change
        ),
        "config": asdict(config),
    }


# ============================================================
# Export helpers
# ============================================================


def save_benchmark_results(
    results: Dict[str, Any],
    output_dir: str | Path,
    dataset_name: str,
):
    """
    Save all benchmark outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results["rankings_by_repeat"].to_csv(
        output_dir
        / f"{dataset_name}_rankings_by_repeat.csv",
        index=False,
    )

    results["rank_summary"].to_csv(
        output_dir
        / f"{dataset_name}_rank_summary.csv",
        index=False,
    )

    results["repeat_results"].to_csv(
        output_dir
        / f"{dataset_name}_repeat_results.csv",
        index=False,
    )

    results["metric_summary"].to_csv(
        output_dir
        / f"{dataset_name}_metric_summary.csv",
        index=False,
    )

    metadata = {
        "dataset": dataset_name,
        "config": results["config"],
        "spearman_rank_correlation": (
            results[
                "spearman_rank_correlation"
            ]
        ),
        "spearman_rank_pvalue": (
            results[
                "spearman_rank_pvalue"
            ]
        ),
        "mean_abs_rank_change": (
            results[
                "mean_abs_rank_change"
            ]
        ),
    }

    with open(
        output_dir
        / f"{dataset_name}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )
